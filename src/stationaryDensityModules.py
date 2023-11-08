#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This file is the module to compute stationary density of a given model.
It is independent from mfr.modelSoln, although mfr.modelSoln depends on this
module.

Questions/suggestions: please contact

Joe Huang:       jhuang12@uchicago.edu
Paymon Khorrami: paymon@uchicago.edu
Fabrice Tourre:  fabrice@uchicago.edu

"""

from __future__ import division, print_function, absolute_import



from numba import jit
import numpy as np
from scipy.sparse import csc_matrix, identity
from scipy.sparse.linalg import factorized
from scipy.interpolate import RegularGridInterpolator
import warnings
from pyMKL import pardisoSolver

######################################################
############## Stationary Density Module #############
######################################################

## Auxiliary functions
def getStateMatInfo(stateMat):

    ###################################################
    ##################### Input #######################
    ## stateMat: a numpy matrix of the state space ####
    ###################################################
    ###################################################

    ###########################################################################
    ## This function gets information, such as upper limits and lower limits, #
    ## of the state space. It is used to prepare for the creation of the      #
    ## Feynman Kac matrix using numba.                                        #
    ###########################################################################


    upperLims = np.max(stateMat,0)
    lowerLims = np.min(stateMat,0)
    S = np.size(stateMat,0)
    N = np.size(stateMat,1)
    dVec = np.matrix(np.zeros([1,N]))
    increVec = np.matrix(np.zeros([1,N]))

    for i in range(1,S):
        for n in range(0,N):
            diff = stateMat[i,n] - stateMat[i-1,n]
            if (diff > 0 and dVec[0,n] == 0 and increVec[0,n] == 0):
                dVec[0,n] = diff
                increVec[0,n] = i
    return upperLims, lowerLims, S, N, dVec, increVec

def processInputs(stateMat, model, bc):
    ##############################################
    ## This function processes dictionary model ##
    ##############################################

    ## stateMat is either a tuple or a np.matrix that contains the grid
    ## model is a dictionary that has the drifts and vols
    ## bc is a dictionary that contains the boundary conditions
    ## Step 0: Convert stateMat into a matrix
    if isinstance(stateMat, tuple):
        stateGrid = np.meshgrid(*stateMat,  indexing='ij'); stateGrid = [np.matrix(x.flatten()).T for x in stateGrid]
        stateGrid = np.concatenate(stateGrid, axis = 1)
    else:
        stateGrid = stateMat

    dimsGrid = stateGrid.shape ## dimensions of the grid
    modelOut = {'sigmaX':[0] * dimsGrid[1]}

    ## Step 1: Check model['muX']

    #### model['muX'] is either a matrix or a function
    if not type(model['muX']) is np.matrixlib.defmatrix.matrix:
        ### The user did not input numerical values. Try to treat it as
        ### function handles.
        try:
            testOutput = np.matrix(model['muX'](stateGrid))
        except ValueError:
            print('model[\'muX\'] has to be either a function or a numpy matrix.')

        if not (testOutput.shape == dimsGrid):
            raise ValueError('The dimensions of the output of model[\'muX\'] are incorrect.'\
            ' It has to be k x ' + str(int(dimsGrid[1])) + ', where k is the number of rows of the input.')
        else:
            modelOut['muX'] = testOutput
    else:
        if not (model['muX'].shape == dimsGrid):
            raise ValueError('The dimensions of model[\'muX\'], if a matrix, must be the same as'\
            ' the grid, which is ' + str(dimsGrid))
        else:
            modelOut['muX'] = model['muX']

    ## Step 2: Check model['sigmaX']

    if not (len(model['sigmaX']) == dimsGrid[1]):
        raise ValueError('model[\'sigmaX\'] has to be a list of ' + str(dimsGrid[1]) + ' element(s).')

    for n in range(len(model['sigmaX'])):
        if not type(model['sigmaX'][n]) is np.matrixlib.defmatrix.matrix:
            try:
                testOutput = np.matrix(model['sigmaX'][n](stateGrid))
            except ValueError:
                print('Element ' + str(n) + 'of model[\'sigmaX\'] is neither a function nor a matrix.')

            if not (testOutput.shape[0] == dimsGrid[0]):
                raise ValueError('The dimensions of the output of model[\'sigmaX\'][' + str(n) + '] are incorrect.'\
                ' It has to be k x m, where k is the number of rows of the input and m is the number of shocks.')
            else:
                modelOut['sigmaX'][n] = testOutput
        else:
            if not (model['sigmaX'][n].shape[0] == dimsGrid[0]):
                raise ValueError('The number of rows of model[\'sigmaX\'][' + str(n) + '], if a matrix, must be the same as'\
                ' the total number of grid points, which is ' + str(dimsGrid[0])) + '.'
            else:
                modelOut['sigmaX'][n] = model['sigmaX'][n].copy()

    return stateGrid, modelOut

def convertLogW(inputMat, stateGridW, stateMatLogW):

    #########################Inputs#####################################
    #### inputMat :    reshaped matrix                                 #
    #### stateGridW:   grid to which data will be converted (matrix)   #
    #### stateMatLogW: grid from which data will be converted (tuple)  #
    ####################################################################


    if len(inputMat.shape) == 1:
        nCols = 1;
        interp  = RegularGridInterpolator(stateMatLogW, inputMat.reshape([x.shape[0] for x in stateMatLogW], order ='F'), bounds_error = False, fill_value = None)
        inputMat = interp(np.array(stateGridW))

    else:
        nCols = inputMat.shape[1];

        for i in range(nCols):
            interp  = RegularGridInterpolator(stateMatLogW, inputMat[:,i].reshape([x.shape[0] for x in stateMatLogW], order ='F'), bounds_error = False, fill_value = None)
            inputMat[:,i] = interp(np.array(stateGridW))

    return inputMat.copy()


def computeDent(stateMat, model, bc = {'natural': True}, usePardiso = False, iparms = {}, \
explicit = False, dt = 0.1, tol = 1e-5, maxIters = 100000, verb = False, betterCP = True):

    ############################################################################################
    #########                                Compute Stationary Density                 ########
    ############################################################################################

    ## This function computes the stationary density in two ways.
    #### Method (1): It will try to solve the eigenvector problem by looking for
    ####             an eigenvector of the submatrix of the Feynman Kac matrix
    ####             with the same sign.
    #### Method (2): It will solve the Fokker Planck system using explicit scheme.
    ####             The user should try to use method (1) first because it is more efficient.


    ##### Step 1:Construct the finite difference scheme matrix of Feynmann Kac Formula####
    stateGrid, modelOut = processInputs(stateMat, model, bc)
    upperLims, lowerLims, S, N, dVec, increVec = getStateMatInfo(stateGrid)
    vals, colInd, rowInd, atBounds, corners = createMatrixDent(upperLims, lowerLims, S, N, dVec, increVec, stateGrid, modelOut,bc, betterCP)

    FKmat = (csc_matrix((vals, (rowInd, colInd)), shape=(S, S))) ## Feynman Kac matrix

    ##### Step 2: Recognizing the Feyman Kac matrix is the transpose of Fokker Planck matrix#####
    linSys    = (FKmat.T).tocsc()
    linSysSub = linSys[1:,1:]
    dentPrime = None
    f_t       = None
    ##### Step 3: Solve the equation####
    if not explicit:
        ## Method (1)
        rhs = -linSys[1:,0]
        x   = None
        if not usePardiso:
            solve = factorized(linSysSub)
            x     = solve(rhs.toarray())
        else:
            pSolve = pardisoSolver(linSysSub.tocsr(), mtype=11)
            if bool(iparms):
                for k, v in iparms.items(): pSolve.iparm[k] = v
            pSolve.factor()
            x      = pSolve.solve(rhs.toarray()).reshape(S - 1, 1)
        dentPrime = np.insert(x,0,1)
    else:
        ## Method (2)
        f_0       = np.matrix(np.full( (S, 1), 1.0 / (S))) ## using uniform distribution as the guess
        f_t       = np.zeros(f_0.shape)
        I         = identity(S, dtype='float', format='dia')
        for i in range(maxIters):
            if (i == 0) or (np.max(np.abs(f_t - f_0)) / dt > tol):
                ## Tolerance not met. Keep iterating
                time_derivs = np.max(np.abs(f_t - f_0)) / dt
                if i > 0:
                    f_0 = f_t.copy()
                if verb:
                    print('Iteration: ', i,'. Time derivative is ', time_derivs, sep = '')
                f_t = (I + linSys * dt) * f_0

            else:
                if verb:
                    print('Tolerance met.')
                break
        f_t = f_t / sum(f_t) ## normalize
    dent = f_t if explicit else dentPrime / dentPrime.sum()

    ## Check the stationary density. There is a chance that both method could fail.
    ## If so, warn the user and make suggestions.
    posSum = np.sum(dent[dent > 0])
    negSum = np.sum(dent[dent < 0])
    if np.min([posSum, np.abs(negSum)]) > 0.02:
        if not explicit:
            ## Method (1) did not work; could due to the Barles & Souganidis sufficient
            ## condition not satisfied. Suggest the user to use the explicit method instead.

            msg = 'Stationary density computed could be degenerate.'\
            ' We suggest that you use the explicit scheme by setting explicit = True'
        else:
            ## Method (2) did not work. Could be that the time step is too large.
            ## Make a suggestion based on the CFL condition.

            suggestDT = np.min([np.power(x,2) for x in dVec]) / 10.0
            dist      = '%e' % suggestDT
            dist      = int(dist.partition('-')[2]) ## find the nearest nonzero integer
            suggestDT = np.round(suggestDT, dist + 1 )
            msg = 'Stationary density computed could be degenerate.'\
                ' We suggest that you reduce the time step size dt to ' + str(suggestDT) + '.'
        warnings.warn(msg)

    return dent, FKmat, stateGrid


def createMatrixDent(upperLims, lowerLims, S, N, dVec, increVec, stateMat, model,bc, betterCP):

    #################################################################
    #########Prepare inputs for the jit nopython function############
    #################################################################

    ### The user can either input numerical values for arguments model['sigmaX'],
    ### and model['muX'] or lambda functions (e.g. interpolants). If the user
    ### decides to use numerical values, the user must input stateMat as a matrix
    ### that contains all the grid points. If not, the user must input stateMat
    ### as a tuple that contains the discretization vectors of each state
    ### variable.

    ### Eventually, this function will convert user inputs into numerical values
    ### for numba.


    muX = model['muX']
    sigmaX = tuple(model['sigmaX'])

    if not bc['natural']:
        a0 = bc['a0']
        first = bc['first']
        second = bc['second']
        third = bc['third']
    else:
        a0 = np.matrix([0]); first = np.matrix([0]);
        second = np.matrix([0]); third = np.matrix([0]);

    natural = bc['natural']

    firstCoefs = np.matrix(np.zeros((S,N)))
    secondCoefs = np.matrix(np.zeros((S,N)))
    for n in range(0,N):
        firstCoefs[:,n] = muX[:,n]

    for n in range(0,N):
        secondCoefs[:,n] = 0.5 * np.multiply(sigmaX[n], sigmaX[n]).sum(axis=1)

    ######################################
    #########Execute function#############
    ######################################
    vals, colInd, rowInd, atBounds, corners = createMatrixInnerDensity(stateMat,
                upperLims, lowerLims, S, N, dVec, increVec, sigmaX,
                 firstCoefs, secondCoefs, a0, first, second, third, natural, betterCP)
    return vals, colInd, rowInd, atBounds, corners





@jit(nopython=True)
def createMatrixInnerDensity(stateMat, upperLims, lowerLims, S, N, dVec, increVec, sigmaX,
                 firstCoefs, secondCoefs, a0, first, second, third, natural, betterCP):


    rowInd   = [];
    colInd   = [];
    vals     = [];
    corners  = [];
    atBounds = [];

    atBoundsInd = list(range(N))

    for i in range(0,S):

        atBound = False
        corner = False

        ######################
        ###Check boundaries###
        ######################

        n = N - 1

        ################Start while loop##################
        while n >= 0:
            atBoundsInd[n] = 1
            ####################################
            ####Check if it is at boundary######
            ####################################

            if ( abs((stateMat[i,n] - upperLims[0,n])) < (dVec[0,n] / 2.0) ) or ( abs((stateMat[i,n] - lowerLims[0,n])) < (dVec[0,n] / 2.0) ):
                atBound = True

                ####Check if it's at one of the corners
                n_sub = n - 1

                while n_sub >= 0:

                    if ( abs((stateMat[i,n_sub] - upperLims[0,n_sub])) < (dVec[0,n_sub] / 2.0) ) or ( abs((stateMat[i,n_sub] - lowerLims[0,n_sub])) < (dVec[0,n_sub] / 2.0) ):
                        corner = True
                    ##################Reduce n_sub by 1 so that the inner while loop can end#################
                    n_sub = n_sub - 1
                    #########################################################################################

            ###################################################
            ####Check if it is at upper or lower boundary######
            ###################################################

            if ( abs((stateMat[i,n] - upperLims[0,n])) < (dVec[0,n] / 2.0) ): ####Upper boundary
                atBoundsInd[n] = -1


                if ( (not natural) and (not corner) ):

                    ####If not using natural bdries, set as specified
                    vals.append( - first[0,n] / dVec[0,n] + second[0,n] / (dVec[0,n] ** 2) )
                    rowInd.append( i ); colInd.append( i );

                    vals.append( first[0,n] / dVec[0,n] - 2 * second[0,n] / (dVec[0,n] ** 2) )
                    rowInd.append( i ); colInd.append( i - increVec[0,n] );

                    vals.append( second[0,n] / (dVec[0,n] ** 2) )
                    rowInd.append( i ); colInd.append( i - 2 * increVec[0,n] );
                elif (natural ):
                    ####If natural, use natural bdries

                    vals.append( firstCoefs[i,n] / dVec[0,n] + secondCoefs[i,n] / (dVec[0,n] ** 2))
                    rowInd.append( i ); colInd.append( i );

                    vals.append(  - firstCoefs[i ,n] / dVec[0,n] - 2.0 * secondCoefs[i,n] / (dVec[0,n] ** 2))
                    rowInd.append( i ); colInd.append( i - increVec[0,n] );

                    vals.append( secondCoefs[i ,n] / (dVec[0,n] ** 2) )
                    rowInd.append(i); colInd.append( i - 2 * increVec[0,n] );

            elif ( abs((stateMat[i,n] - lowerLims[0,n])) < (dVec[0,n] / 2.0) ): ####Lower boundary
                atBoundsInd[n] = -1

                if ( (not natural) and (not corner)  ):
                    ####If not using natural bdries, set as specified
                    vals.append(  - first[0,n] / dVec[0,n] + second[0,n] / (dVec[0,n] ** 2) )
                    rowInd.append( i ); colInd.append( i );

                    vals.append( first[0,n] / dVec[0,n] - 2 * second[0,n] / (dVec[0,n] ** 2) )
                    rowInd.append( i ); colInd.append( i + increVec[0,n]);

                    vals.append(second[0,n] / (dVec[0,n] ** 2));
                    rowInd.append( i ); colInd.append( i + 2 * increVec[0,n] );
                elif (natural ):
                    ####If natural, use natural bdries
                    vals.append(  - firstCoefs[i,n] / dVec[0,n] + secondCoefs[i,n] / (dVec[0,n] ** 2) )
                    rowInd.append(i); colInd.append(i);

                    vals.append(  firstCoefs[i,n] / dVec[0,n] - 2.0 * secondCoefs[i,n] / (dVec[0,n] ** 2))
                    rowInd.append(i); colInd.append(i + increVec[0,n]);

                    vals.append( secondCoefs[i,n] / (dVec[0,n] ** 2) )
                    rowInd.append( i ); colInd.append( i + 2 * increVec[0,n] );

            ####Reduce n by 1 so that the while loop can end####
            n = n - 1
            ####################################################

        ######End of while loop########

        if (corner):  ####Take care of the corners
            if (not natural):
                # At each corner, we will assume that the corner point is equal
                # to the weighted average of the surrounidng points
                # e.g.,in 2D, phi_t+1(1,1) = 1/2 * (phi_t+1(2,1) + phi_t+1(1,2))

                vals.append( -1.0 )
                rowInd.append( i ); colInd.append( i );

                n = N - 1;
                while n >= 0:


                    ## Check if upper or lower boundary
                    if ( abs((stateMat[i,n] - upperLims[0,n])) < (dVec[0,n] / 2.0) ):
                        vals.append( 1.0  / N )
                        rowInd.append( i ); colInd.append( i - increVec[0,n] );
                    elif ( abs((stateMat[i,n] - lowerLims[0,n])) < (dVec[0,n] / 2.0) ):
                        vals.append( 1.0  / N )
                        rowInd.append( i ); colInd.append( i + increVec[0,n] );

                    ## Reducing n by 1
                    n = n - 1

            corners.append( i)

        if (atBound):
            atBounds.append( i )

        ####################################
        #####handle nonboundary elements####
        ####################################

        n = N - 1

        ##############Start while loop#############

        while n >= 0:
            if ( ((not atBound) and (not natural)) or ( (natural) and (atBoundsInd[n] > 0) ) ):

                #####First derivatives#####
                if (not (firstCoefs[i,n] == 0)):

                    ### upwinding
                    vals.append( ( -firstCoefs[i,n] * (firstCoefs[i,n] > 0) + firstCoefs[i,n] * (firstCoefs[i,n] < 0) ) / dVec[0,n] )
                    rowInd.append( i ); colInd.append( i );

                    vals.append(  (firstCoefs[i ,n] * (firstCoefs[i,n] > 0)) / dVec[0,n])
                    rowInd.append(i); colInd.append( i + increVec[0,n]);

                    vals.append(  (-firstCoefs[i,n] * (firstCoefs[i,n] < 0)) / dVec[0,n] )
                    rowInd.append( i ); colInd.append( i - increVec[0,n] );

                #####Second derivatives#####
                if (not (secondCoefs[i,n] == 0)):
                    vals.append( (-2.0 * secondCoefs[i,n]) / (dVec[0,n] ** 2) )
                    rowInd.append( i ); colInd.append( i );

                    vals.append( (secondCoefs[i,n]) / (dVec[0,n] ** 2) )
                    rowInd.append( i ); colInd.append( i + increVec[0,n] );

                    vals.append( (secondCoefs[i,n]) / (dVec[0,n] ** 2) )
                    rowInd.append( i ); colInd.append( i - increVec[0,n] );

            #####Cross partials#####
            if ( ((not atBound) and (not natural)) or ( (natural) ) ):

                n_sub = n - 1
                idx   = 0
                while n_sub >= 0:
                    if not betterCP:
                        ## Using the naive way of approximation cross partials
                        idx = int( i + increVec[0,n] * (1 + \
                              (-1) * ( abs((stateMat[i,n] - upperLims[0,n])) < (dVec[0,n] / 2.0) )  + \
                              (1) * ( abs((stateMat[i,n] - lowerLims[0,n])) < (dVec[0,n] / 2.0) )  )  + \
                              increVec[0,n_sub] * (1 + \
                              (-1) * ( abs((stateMat[i,n_sub] - upperLims[0,n_sub])) < (dVec[0,n_sub] / 2.0) )  + \
                              (1) * ( abs((stateMat[i,n_sub] - lowerLims[0,n_sub])) < (dVec[0,n_sub] / 2.0) ) ) )

                        vals.append(np.multiply(sigmaX[n][i,:].transpose(), sigmaX[n_sub][i,:]).sum()/ (4 * dVec[0,n] * dVec[0,n_sub]))
                        rowInd.append( i ); colInd.append( idx );

                        idx = int(i - increVec[0,n] * (1 + \
                              -1 * ( (-1) * ( abs((stateMat[i,n] - upperLims[0,n])) < (dVec[0,n] / 2.0) )  + \
                              (1) * ( abs((stateMat[i,n] - lowerLims[0,n])) < (dVec[0,n] / 2.0) ) ) ) \
                              - increVec[0,n_sub] * (1 + \
                              -1 *  ( (-1) * ( abs((stateMat[i,n_sub] - upperLims[0,n_sub])) < (dVec[0,n_sub] / 2.0))  + \
                              (1) * ( abs((stateMat[i,n_sub] - lowerLims[0,n_sub])) < (dVec[0,n_sub] / 2.0)) ) ))

                        vals.append( np.multiply(sigmaX[n][i,:].transpose(), sigmaX[n_sub][i,:]).sum() / (4 * dVec[0,n] * dVec[0,n_sub]))
                        rowInd.append( i ); colInd.append( idx );

                        idx = int(i + increVec[0,n] * (1 + \
                              (-1) * ( abs((stateMat[i,n] - upperLims[0,n])) < (dVec[0,n] / 2.0))  + \
                              (1) * ( abs((stateMat[i,n] - lowerLims[0,n])) < (dVec[0,n] / 2.0) ) ) \
                              - increVec[0,n_sub] * (1 + \
                              -1 * ( (-1) * ( abs((stateMat[i,n_sub] - upperLims[0,n_sub])) < (dVec[0,n_sub] / 2.0))  + \
                              (1) * ( abs((stateMat[i,n_sub] - lowerLims[0,n_sub])) < (dVec[0,n_sub] / 2.0)) ) ))
                        vals.append( -1.0 * np.multiply(sigmaX[n][i,:].transpose(), sigmaX[n_sub][i,:]).sum() / (4 * dVec[0,n] * dVec[0,n_sub]))
                        rowInd.append( i ); colInd.append(idx);

                        idx = int(i - increVec[0,n] * (1 + \
                              -1 * ( (-1) * ( abs((stateMat[i,n] - upperLims[0,n])) < (dVec[0,n] / 2.0) ) + \
                              (1) * ( abs((stateMat[i,n] - lowerLims[0,n])) < (dVec[0,n] / 2.0) )) ) \
                              + increVec[0,n_sub] * (1 + \
                              (-1) * ( abs((stateMat[i,n_sub] - upperLims[0,n_sub])) < (dVec[0,n_sub] / 2.0) ) + \
                              (1) * ( abs((stateMat[i,n_sub] - lowerLims[0,n_sub])) < (dVec[0,n_sub] / 2.0) ) ) )
                        vals.append( -1.0 * np.multiply(sigmaX[n][i,:].transpose(), sigmaX[n_sub][i,:]).sum() / (4 * dVec[0,n] * dVec[0,n_sub]))
                        rowInd.append(i); colInd.append(idx);
                    else:
                        ## Positive elements
                        idx = int( i + increVec[0,n] * (1 + \
                              (-1) * ( abs((stateMat[i,n] - upperLims[0,n])) < (dVec[0,n] / 2.0) ) + \
                              (1) * ( abs((stateMat[i,n] - lowerLims[0,n])) < (dVec[0,n] / 2.0) ) ) + increVec[0,n_sub] * (1 + \
                              (-1) * ( abs((stateMat[i,n_sub] - upperLims[0,n_sub])) < (dVec[0,n_sub] / 2.0))  + \
                              (1) * ( abs((stateMat[i,n_sub] - lowerLims[0,n_sub])) < (dVec[0,n_sub] / 2.0))  ) )

                        vals.append(np.multiply(sigmaX[n][i,:].transpose(), sigmaX[n_sub][i,:]).sum()/ (2.0 * dVec[0,n] * dVec[0,n_sub]))
                        rowInd.append( i ); colInd.append( idx );

                        idx = int(i - increVec[0,n] * (1 + \
                              -1 * ( (-1) * ( abs((stateMat[i,n] - upperLims[0,n])) < (dVec[0,n] / 2.0)) + \
                              (1) * ( abs((stateMat[i,n] - lowerLims[0,n])) < (dVec[0,n] / 2.0)) ) ) - increVec[0,n_sub] * (1 + \
                              -1 * ( (-1) * ( abs((stateMat[i,n_sub] - upperLims[0,n_sub])) < (dVec[0,n_sub] / 2.0)) + \
                              (1) * ( abs((stateMat[i,n_sub] - lowerLims[0,n_sub])) < (dVec[0,n_sub] / 2.0)) ) ))

                        vals.append( np.multiply(sigmaX[n][i,:].transpose(), sigmaX[n_sub][i,:]).sum() / (2.0 * dVec[0,n] * dVec[0,n_sub]))
                        rowInd.append( i ); colInd.append( idx );

                        idx = int(i + increVec[0,n] * ( \
                              (-1) * ( abs((stateMat[i,n] - upperLims[0,n])) < (dVec[0,n] / 2.0) ) + \
                              (1) * ( abs((stateMat[i,n] - lowerLims[0,n])) < (dVec[0,n] / 2.0)) ) + increVec[0,n_sub] * ( \
                              (-1) * ( abs((stateMat[i,n_sub] - upperLims[0,n_sub])) < (dVec[0,n_sub] / 2.0) ) + \
                              (1) * ( abs((stateMat[i,n_sub] - lowerLims[0,n_sub])) < (dVec[0,n_sub] / 2.0) ) ) )
                        vals.append( 2.0 * np.multiply(sigmaX[n][i,:].transpose(), sigmaX[n_sub][i,:]).sum() / (2.0 * dVec[0,n] * dVec[0,n_sub]))
                        rowInd.append( i ); colInd.append( idx );

                        ## Negative elements
                        idx = int( i + increVec[0,n] * (1 + \
                              (-1) * ( abs((stateMat[i,n] - upperLims[0,n])) < (dVec[0,n] / 2.0))  + \
                              (1) * ( abs((stateMat[i,n] - lowerLims[0,n])) < (dVec[0,n] / 2.0))  ) + increVec[0,n_sub] * ( \
                              (-1) * ( abs((stateMat[i,n_sub] - upperLims[0,n_sub])) < (dVec[0,n_sub] / 2.0) ) + \
                              (1) * ( abs((stateMat[i,n_sub] - lowerLims[0,n_sub])) < (dVec[0,n_sub] / 2.0) ) ) )

                        vals.append(-np.multiply(sigmaX[n][i,:].transpose(), sigmaX[n_sub][i,:]).sum()/ (2.0 * dVec[0,n] * dVec[0,n_sub]))
                        rowInd.append( i ); colInd.append( idx );

                        idx = int(i + increVec[0,n_sub] * (1 + \
                              (-1) * ( abs((stateMat[i,n_sub] - upperLims[0,n_sub])) < (dVec[0,n_sub] / 2.0)) + \
                              (1) * ( abs((stateMat[i,n_sub] - lowerLims[0,n_sub])) < (dVec[0,n_sub] / 2.0))  ) + increVec[0,n] * ( \
                              (-1) * ( abs((stateMat[i,n] - upperLims[0,n])) < (dVec[0,n] / 2.0) ) + \
                              (1) * ( abs((stateMat[i,n] - lowerLims[0,n])) < (dVec[0,n] / 2.0) ) ) )

                        vals.append( -np.multiply(sigmaX[n][i,:].transpose(), sigmaX[n_sub][i,:]).sum() / (2.0 * dVec[0,n] * dVec[0,n_sub]))
                        rowInd.append( i ); colInd.append( idx );

                        idx = int(i - increVec[0,n] * (1 + \
                              -1 *( (-1) * ( abs((stateMat[i,n] - upperLims[0,n])) < (dVec[0,n] / 2.0))  + \
                              (1) * ( abs((stateMat[i,n] - lowerLims[0,n])) < (dVec[0,n] / 2.0)) ) ) + increVec[0,n_sub] * ( \
                              (-1) * ( abs((stateMat[i,n_sub] - upperLims[0,n_sub])) < (dVec[0,n_sub] / 2.0) ) + \
                              (1) * ( abs((stateMat[i,n_sub] - lowerLims[0,n_sub])) < (dVec[0,n_sub] / 2.0) ) ) )

                        vals.append( -np.multiply(sigmaX[n][i,:].transpose(), sigmaX[n_sub][i,:]).sum() / (2.0 * dVec[0,n] * dVec[0,n_sub]))
                        rowInd.append( i ); colInd.append( idx );

                        idx = int(i - increVec[0,n_sub] * (1 + \
                              -1 * ((-1) * ( abs((stateMat[i,n_sub] - upperLims[0,n_sub])) < (dVec[0,n_sub] / 2.0))  + \
                              (1) * ( abs((stateMat[i,n_sub] - lowerLims[0,n_sub])) < (dVec[0,n_sub] / 2.0)) ) ) + increVec[0,n] * ( \
                              (-1) * ( abs((stateMat[i,n] - upperLims[0,n])) < (dVec[0,n] / 2.0) ) + \
                              (1) * ( abs((stateMat[i,n] - lowerLims[0,n])) < (dVec[0,n] / 2.0) ) ) )

                        vals.append( -np.multiply(sigmaX[n][i,:].transpose(), sigmaX[n_sub][i,:]).sum() / (2.0 * dVec[0,n] * dVec[0,n_sub]))
                        rowInd.append( i ); colInd.append( idx );

                    n_sub = n_sub - 1
            ####Reduce n by 1 so that the while loop can end####
            n = n - 1
            ####################################################
    return np.array(vals), np.array(colInd), np.array(rowInd), np.array(atBounds), np.array(corners)
