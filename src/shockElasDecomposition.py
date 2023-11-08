#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file is the module to compute shock elasticities of a given model.
It is independent from mfr.modelSoln, although mfr.modelSoln depends on this
module.

Questions/suggestions: please contact

Joe Huang:       jhuang12@uchicago.edu
Paymon Khorrami: paymon@uchicago.edu
Fabrice Tourre:  fabrice@uchicago.edu
"""

from __future__ import division, print_function, absolute_import


import itertools

from numba import jit, prange
import numpy as np
import collections
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import factorized
import math
from scipy.interpolate import RegularGridInterpolator, interpn
import copy
from pyMKL import pardisoSolver
import warnings

def getStateMatInfo(stateMat):

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
    stateGrid = np.meshgrid(*stateMat,  indexing='ij'); stateGrid = [np.matrix(x.T.flatten(order = 'C')).T for x in stateGrid]
    stateGrid = np.concatenate(stateGrid, axis = 1)
    dimsGrid = stateGrid.shape ## dimensions of the grid

    ## Step 1: Check model['muX']
    try:
        testOutput = np.matrix(model['muX'](stateGrid[0,:]))
    except ValueError:
        print('model[\'muX\'] has to be either a function or a numpy matrix.')

    if not (testOutput.shape[1] == dimsGrid[1]):
        raise ValueError('The dimensions of the output of model[\'muX\'] are incorrect.'\
        ' It has to be k x ' + str(int(dimsGrid[1])) + ', where k is the number of rows of the input.')

    ## Step 2: Check model['sigmaX']

    if not (len(model['sigmaX']) == dimsGrid[1]):
        raise ValueError('model[\'sigmaX\'] has to be a list of ' + str(dimsGrid[1]) + ' element(s).')

    for n in range(len(model['sigmaX'])):
        try:
            testOutput = np.matrix(model['sigmaX'][n](stateGrid[0,:]))
        except ValueError:
            print('Element ' + str(n) + 'of model[\'sigmaX\'] is neither a function nor a matrix.')

        '''
        if not (testOutput.shape[1] == dimsGrid[1]):
            raise ValueError('The dimensions of the output of model[\'sigmaX\'][' + str(n) + '] are incorrect.'\
            ' It has to be k x m, where k is the number of rows of the input and m is the number of shocks.')
        '''
    return stateGrid

@jit(nopython=True)
def createMatrixInner(stateMat, upperLims, lowerLims, S, N, dVec, increVec, sigmaX,
                 levelCoefs, firstCoefs, secondCoefs,
                 T, dt, a0, level, first, second, third,
                 natural, betterCP):

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
                    vals.append( level[0,n] + first[0,n] / dVec[0,n] )
                    rowInd.append( i ); colInd.append( i );
            
                    vals.append( -first[0,n] / dVec[0,n] )
                    rowInd.append( i ); colInd.append( i - increVec[0,n] );


                elif (natural):
                    ####If natural, use natural bdries

                    vals.append(firstCoefs[i,n] / dVec[0,n] + secondCoefs[i,n] / (dVec[0,n] ** 2))
                    rowInd.append( i ); colInd.append( i );

                    vals.append( - firstCoefs[i,n] / dVec[0,n] - 2.0 * secondCoefs[i,n] / (dVec[0,n] ** 2))
                    rowInd.append( i ); colInd.append( i - increVec[0,n] );

                    vals.append( secondCoefs[i,n] / (dVec[0,n] ** 2) )
                    rowInd.append(i); colInd.append( i - 2 * increVec[0,n] );


            elif ( abs((stateMat[i,n] - lowerLims[0,n])) < (dVec[0,n] / 2.0) ): ####Lower boundary
                atBoundsInd[n] = -1

                if ( (not natural) and (not corner) ):
                    
                    ####If not using natural bdries, set as specified
                    vals.append( level[0,n] - first[0,n] / dVec[0,n]  )
                    rowInd.append( i ); colInd.append( i );

                    vals.append( first[0,n] / dVec[0,n]  )
                    rowInd.append( i ); colInd.append( i + increVec[0,n]);


                elif (natural):
                    ####If natural, use natural bdries
                    vals.append( - firstCoefs[i,n] / dVec[0,n] + secondCoefs[i,n] / (dVec[0,n] ** 2) )
                    rowInd.append(i); colInd.append(i);

                    vals.append( firstCoefs[i,n] / dVec[0,n] - 2.0 * secondCoefs[i,n] / (dVec[0,n] ** 2))
                    rowInd.append(i); colInd.append(i + increVec[0,n]);

                    vals.append( secondCoefs[i,n] / (dVec[0,n] ** 2) )
                    rowInd.append( i ); colInd.append( i + 2 * increVec[0,n] );
            ####Reduce n by 1 so that the while loop can end####
            n = n - 1
            ####################################################
        ######End of while loop########

        if (corner):  ####Take care of the corners
            if (not natural):

                vals.append( -1.0)
                rowInd.append( i ); colInd.append( i );

                for n in range(N):
                    if ( abs((stateMat[i,n] - upperLims[0,n])) < (dVec[0,n] / 2.0) ): ####Upper boundary
                        vals.append( 1.0 / N)
                        rowInd.append( i ); colInd.append( i - increVec[0,n]);
                    elif ( abs((stateMat[i,n] - lowerLims[0,n])) < (dVec[0,n] / 2.0) ):
                        vals.append( 1.0 / N)
                        rowInd.append( i ); colInd.append( i + increVec[0,n]);

            corners.append( i)

        if (atBound):
            atBounds.append( i )

        if ( (not atBound) and (not natural) ):
            vals.append( levelCoefs[i,0] )
            rowInd.append( i ); colInd.append( i );
        elif (natural):
            vals.append(levelCoefs[i,0])
            rowInd.append(i); colInd.append( i );

        ####################################
        #####handle nonboundary elements####
        ####################################

        n = N - 1

        ##############Start while loop#############

        while n >= 0:

            if ( ((not atBound) and (not natural)) or ( (natural) and (atBoundsInd[n] > 0) ) ):
                #####First derivatives#####
                if (not (firstCoefs[i,n] == 0)):
                    vals.append( (-firstCoefs[i,n] * (firstCoefs[i,n] > 0) + firstCoefs[i,n] * (firstCoefs[i,n] < 0)) / dVec[0,n] )
                    rowInd.append( i ); colInd.append( i );

                    vals.append((firstCoefs[i,n] * (firstCoefs[i,n] > 0)) / dVec[0,n])
                    rowInd.append(i); colInd.append( i + increVec[0,n]);

                    vals.append( (-firstCoefs[i,n] * (firstCoefs[i,n] < 0)) / dVec[0,n] )
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
    vals = np.array(vals) * dt

    return vals, np.array(colInd), np.array(rowInd), np.array(atBounds), np.array(corners)


def createMatrix(upperLims, lowerLims, S, N, dVec, increVec, stateMat, model,bc, betterCP):

    #################################################################
    #########Prepare inputs for the jit nopython function############
    #################################################################

    ####Convert function arguments for jit nopython mode

    muX = np.matrix(model['muX'](stateMat))
    sigmaX = []
    for n in range(N):
        sigmaX.append( np.matrix(model['sigmaX'][n](stateMat)) )
    sigmaX = tuple(sigmaX)

    T = model['T']
    dt = model['dt']
    muC = np.matrix(model['muC'](stateMat))*0.0
    sigmaC = np.matrix(model['sigmaC'](stateMat))*0.0

    if not bc['natural']:
        a0 = bc['a0']
        level = np.matrix(np.repeat(bc['level'],N),'d')
        first = bc['first']
        second = np.matrix([0]); 
        third = np.matrix([0]);
    else:
        a0 = np.matrix([0]); first = np.matrix([0]);
        second = np.matrix([0]); third = np.matrix([0]);
        level = np.matrix([0]);

    natural = bc['natural']

    levelCoefs = -1.0 / model['dt'] + muC.reshape([S,1]) + 0.5 * np.transpose(np.matrix((np.linalg.norm(sigmaC, axis=1) ** 2)))

    firstCoefs = np.matrix(np.zeros((S,N)))
    secondCoefs = np.matrix(np.zeros((S,N)))
    for n in range(0,N):
        firstCoefs[:,n] = muX[:,n] + np.multiply(sigmaX[n],sigmaC).sum(axis=1)

    for n in range(0,N):
        secondCoefs[:,n] = 0.5 * np.multiply(sigmaX[n], sigmaX[n]).sum(axis=1)


    ######################################
    #########Execute function#############
    ######################################

    vals, colInd, rowInd, atBounds, corners = createMatrixInner(stateMat,
                upperLims, lowerLims, S, N, dVec, increVec, sigmaX,
                 levelCoefs, firstCoefs, secondCoefs,
                 T, dt, a0, level, first, second, third,
                 natural, betterCP)
    linSys = csc_matrix((vals, (rowInd, colInd)), shape=(S, S))
    return linSys, atBounds, corners


################################################################
############Function to solve linear system through time########
################################################################

def solvePhi(upperLims, lowerLims, S, N, numShocks, dVec, increVec, stateMat, model, bc, usePardiso, iparms, betterCP):

    #####Step 0: Construct linear system####
    linSys, atBounds, corners = createMatrix(upperLims, lowerLims, S, N, dVec, increVec,
                                             stateMat, model, bc, betterCP)
    ####Step 1: Initialize  matrix to store results and factorize####
    phit   = np.zeros([linSys.shape[0], 1+numShocks, model['T']])
    phi0   = np.concatenate([np.ones([S,1]), np.asarray(model['sigmaC'](stateMat))], axis = 1)
    solve  = None
    pSolve = None

    if usePardiso:
        pSolve = pardisoSolver(linSys.tocsr(), mtype=11)
        if bool(iparms):
            for k, v in iparms.items(): pSolve.iparm[k] = v
        pSolve.factor()
    else:
        solve = factorized(linSys)
    for t in range(0,model['T']):
        if (t == 0):
            phit[:,:,t] = phi0

        else:
            phit[:,:,t] =  - phit[:,:,(t-1)]

            ######Take care of the known vector for boundaries#####
            if (not bc['natural']):
                phit[atBounds.astype(int), :, t] = -bc['a0']

            ######Solve the system#####
            if usePardiso:
                phit[:,:,t] = pSolve.solve(phit[:,:,t])
            else:
                phit[:,:,t] = solve(phit[:,:,t])

            if (not bc['natural']):
                ######Adjust corners######
                if N > 1:
                    phit[corners, :, t] = 0.0
    return phit, linSys, atBounds

#################################################################
############       Functions to comptue elasticities     ########
#################################################################

class elas():  ####False class to store elasticities
    pass

def computeElasSub(stateMat, model, bc, x0, usePardiso, iparms, betterCP):

    ############################################################################################
    #########Inner function that gets called twice for shock expo and price respectively########
    ############################################################################################

    #####Step 0: Solve Feymann Kac and initialize preliminaries####
    stateGrid = processInputs(stateMat, model, bc)

    upperLims, lowerLims, S, N, dVec, increVec = getStateMatInfo(stateGrid)
    eps = .0000000001   ###epsilon
    numStarts = x0.shape[0]   ###number of starting points
    numShocks = (1 if len(model['sigmaC'](x0).shape) == 1 else model['sigmaC'](x0).shape[1]) ###number of shocks
    phit, linSys, atBounds = \
    solvePhi(upperLims, lowerLims, S, N, numShocks, dVec, increVec, stateGrid, model, bc, usePardiso, iparms, betterCP)

    ###Set up the points at which we want to interpolate
    x_evals = np.array([ np.matrix(np.zeros([2 * N + 1, N])) for i in range(numStarts)])
    for i in range(numStarts):
        for n in range(0, 2 * (N), 2):
            x_evals[i][n, :] = x0[i,:]
            x_evals[i][n + 1, :] = x0[i,:]
            x_evals[i][n, math.floor(n / 2) ] = x0[i, math.floor(n / 2) ] + eps
            x_evals[i][n + 1, math.floor(n / 2) ] = x0[i, math.floor(n / 2) ] - eps
        x_evals[i][-1,:] = x0[i,:]
    #####Step 1: Compute elasticities (first type) #####

    #####Interpolate for phit at different x_evals

    ###Create arrays to store interpolated results; res: interpolate
    res   = np.array([ np.matrix(np.zeros([x_evals[0].shape[0], model['T']])) for i in range(numStarts)])
    exp1s = np.array([ np.matrix(np.zeros([1, model['T']])) for i in range(numStarts)])
    for i in range( numStarts ):
        for t in range(model['T']):
            F = RegularGridInterpolator(stateMat, phit[:,0,t].reshape([x.shape[0] for x in stateMat], order ='F') )
            res[i][:,t] = F( x_evals[i] ).T
        exp1s[i] = res[i][-1,:]

    #####Step 2: Compute elasticities (second type) #####
    exp2s = np.array(np.zeros([numStarts, numShocks, model['T']]))
    for r in range(numShocks):
        for t in range(model['T']):
            F = RegularGridInterpolator(stateMat, phit[:,r + 1,t].reshape([x.shape[0] for x in stateMat], order ='F') )
            exp2s[:,r,t] = F( x_evals[:,-1,:] )
    secondType = exp2s / exp1s
    firstType = exp1s
    thirdType = exp2s

    return firstType, secondType, thirdType, phit, linSys, atBounds

def computeElas(stateMat, model, bc, x0, usePardiso = False, iparms = {}, betterCP = True):

    ## This is the main function used to compute shock elasticities.

    #############Steo 1: Compute shock exposure elasticities
    expoElas = elas()
    firstType, secondType, thirdType, phit1, linSys1, atBounds = computeElasSub(stateMat, model, bc, x0, usePardiso, iparms, betterCP)
    expoElas.firstType = firstType; expoElas.secondType = secondType; expoElas.thirdType = thirdType

    #############Steo 2: Compute shock cost elasticities

    modelCopy = model.copy()
    modelCopy['muC'] = (lambda f1, f2: lambda x: f1(x) + f2(x))(model['muC'], model['muS'])
    modelCopy['sigmaC'] = (lambda f1, f2: lambda x: f1(x) + f2(x))(model['sigmaC'], model['sigmaS'])

    costElas = elas()
    firstType, secondType, thirdType, phit2, linSys2, atBounds = computeElasSub(stateMat, modelCopy, bc, x0, usePardiso, iparms, betterCP)
    costElas.firstType = firstType; costElas.secondType = secondType; costElas.thirdType = thirdType

    #############Steo 3: Compute shock price elasticities
    priceElas = elas()
    priceElas.firstType = expoElas.firstType - costElas.firstType
    priceElas.secondType = expoElas.secondType - costElas.secondType
    priceElas.thirdType = expoElas.thirdType - costElas.thirdType

    return expoElas, priceElas, costElas, None, linSys1
