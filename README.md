# Risk Uncertainty Value

`Expansion Suite` is an open source Python toolbox for macro-finance research. It solves the nonlinear DSGE model with recursive utility formulation motivated by robustness or risk concerns based on the small-noise expansion method.



`Expansion Suite` is built on the Linear Quadratic framework proposed by [Borovička and Hansen (2014)](https://larspeterhansen.org/wp-content/uploads/2016/10/Examining-Macroeconomic-Models-through-the-Lens-of-Asset-Pricing.pdf). It follows the algorithm in [Exploring Recursive Utility](https://larspeterhansen.org/class-notes/) to approximate the model solution under uncertainty. 



## Jupyter Notebook Illustrations

[uncertainexpansion.ipynb](https://github.com/lphansen/RiskUncertaintyValue/blob/main/uncertainexpansion.ipynb)

- A guide to solve the DSGE model with `Expansion Suite`. 

- An example of the Adjustment Cost Model is provided in the notebook.

[shockelasticity.ipynb](https://github.com/lphansen/RiskUncertaintyValue/blob/main/shockelasticity.ipynb)

- A guide to compute shock elasticities with `Expansion Suite` 

- An example of the Bansal Yaron Long-run Risk Model is provided in the notebook.

[quickguide.ipynb](https://github.com/lphansen/RiskUncertaintyValue/blob/main/quickguide.ipynb)

- Provides a five-minute guide to solve the DSGE model using the expansion suite, as well as how to 
  - compute shock elasticities and IRF
  - approximate and simulate variables based on model solutions. 
- Provides some examples for `LinQuadVar` computations

## Source Files

[lin_quad.py](https://github.com/lphansen/RiskUncertaintyValue/blob/main/src/lin_quad.py)
- Defines the linear-quadratic variable structure to facilitate operations in expansion solvers and elasticity calculation with `LinQuadVar` class.

[lin_quad_util.py](https://github.com/lphansen/RiskUncertaintyValue/blob/main/src/lin_quad_util.py)
- Integrated operation tools on `LinQuadVar`.

[utilities.py](https://github.com/lphansen/RiskUncertaintyValue/blob/main/src/utilities.py)
- Matrix and linear algebra operation functions facilitate the computation.

[derivatives.py](https://github.com/lphansen/RiskUncertaintyValue/blob/main/src/derivatives.py)
- Functions to compute numerical derivatives used in the expansion solver.

[uncertain_expansion.py](https://github.com/lphansen/RiskUncertaintyValue/blob/main/src/uncertain_expansion.py)

- Functions to implement first and second order expansion, approximate the continuation values, change of probability measure, and iteration schemes.

[elasticity.py](https://github.com/lphansen/RiskUncertaintyValue/blob/main/src/elasticity.py)
- Functions to compute shock elasticities.

[BY_example_sol.py](https://github.com/lphansen/RiskUncertaintyValue/blob/main/src/BY_example_sol.py)

- Produce the solutions for the Bansal Yaron Long-run Risk Model used in the [shockelasticity.ipynb](https://github.com/lphansen/RiskUncertaintyValue/blob/main/shockelasticity.ipynb).



## Recent Updates, by Jan. 27, 00:12, CT

1. [uncertainexpansion.ipynb](https://github.com/lphansen/RiskUncertaintyValue/blob/main/uncertainexpansion.ipynb)
   - remove `gb_tp1` as a state variable
   
   - parameters updating for the adjustment cost model example, which are consistent with the homework.
   
     - Note the $\gamma=1$ cannot be solved by the `Expansion Suite`, the reason is that $1-\gamma$ will show in denominators of the recursive utility adjustment terms, we can try to use values close to $1$, `γ=1+1e-5`, for example.
   
     ```python
     γ = 10.
     ρ = 4./3
     β = np.exp(-0.0025)
     a = 0.0288
     ϕ_2 = 88.
     ϕ_1 = 1/ϕ_2
     α_k = 0.0088
     σ_k = np.array([[0.477],[0]]) * 0.01
     A = np.exp(-0.014)
     B = np.array([[0.011,0.025]]) * 0.01
     ```
   
2. [uncertain_expansion.py](https://github.com/lphansen/RiskUncertaintyValue/blob/main/src/uncertain_expansion.py), [lin_quad_util.py](https://github.com/lphansen/RiskUncertaintyValue/blob/main/src/lin_quad_util.py), [BY_example_sol.py](https://github.com/lphansen/RiskUncertaintyValue/blob/main/src/BY_example_sol.py)
   - New functions and methods added
   
3. [quickguide.ipynb](https://github.com/lphansen/RiskUncertaintyValue/blob/main/quickguide.ipynb)
   - New notebook added
   - Section 1 is a five-minute guide for the `Expansion Suite` as a DSGE solver. 
   - Section 2 are examples for the computations in `LinQuadVar`.



















