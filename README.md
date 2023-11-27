# Risk Uncertainty Value

`Expansion Suite` is an open source Python toolbox for macro-finance research. It solves the nonlinear DSGE model with recursive utility formulation motivated by robustness or risk concerns based on the small-noise expansion method.



`Expansion Suite` is built on the Linear Quadratic framework proposed by [Borovička and Hansen (2014)](https://larspeterhansen.org/wp-content/uploads/2016/10/Examining-Macroeconomic-Models-through-the-Lens-of-Asset-Pricing.pdf). It follows the algorithm in [Exploring Recursive Utility](https://larspeterhansen.org/class-notes/) to approximate the model solution under uncertainty. 



We look forward to your comments and appreciate your feedback!




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



## Installation and Usage

We can directly download the whole GitHub repo to use the `Expansion Suite`. We can also use all functions in the  `Expansion Suite` from the Python package `mfrpy`.

One line code

```python
pip install mfrpy
```

Upgrade with one line code

```python
pip install --upgrade mfrpy
```

After installation, we can import the modules and functions from `mfrpy` in the Jupyter notebook.

```python
from mfrpy.uncertain_expansion import uncertain_expansion
from mfrpy.elasticity import exposure_elasticity, price_elasticity
from mfrpy.lin_quad import LinQuadVar
```

For example, we can solve the nonlinear DSGE model with `uncertain_expansion` as shown in the notebook using five inputs

```
ModelSol = uncertain_expansion(eq, ss, var_shape, args, gc)
```

We can use the `exposure_elasticity` , `price_elasticity` to calcalute the shock elasticities of the variables we are interested in. We can also compute the shock elasticties for all jump and state variables in the model using the method `elasticities` defined in the `ModelSolution` class.

```python
expo_elas = exposure_elasticity(log_M_growth, X1_tp1, X2_tp1, T=400, shock=0)
price_elas = price_elasticity(log_G_growth, log_S_growth, X1_tp1, X2_tp1, T=400, shock=0)
expo_elas, price_elas = ModelSol.elasticities(log_SDF_ex, T=400, shock=0)
```







