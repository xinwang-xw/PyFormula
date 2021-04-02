# PyFormula

This package aims at providing the R-like formula for data analysis in Python. The underlying data is a Pandas dataframe and the package generates the numpy array for scikit-learn.

## Background

R-like formula is easy to use for data analysis, for example, 

```r
lm(y ~ 1 + c(x1) + x2:x3, data = df)
```

By using the above formula, the user can create the dummy variables for x1 and the multiplicative term from x2 and x3. Using the formula allows the user to focus on the relationship between the response and the explanatory variables, without populating the data from the Pandas dataframe manually.

## Usage

The user can create the Formula object and call this object to populate the response and the explanatory variables by the formula string. One usage is demonstrated as below:

```python
from formula import Formula

fml = Formula()
X, y = fml("y ~ 1 + c(x1) + x2:x3", data=df)
```

The above code populates the data from the dataframe df by using the formula specified in the first argument string. The package currerntly supports the following transformation:
* **c(variable)** --- creating the dummary variables for the "variable"
* **variable1 : variable2**  --- creating the multiplicative (i.e. interactive) term for variable1 and variable 2
* **variable1 * variable2**  --- creating three terms, i.e. variable1, variable 2 and variable1:variable2
* **I(variable^power)** --- creating the exponential term, i.e. the variable with the power, where power should be integer, floating number or fractional number. For example, I(x1^(1/2)) creates the term x1 with the power 1/2. And the user can use I(x1^(1 1/2)) means the term x1 with the power 3/2, which is equivalent to use I(x1^1.5)
* **poly(variable, power)** --- creating the polynomial terms, that is, variable, variable^1, ..., variable^power, where power should be no smaller than 1. 
* **op(variable)** --- creating the transformed data with the operator op, where op should be one of the one-parameter function log, exp, sin, cos, tan, tanh, sqrt, square, and etc.

## TODO List

The package is still on developing and populates the data correctly for now, but the running speed or chunksize for large data haven't been taken into account
