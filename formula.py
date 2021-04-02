# Authors: Xin Wang <watson.x.wang@gmail.com>
# License: MIT

import re
import numpy as np
import pandas as pd
from fractions import Fraction

class Formula:
    """Create the formula to populate the data from Pandas dataframe

    Parameters
    ----------
    chunksize : int
        The size to yield the data by chunk. The default value None means returning all data.
    
    Attributes
    ----------
    dummy_pattern : str
        The pattern to create the dummy variable, that is, c(variable)
    
    power_pattern : str
        The pattern to create the exponetial term, that is, I(variable^power)
        
    polynomial_patter : str
        The pattern to create the polynomial terms, that is, poly(variable, power)

    transform_pattern : str
        The pattern to create the transformed term

    transform_ops : dict
        The operators supported in the package

    chunksize : int
        The chunksize to yield the data by chunk

    Examples
    --------
    >>> from sklearn import datasets
    >>> from formula import Formula
    >>> data = datasets.load_iris()
    >>> X_df = pd.DataFrame(data['data'])
    >>> X_df.columns = ['sepal_len', 'sepal_width', 'petal_len', "petal_width"]
    >>> Y_df = pd.DataFrame({'Class': data['target']})
    >>> df = pd.concat([X_df, Y.reset_index(drop=True)], axis=1)
    >>> fml = Formula()
    >>> X, Y = fml("Class ~ 1 + I(sepal_len^(0.5)) + log(petal_len)", data=df)
    """
    transform_ops = {"log": np.log, 
                     "exp": np.exp,
                     "sin": np.sin, 
                     "cos": np.cos,
                     "tan": np.tan,
                     "tanh": np.tanh,
                     "sqrt": np.sqrt}

    def __init__(self, chunksize=None):
        # Create the matching pattern
        self.dummy_pattern = r"^c\([\w\d]+\)$"
        self.power_pattern = r"^I\(.*\)$"
        self.polynomial_pattern = r"^poly\(.*\)$"
        self.transform_pattern = r"^({})\([\w\d]+\)$".format("|".join(Formula.transform_ops.keys()))
        # Create the chunksize for generator (Not use for now)
        self.chunksize = chunksize

    def parse_op_(self, feature):
        """Parse the feature corresponding to different patterns
        """
        if feature in self.data.columns:
            return self.data[feature].values
        elif re.search(self.dummy_pattern, feature):
            feature = feature[feature.find("(")+1:feature.find(")")]
            return pd.get_dummies(self.data[feature]).values.T
        elif re.search(self.transform_pattern, feature):
            op = feature[:feature.find("(")]
            feature = feature[feature.find("(")+1:feature.find(")")]
            return self.data[feature].transform(op).values
        elif re.search(self.power_pattern, feature):
            feature = feature[feature.find("(")+1:feature.rfind(")")]
            feature, power = list(map(str.strip, feature.split("^")))
            if "(" in power and ")" in power:
                power = power[power.find("(")+1:power.find(")")].strip()
                if "/" in power:
                    power = sum(Fraction(s) for s in power.split())
            return self.data[feature].transform(lambda x: x**(float(power))).values  
        elif re.search(self.polynomial_pattern, feature):
            feature = feature[feature.find("(")+1:feature.find(")")]
            feature, power = list(map(str.strip, feature.split(",")))
            power = int(power)
            if power < 1:
                raise ValueError("Power should be no small than 1 in the poly")
            return np.array([self.data[feature].transform(lambda x: x**p).values 
                             for p in range(1, power + 1)])
        else:
            raise ValueError("The current operation is not supported")
        
    def parse_features_(self, indep):
        """Parse the independent variables and dispatch the feature
        to different processing unit
        """
        features = list(map(str.strip, indep.split("+")))
        if len(set(features)) < len(features):
            raise ValueError("The features should be different")
        X = []
        for feature in features:
            if feature == "1":
                X.append(np.ones(self.data.shape[0]))
            elif ":" in feature:
                sub_feature = list(map(str.strip, feature.split(":")))
                if len(sub_feature) != 2:
                    raise ValueError("The operator : should be binary")
                X1, X2 = map(self.parse_op_, sub_feature)
                X.append(X1 * X2)
            elif "*" in feature:
                sub_feature = list(map(str.strip, feature.split("*")))
                if len(sub_feature) != 2:
                    raise ValueError("The operator * should be binary")
                X1, X2 = map(self.parse_op_, sub_feature)
                X.append(X1)
                X.append(X2)
                X.append(X1*X2)
            else:
                X.append(self.parse_op_(feature))
        return np.vstack(X).T

    def __call__(self, string, data):
        """Populate the data based on the formula string.
        """
        if string.count("~") > 1:
            raise ValueError("~ should be used to separate the response and the features")

        dep, indep = map(str.strip, string.split("~"))
        self.data = data
        # Get the response data
        if dep not in self.data.columns:
            raise ValueError("The response columns should be in the data")
        y = self.data[dep].values

        # Get the feature data
        X = self.parse_features_(indep)
        return X, y
