import warnings
warnings.filterwarnings("ignore")

import os, pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

from joblib import dump, load
from src.methods.stratified import create_folds
from src.methods.causal_models import *
from src.methods.performance import *
from src.data.fun_synthetic_data import *
from src.data.data import load_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from econml.dml import ForestDMLCateEstimator
from sklearn.ensemble import AdaBoostClassifier

pd.options.mode.chained_assignment = None
np.seterr(divide = 'ignore')
