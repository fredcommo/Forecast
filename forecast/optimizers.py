# Importing the Packages:
import functools

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
# from sklearn.ensemble import GradientBoostingRegressor as XGBReg
import xgboost as xgb
XGBReg = xgb.XGBRegressor

############################################
# Decorator:
def optimizer(model):
    def deco_optim(parameterizer):
        @functools.wraps(parameterizer)
        def wrapper_optim(*args):
            params = parameterizer(*args)
            return model(**params)
        return wrapper_optim
    return deco_optim


############################################
# Set trial parameters for each model:
@optimizer(LinearRegression)
def LinearRegression_optimizer(trial):
    params = {}
    return params

@optimizer(Ridge)
def Ridge_optimizer(trial):
    params = {
        "alpha": trial.suggest_float("ridge_alpha", 0.1, 2, step=0.1),
        "positive": trial.suggest_categorical("ridge_positive", [True, False]),
        "solver": "auto"
        }
    return params

@optimizer(Lasso)
def Lasso_optimizer(trial):
    params = {
        "alpha": trial.suggest_float("lasso_alpha", 0.1, 2, step=0.1),
        'warm_start': trial.suggest_categorical("lasso_warm_start", [True, False]),
        "positive": trial.suggest_categorical("lasso_positive", [True, False]),
        "selection": trial.suggest_categorical("lasso_selection", ["cyclic", "random"])
        }
    return params

@optimizer(ElasticNet)
def ElasticNet_optimizer(trial):
    params = {
        "alpha": trial.suggest_float("elastic_alpha", 0.1, 2, step=0.1),
        "l1_ratio": trial.suggest_float("elastic_l1_ratio", 0.1, 1, step=0.1),
        'warm_start': trial.suggest_categorical("elastic_warm_start", [True, False]),
        "positive": trial.suggest_categorical("elastic_positive", [True, False]),
        "selection": trial.suggest_categorical("elastic_selection", ["cyclic", "random"])
        }
    return params

@optimizer(XGBReg)
def XGBReg_optimizer(trial):
    params = {
        "learning_rate": trial.suggest_float("xgbr_learning_rate", 0.1, 1, step=0.1),
        "n_estimators": trial.suggest_categorical("xgbr_n_estimators", [200, 500, 1000])
        }
    return params
