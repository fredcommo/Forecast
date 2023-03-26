import optuna
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
# from sklearn.ensemble import GradientBoostingRegressor as XGBReg
import xgboost as xgb
XGBReg = xgb.XGBRegressor

from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import make_scorer

from forecast.optimizers import(
    LinearRegression_optimizer,
    Ridge_optimizer,
    Lasso_optimizer,
    ElasticNet_optimizer,
    XGBReg_optimizer
)

from forecast.metrics import wape

#############################################################################
# Uncomment/comment the line below if you want/don't want to print jobs on terminal:
optuna.logging.set_verbosity(optuna.logging.WARNING)
#############################################################################

def logging_callback(study, frozen_trial):
    previous_best_value = study.user_attrs.get("previous_best_value", None)
    if not previous_best_value:
        previous_best_value = 0
    if previous_best_value < study.best_value:
        diff = study.best_value - previous_best_value
        study.set_user_attr("previous_best_value", study.best_value)
        print(
            " Trial {} finished with \n\t- best value: {:.6f} \n\t- improved by: {} \n\t- parameters: {}.\n".format(
            frozen_trial.number,
            frozen_trial.value,
            diff,
            frozen_trial.params,
            )
        )


def objective(trial, Xtrain, ytrain, test_size, model_list, njobs):

    # Setup values for the hyperparameters optimization:
    classifier_name = trial.suggest_categorical("classifier", model_list)
    classifier_optimizer = f"{classifier_name}_optimizer"
    classifier_obj = eval(classifier_optimizer)(trial)

    Xtrain = np.array(Xtrain)
    ytrain = np.array(ytrain).reshape(-1, 1)
    
    # Scoring method:
    score = cross_val_score(
        classifier_obj,
        Xtrain, ytrain,
        n_jobs=njobs,
        # scoring="neg_mean_squared_error",
        scoring="neg_mean_absolute_percentage_error",
        # scoring=make_scorer(wape, greater_is_better=True),
        cv=TimeSeriesSplit(n_splits=5, test_size=test_size),
        error_score=0
    )

    return score.mean()


def optimize(self, model_list, timeout=3*60, njobs=2, n_trials=10):
    """
    Run optimizer
    note: to pass args to the objective func, wrap it inside a lambda func + args
    and call the lambda func in study.optimize()

    timeout (int): max training time, in sec
    """
    
    Xtrain = self.get_Xtrain()
    ytrain = self.get_ytrain()
    
    objective_func = lambda trial: objective(trial, Xtrain, ytrain, self.test_size, model_list, njobs)
    study = optuna.create_study(direction="maximize")
    study.optimize(
        objective_func,
        n_trials=n_trials,
        n_jobs=njobs,
        timeout=timeout,
        # Uncomment/comment the line below if you want to see/not see the progress bar on terminal
        show_progress_bar=True,
        gc_after_trial=True,
        callbacks=[logging_callback]
        )

    self.study = study

##########################################
# Additional methods
##########################################

def get_best_model(self):

    def clean_param_names(params):
        """
        Retrieve the original param names,
        so they can be passed to the best model
        """
        clean = lambda p: "_".join(p.split("_")[1:])
        return {clean(p): v for p, v in params.items()}

    study = self.study
    best_value = study.best_value
    best_params = study.best_params
    best_model = best_params.pop('classifier')
    best_params = clean_param_names(best_params)

    ############################################
    # Getting the best result
    print(f"\nBest performance: {best_value:.3f}")
    print(f"Best algorithm: {best_model}")
    print(f"Best parameters (ready to use): {best_params}\n")

    return best_model, best_params


def train_best_model(self):

    best_model, best_params = self.get_best_model()

    print(f"Running {best_model} as the best model")
    print("Params:")
    print(best_params)

    model = eval(best_model)(**best_params)
    Xtrain = np.array(self.get_Xtrain())
    ytrain = np.array(self.get_ytrain()).reshape(-1, 1)
    model.fit(Xtrain, ytrain)

    self.model = model

def predict(self, what="train"):
    model = self.model
    if what == "train":
        return model.predict(np.array(self.get_Xtrain())).reshape(-1, 1)
    elif what == "test":
        return model.predict(np.array(self.get_Xtest())).reshape(-1, 1)
    else:
        return None

def plot(self):

    study = self.study
    best_value = study.best_value
    best_params = study.best_params
    best_model = best_params.pop('classifier')

    y = self.get_ytest()
    yhat = self.predict("test")
    date_time = self.date_time
    
    plt.figure(figsize=(18, 8))

    plt.plot(date_time[-self.test_size:], y, linewidth=3, label="Observed")
    plt.plot(date_time[-self.test_size:], yhat, linewidth=3, c='orange', label="predicted")

    title = f"Best model: {best_model}, Best metrics (CV): {best_value:.4f}, WAPE: {wape(y, yhat):.3f}"
    plt.title(title, fontsize=22)
    plt.xlabel('Time', fontsize=18)
    plt.xticks(fontsize=14)
    plt.ylabel('Values', fontsize=18)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)

    plt.show()
