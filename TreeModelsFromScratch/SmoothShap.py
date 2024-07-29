import numpy as np
import pandas as pd
#import shap
from sklearn.linear_model import LinearRegression
#from sklearn.base import clone
from warnings import warn#, catch_warnings, simplefilter
from copy import deepcopy
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.model_selection import KFold
from shap.explainers._tree import SingleTree
import itertools
from tqdm import tqdm
import scipy.stats as st
import math


# Utility functions to verify SHAP model which has been created based on tree output
def verify_shap_model(tree, explainer, X):
    '''Verify the integrity of SHAP explainer model by comparing output of export_tree_for_SHAP vs original model'''

    if tree.treetype=="classification":
        # Make sure that the ingested SHAP model (a TreeEnsemble object) makes the same predictions as the original model
        assert np.abs(
            explainer.model.predict(X) -
            tree.predict_proba(X)[:, 1]).max() < 1e-4

        # make sure the SHAP values sum up to the model output (this is the local accuracy property)
        assert np.abs(explainer.expected_value +
                    explainer.shap_values(X).sum(1) -
                    tree.predict_proba(X)[:, 1]).max() < 1e-4
    else:
        # Make sure that the ingested SHAP model (a TreeEnsemble object) makes the same predictions as the original model
        assert np.abs(explainer.model.predict(X) -
                    tree.predict(X)).max() < 1e-4

        # make sure the SHAP values sum up to the model output (this is the local accuracy property)
        assert np.abs(explainer.expected_value +
                    explainer.shap_values(X).sum(1) -
                    tree.predict(X)).max() < 1e-4


def smooth_shap(shap_values_inbag, shap_values_oob, detailed_output=False):
    """
    Compute smooth shap score per feature based on oob and inbag SHAP correlations.
    Parameters
    ----------
    shap_values_inbag : ndarray of shape (n_samples, n_features)
        SHAP values calculated on Inbag samples
    shap_values_oob : ndarray of shape (n_samples, n_features)
        SHAP values calculated on OOB samples
    detailed_output : bool, default=False
        If linear models and shap_vals should also be returned
    Returns
    -------
    smooth_shap_vals: ndarray of shape (n_samples, n_features)
    mean_smooth_shap: array of shape (n_features,)
    lin_coefs : list
        List of coefficients obtained from linear models used to smooth SHAP values.
        Order of features depends on order of the given samples.
    """

    lin_models = []
    lin_coefs = []
    smooth_shap_vals = []

    # Check if there are any y obs where there is no oob prediction:
    if np.isnan(shap_values_oob).any():

        # identify index of all nan values (where no oob or inbag pred is found)
        nan_indxs = np.argwhere((np.isnan(shap_values_inbag))
                                | (np.isnan(shap_values_oob)))

        #Throw UserWarning of how many values did not have an oob prediction
        message = """{} out of {} samples do not have Inbag or OOB shap values. This probably means too few trees were used to compute any reliable OOB estimates. These samples were dropped before computing the smooth shap score""".format(
            int(nan_indxs.shape[0] / shap_values_oob.shape[1]), len(shap_values_oob))
        warn(message)

        # drop these NaN values from x (shap inbag) and y (shap oob)
        mask = np.ones(shap_values_oob.shape[0], dtype=bool)
        mask[nan_indxs[:, 0]] = False
        shap_values_inbag = shap_values_inbag[mask]
        shap_values_oob = shap_values_oob[mask]

    #for each feature fit linear regression shap_oob = shap_inbag*beta+u1 and predict shap oob
    for i in range(shap_values_inbag.shape[1]):
        linear_reg = LinearRegression().fit(shap_values_inbag[:, i].reshape(-1, 1), shap_values_oob[:, i])
        lin_models.append(linear_reg)
        lin_coefs.append(linear_reg.coef_[0])
        smooth_shap_vals.append(linear_reg.predict(shap_values_inbag[:,i].reshape(-1, 1)))

    # Turn into array of shape (rows = observations, col=feats)
    smooth_shap_vals = np.array(smooth_shap_vals).T

    # Mean absolte smooth shap value per feature
    mean_smooth_shap = np.abs(smooth_shap_vals).mean(axis=0)

    if detailed_output:
        return smooth_shap_vals, mean_smooth_shap, lin_coefs, lin_models, shap_values_inbag, shap_values_oob

    return smooth_shap_vals, mean_smooth_shap, lin_coefs


def conf_int_ratio_two_var(pop_1, pop_2, alpha=0.05):
    '''
    Not used in Thesis.
    Calculate shrinkage parameter based on confidence interval based on 2-sample difference in variances.
    '''

    # number of samples per population
    n1 = len(pop_1)
    n2 = len(pop_2)

    # variance if samples per population
    var1 = np.var(pop_1, ddof=1)
    var2 = np.var(pop_2, ddof=1)
    var_rat = var1/var2

    # F value for normal distribution  with alpha and degrees of freedom dfn and dfd
    f_val_low = st.f.ppf(q=(alpha/2), dfn=n1-1, dfd=n2-1)
    f_val_up = st.f.ppf(q=1-(alpha/2), dfn=n1-1, dfd=n2-1)

    # confidence interval
    conf_int = np.array([f_val_low*var_rat, f_val_up*var_rat])

    # if upper CI < 1., then take CI upper, else take 1
    if conf_int[1]<1.:
        m = conf_int[1]
    else:
        m = 1.

    return conf_int, m

def conf_int_ratio_mse_ratio(pop_1, pop_2, node_val_inbag, node_dict_inbag, node_dict_oob, type="regression", alpha=0.05):
    '''AugHS MSE: Calculate shrinkage parameter based on confidence interval based on 2-sample difference in MSE'''

    # number of samples per population
    n1 = len(pop_1) #pop1 = y_true_inbag
    n2 = len(pop_2) #pop2 = y_true_oob

    # Get reestiated node values oob
    if type=="classification":
        node_val_oob = node_dict_oob["value"] # 0 or 1
        node_prob_oob = node_dict_oob["prob_distribution"] # probability dis between 0 and 1
        node_prob_inbag = node_dict_inbag["prob_distribution"]
    else:
        node_val_oob = node_dict_oob["value"] # mean value of targets in node

    #full array of node vals for MSE calc
    node_val_pop1 = np.full(pop_1.shape, node_val_inbag)
    node_val_pop2 = np.full(pop_2.shape, node_val_inbag)

    # MSE for OOB and Inbag
    mse_inbag = mean_squared_error(node_val_pop1, pop_1)
    mse_oob = mean_squared_error(node_val_pop2, pop_2)

    #if mse inbag & mse oob == 0 then set shrinkage to 1
    if (mse_inbag==0) & (mse_oob==0):
        return np.array([0,0]), 1.

    elif (mse_inbag!=0) & (mse_oob==0):
        #check difference in node valeus
        if math.isclose(node_val_inbag, node_val_oob):
            return np.array([0,0]), 1.
        # for clf additionally check if prob distribution is close
        elif (type=="classification") & (math.isclose(node_prob_inbag, node_prob_oob)):
            return np.array([0,0]), 1.
        # otherwise set to 0
        else:
            return np.array([1,1]), 0.

    mse_rat = mse_inbag/mse_oob

    #Set degrees of freeedom to 1 if only one sample in node (otherwise f value can not be determined)
    if n1==1:
        n1+=1
    if n2==1:
        n2+=1

    # f value for normal distribution  with alpha and degrees of freedom dfn and dfd
    f_val_low = st.f.ppf(q=(alpha/2), dfn=n1-1, dfd=n2-1)
    f_val_up = st.f.ppf(q=1-(alpha/2), dfn=n1-1, dfd=n2-1)

    # confidence interval
    conf_int = np.array([f_val_low*mse_rat, f_val_up*mse_rat])

    # if upper CI < 1. and != inf, then take CI upper, else take 1
    if (conf_int[1]<1.) & ((~np.array_equal(conf_int, np.array([-np.inf, -np.inf]))) or (~np.array_equal(conf_int, np.array([np.inf, np.inf])))):
        m = conf_int[1]
    else:
        m = 1.

    return conf_int, m



def conf_int_cohens_d(pop_1, pop_2, reg_param=2, alpha=0.05, cohen_statistic="f"):
    '''Not used in Thesis.
    Calculate shrinkage parameter based on confidence interval based on 2-sample difference in means'''

    # number of samples per population
    n1 = len(pop_1)
    n2 = len(pop_2)

    # Mean of sample populations
    mu1 = np.mean(pop_1)
    mu2 = np.mean(pop_2)

    # sample variances
    var1 = np.var(pop_1, ddof=1)
    var2 = np.var(pop_2, ddof=1)

    var_pooled = ((n1-1)*var1+(n2-1)*var2)/(n1+n2-2)

    # cohens d
    d = (mu1-mu2)/np.sqrt(var_pooled)

    eff_siz = np.sqrt((n1+n2)/(n1*n2)+(d**2/(2*(n1+n2))))

    # confidence interval
    if cohen_statistic=="f":
        conf_int = np.array([d-1.96*eff_siz, d+1.96*eff_siz])

    elif cohen_statistic=="t":

        df_pooled = (n1+n2-2) # degrees of freedom pooled
        t_low = np.abs(st.t.ppf(q=(alpha/2), df=df_pooled))
        t_up = np.abs(st.t.ppf(q=(1-alpha/2), df=df_pooled))

        conf_int = np.array([d-t_low*eff_siz, d+t_up*eff_siz])

    # smoothing parameter: if 0 within the range, take 0, else take min abs CI
    if conf_int[0]<=0<=conf_int[1]:
        m = (1+reg_param*0)**(-1) # ==1
    else:
        m = (1+reg_param*np.min(np.abs(conf_int)))**(-1)

    return conf_int, m


def cross_val_score_scratch(estimator, X, y, cv=10, scoring_func=roc_auc_score, shuffle=True, random_state=None):
    '''Perform k-fold cross validation scoring for estimators with .fit and .predict function (imodels and scratch models)'''

    kf = KFold(n_splits=cv, shuffle=shuffle, random_state=random_state)
    scores = []

    for train_index, test_index in kf.split(X):

        # Create true copy of estimator (refitting of scratch models is not possbile)
        estimator_copy = deepcopy(estimator)

        #split data
        if isinstance(X, pd.DataFrame):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        else:
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

        #fit estimator, predict & score
        estimator_copy.fit(X=X_train, y=y_train)
        y_pred = estimator_copy.predict(X_test)
        scores.append(scoring_func(y_test, y_pred))

    return scores

def GridSearchCV_scratch(estimator, grid, X, y, cv=10, scoring_func=None, fit_best_est=True, shuffle=True, random_state=None):
    '''Perform k-fold cross validation GridSearch RandomForest models. Similar to sklearn GridSearch.
    Parameters
    ----------
    estimator : RandomForest instance
        Unfitted RandomForest model
    grid : dict
        {model parameter to be tuned: [parameter settings to be tested]
        The key has to be named exactly as the parameter of the RandomForest model and
        have to be part of the following list:
        `["reg_param", "n_trees", "HS_lambda", "max_depth", "min_samples_split",
        "min_samples_leaf", "k", "n_features", "n_feature", "HShrinkage",
        "cohen_reg_param", "cohen_statistic", "HS_nodewise_shrink_type]`
    X : ndarray
        X data for training and hyperparameter tuning
    y : ndarray
        y data for training and hyperparameter tuning
    cv : int, default=10
        Number of folds used per parameter setting
    scoring_func : sklearn scoring function, default=None
        Function to score  and select the best parameter settings
    fit_best_est : bool, default=True
        Fit estimator with best parameter setting
    shuffle : bool, default =True
        If shuffling of the data during k-fold should be used
    random_state : int
        Random seed to replicate the folds
    Returns
    -------
    results: dict
        Results for each parameter setting
        - best_param_comb: Best param combination found
        - best_test_score: Avg. test score over all k-folds for best param comb
        - best_test_scores: Test scores over all k-folds for best param comb
        - mean_test_scores: Avg. test score over all k-folds for all param comb
        - param_combinations: List of param combinations
        - cv_scores_p_split: Test scores over all k-folds for all param combs
    '''

    valid_grid_keys = [
        "reg_param", "n_trees", "HS_lambda", "max_depth", "min_samples_split",
        "min_samples_leaf", "k", "n_features", "n_feature", "HShrinkage",
        "cohen_reg_param", "cohen_statistic", "HS_nodewise_shrink_type"
    ]

    # Check if keys are valid
    for key in grid.keys():
        if key not in valid_grid_keys:
            message = """{} is not a valid hyperparameter for this function. Key has to be part of the following list
            {}""".format(key, valid_grid_keys)
            warn(message)
            break

    keys = list(grid.keys())


    # If to show for each split progress bar
#    if pbar:
#        pos_combs = tqdm(list(itertools.product(*grid.values()))) #Get all possible combinations of hyperparameters from grid
#    else:
    pos_combs = list(itertools.product(*grid.values())) #Get all possible combinations of hyperparameters from grid

    cv_scores = np.zeros((len(pos_combs), cv)) #array to store cv results

    #iterrate over all possible combinations
    for j, param_comb in enumerate(pos_combs):

        # Create deep copy of estimator (necessary for scratch models)
        estimator_copy = deepcopy(estimator)

        #set attributes in param_comb
        for i, key in enumerate(keys):
            setattr(estimator_copy, key, param_comb[i])

        # k fold cross validation
        cv_scores[j,:] = cross_val_score_scratch(estimator_copy, X, y, cv=cv, scoring_func=scoring_func,
                                            shuffle=shuffle, random_state=random_state)


    # find best combination (highest avg. score accross all folds)
    cv_scores_mean = cv_scores.mean(axis=1)
    idx_best_comb = cv_scores_mean.argmax()

    results = {
        "best_param_comb": pos_combs[idx_best_comb],
        "best_test_score": cv_scores_mean[idx_best_comb],
        "best_test_scores": cv_scores[idx_best_comb],
        "mean_test_scores": cv_scores_mean,
        "param_combinations": pos_combs,
        "cv_scores_p_split": cv_scores
    }

    if fit_best_est is False:
        return results

    #fit estimator with best found combination
    for i, key in enumerate(keys):
        setattr(estimator, key, pos_combs[idx_best_comb][i])

    estimator.fit(X, y) #original estimaotr will be fitted inplace does not need to be returned

    return results


def export_imodels_for_SHAP(imodel, is_forest=True):
    '''Exports imodel estimator in a shap readable format'''
    if is_forest:

        tree_dicts = []

        for tree in imodel.estimator_.estimators_:

            # extract the arrays that define the tree
            children_left = tree.tree_.children_left
            children_right = tree.tree_.children_right
            children_default = children_right.copy(
            )  # because sklearn does not use missing values
            features = tree.tree_.feature
            thresholds = tree.tree_.threshold
            if str(type(imodel))[-12:-2] == "Classifier":
                values = tree.tree_.value[:, :, 1]
            else:
                values = tree.tree_.value.reshape(tree.tree_.value.shape[0], 1)
            node_sample_weight = tree.tree_.weighted_n_node_samples

            # define a custom tree model
            tree_dict = {
                "children_left": children_left,
                "children_right": children_right,
                "children_default": children_default,
                "features": features,
                "thresholds": thresholds,
                "values": values,
                "node_sample_weight": node_sample_weight
            }

            tree_dicts.append(tree_dict)

        if str(type(imodel))[-12:-2] == "Classifier":
            model = [
                SingleTree(t, scaling=1.0 / len(tree_dicts), normalize=True)
                for t in tree_dicts
            ]
        else:
            model = [
                SingleTree(t, scaling=1.0 / len(tree_dicts))
                for t in tree_dicts
            ]

        return model

    # If single Tree model

    # extract the arrays that define the tree
    children_left = imodel.estimator_.tree_.children_left
    children_right = imodel.estimator_.tree_.children_right
    children_default = children_right.copy(
    )  # because sklearn does not use missing values
    features = imodel.estimator_.tree_.feature
    thresholds = imodel.estimator_.tree_.threshold

    if str(type(imodel))[-12:-2] == "Classifier":
        values = imodel.estimator_.tree_.value[:, :, 1]
    else:
        values = imodel.estimator_.tree_.value.reshape(
            imodel.estimator_.tree_.value.shape[0], 1)
    node_sample_weight = imodel.estimator_.tree_.weighted_n_node_samples

    # define a custom tree model
    tree_dict = {
        "children_left": children_left,
        "children_right": children_right,
        "children_default": children_default,
        "features": features,
        "thresholds": thresholds,
        "values": values,
        "node_sample_weight": node_sample_weight
    }
    model = {"trees": [tree_dict]}

    return model
