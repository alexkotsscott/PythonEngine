# Bayesian Black Box we wish to optimize - hyperparameters

def lgb_black_box(
    num_leaves,  # int
    min_data_in_leaf,  # int
    learning_rate,
    min_sum_hessian_in_leaf,    # int  
    feature_fraction,
    lambda_l1,
    lambda_l2,
    min_gain_to_split,
    max_depth):
    
    """
    Args:
            None
            In practice would add in the option to set the parameters via this function

    Returns:    
            rmse - as above could refit this script to adapt to other metrics

    Notes:
            First draft

    """

    
    # lgb need some inputs as int but BayesianOptimization library send continuous values values. so we change type.

    num_leaves = int(num_leaves)
    min_data_in_leaf = int(min_data_in_leaf)
    max_depth = int(max_depth)
    
    # all this hyperparameter values are just for test. our goal in this kernel is how to use bayesian optimization
    # you can see lgb documentation for more info about hyperparameters
    params = {
        'num_leaves': num_leaves,
        'max_bin': 63,
        'min_data_in_leaf': min_data_in_leaf,
        'learning_rate': learning_rate,
        'min_sum_hessian_in_leaf': min_sum_hessian_in_leaf,
        'bagging_fraction': 1.0,
        'bagging_freq': 5,
        'feature_fraction': feature_fraction,
        'lambda_l1': lambda_l1,
        'lambda_l2': lambda_l2,
        'min_gain_to_split': min_gain_to_split,
        'max_depth': max_depth,
        'save_binary': True, 
        'seed': 1337,
        'feature_fraction_seed': 1337,
        'bagging_seed': 1337,
        'drop_seed': 1337,
        'data_random_seed': 1337,
        'objective': 'regression',
        'boosting_type': 'gbdt',
        'verbose': 1,
        'metric': 'rmse',
        'is_unbalance': True,
        'boost_from_average': False, 
    }
    
    # Self explanatory below
    
    train_data = lgb.Dataset(X_valid_train.iloc[bayesian_tr_index].values,
                            label = y_valid_train[bayesian_tr_index],
                            feature_name=predictors,
                            free_raw_data = False)
    
    
    validation_data = lgb.Dataset(X_valid_train.iloc[bayesian_val_index].values,
                                 label= y_valid_train[bayesian_val_index],
                                 feature_name=predictors,
                                 free_raw_data=False)
    
    num_round = 5000
    clf = lgb.train(params, train_data, num_round, valid_sets = [validation_data], verbose_eval=250,
                 early_stopping_rounds = 50)
    
    predictions = clf.predict(X_valid_train.iloc[bayesian_val_index].values,
                              num_iteration = clf.best_iteration)

#      we need to compute a regression score. roc_auc_score is a classification score. we can't use it
#     score = metrics.roc_auc_score(y_valid_train[bayesian_val_index], predictions)
    mse = mean_squared_error(y_valid_train[bayesian_val_index], predictions)
    rmse = np.sqrt(mse)
#     our bayesian optimization expect us to give it increasing number to understand this is getting better
    return -rmse
