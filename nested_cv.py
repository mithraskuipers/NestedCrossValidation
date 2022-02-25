# Nested CV procedure [manual]

def nested_cv(X, y, estimator, param_dict, ncv_n_outerfolds = 3, ncv_n_innerfolds = 3, verbose = False):
    
    # Create ParameterGrid 
    
    hyperparam_combs = list(ParameterGrid(param_dict))
    outer_cv_scores = []

    # Outer CV
    
    cv_outer = model_selection.KFold(n_splits = ncv_n_outerfolds, shuffle = True)
    for fold, (cv_outer_train_idx, cv_outer_test_idx) in enumerate(cv_outer.split(X)):
        
        outer_X_train, outer_X_test = X.loc[cv_outer_train_idx], X.loc[cv_outer_test_idx] #.loc, otherwise KeyError: "None of [Int64Index([0, 1, 2, 4], dtype='int64')] are in the [columns]"
        outer_y_train, outer_y_test = y.loc[cv_outer_train_idx], y.loc[cv_outer_test_idx]
    
        # Procedure below imitates sklearn's GridSearchCV()
        
        # Inner CV: Hyperparameter Optimization 
        # For outer fold, run CV for each combhyperparams
        
        hyperparam_scores = []
        
        for curr_param in hyperparam_combs:
            
            inner_cv_scores = []
            
            kfolds = model_selection.KFold(n_splits = ncv_n_innerfolds, shuffle = True)
            for train_idx, idx_test in kfolds.split(outer_X_train):
                
                # Split the Outer CV training set into train/test
                inner_X_train, inner_X_test = X.loc[train_idx], X.loc[idx_test]
                inner_y_train, inner_y_test = y.loc[train_idx], y.loc[idx_test]
        
                # Fit model
                estimator.set_params(**curr_param)
                estimator.fit(inner_X_train, inner_y_train)
                
                # Model performance
                inner_preds = estimator.predict(inner_X_test)
                #inner_cv_scores.append(np.corrcoef(inner_preds, inner_y_test)[0,1]) #slicing because corr returns 2x2 matrix, we only want diagonal
                inner_cv_scores.append(mean_squared_error(inner_preds, inner_y_test))
        
            # Mean score of Inner CV models (with all kinds of hyperparams)
            inner_cv_mean = np.mean(inner_cv_scores)
            
            # Append inner_cv_mean to list
            hyperparam_scores.append(inner_cv_mean)
        
        # For current Outer fold, identify hyperparams that resulted in best model
        best_hyperparams, best_hyperparams_score = min(enumerate(hyperparam_scores), key = operator.itemgetter(1))
    
        outer_result = "\nBest hyperparams of Outer fold {0}:\n{1}\n"
        if verbose : print(outer_result.format(fold, hyperparam_combs[best_hyperparams]))

        # Fit algorithm with best hyperparams (of Outer fold) on Outer train data
        
        estimator.set_params(**hyperparam_combs[best_hyperparams])
        estimator.fit(outer_X_train, outer_y_train)
        predictions = estimator.predict(outer_X_test)
        #outer_cv_scores.append(np.corrcoef(predictions, outer_y_test)[0, 1]) #slicing because corr returns 2x2 matrix, we only want diagonal
        outer_cv_scores.append(mean_squared_error(predictions, outer_y_test))

    # Unbiased estimate of the prediction error (using found best hyperparams)
    
    ncv_error = np.mean(outer_cv_scores)
    
    ncv_res = "\nUnbiased prediction error (MSE) for {0}:\n{1}\n"
    if verbose : print(ncv_res.format(str(estimator).split("(", 1)[0], ncv_error))

    return(ncv_error, outer_cv_scores)
