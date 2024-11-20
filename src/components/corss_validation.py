import numpy as np
from src.components.linear_regression_model import LinearRegresion

def cross_validation(X,y,k_folds=5):
    """
    Perform k-fold cross-validation
    returns R^2 scores for each fold
    """
    
    #Create array to store scores
    scores = np.zeros(k_folds)
    
    #Calculate fold size
    fold_size = len(X) // k_folds
    
    #shuffle indices
    indices = np.random.permutation(len(X))
    
    for i in range(k_folds):
        #Calculate start and end indices for test fold
        start_idx = i * fold_size
        end_idx = start_idx + fold_size if i < k_folds - 1 else len(X)
        
        #Get test indices for this fold
        test_idx = indices[start_idx:end_idx]
        
        #Get train indices
        train_idx = np.concatenate([indices[:start_idx], indices[end_idx:]])
        
        #split data into training and test data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        #Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        #make predictions
        y_pred = model.predict(X_test)
        
        #Calculate and store R^2 score
        scores[i] = r2_score(y_test, y_pred)
        
    return scores