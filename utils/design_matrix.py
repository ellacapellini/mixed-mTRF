import numpy as np 
def build_design_matrix(stimulus, lags, fill_value=0.0):
    n_times = len(stimulus)
    n_lags = len(lags)
    X = np.full((n_times, n_lags), fill_value)
    
    for i, lag in enumerate(lags):
        if lag >= 0: #positive lag: shift stim back in time -> brain at time t responds to stimulus at t-lag
            X[lag:, i] = stimulus[:n_times - lag]
        else:  #negative lag: shift stim forward -> capture brain activity that precedes stim
            X[:, n_times + lag, i] = stimulus[-lag:]
    return X

def build_multifeature_design_matrix(features, lags):
    """Design matrix for multiple stim features simultaneously
    Stack individual design matrices horizontally
    Param:
    each value is: 1D stimulus feature of shape (n_times,) and shared lag grid for all features
    Return:
    maps feature name to its column slice in X (good for extracting individual TRF after fitting)
    """
    n_lags = len(lags)
    blocks =[]
    feature_slices = {}
    col = 0
    for name, feat in features.items():
        block = build_design_matrix(feat, lags)
        blocks.append(block)
        feature_slices[name] = slice(col, col + n_lags)
        col += n_lags
    X = np.hstack(blocks)
    return X, feature_slices  