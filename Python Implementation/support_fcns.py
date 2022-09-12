import numpy as np
import scipy
from scipy.interpolate import interp1d

def _downsample_(arr,npts):
    import scipy
    from scipy.interpolate import interp1d
    interpolated = scipy.interpolate.interp1d(np.arange(len(arr)), arr, axis = 0, fill_value = 'extrapolate')
    downsampled = interpolated(np.linspace(0, len(arr), npts))
    return downsampled

def downsample(arr,ppd_init,ppd_desired):
    import scipy
    from scipy.interpolate import interp1d
    ndays = round(len(arr) / ppd_init)
    newsize = ndays * ppd_desired
    newarr = _downsample_(arr,newsize)
    return newarr

def getdaymat(X,ppd):
    return X.reshape(round(len(X)/ppd),ppd)

def PCA(A,pcnt_variance):
    # A should be centered (or standardized)
    # compute covariance matrix
    covariance_matrix = np.cov(A.T)

    # find largest eigenvalues
    n = A.shape[1]
    eigen_values, eigen_vectors = scipy.linalg.eigh(covariance_matrix)
    sorted_idcs = np.argsort(eigen_values)
    sorted_idcs = sorted_idcs[::-1]
    sorted_evals = eigen_values[sorted_idcs]
    sorted_evecs = eigen_vectors[sorted_idcs]

    total_variance = np.sum(eigen_values)

    vsum = 0
    transform = np.empty([0,n])
    for i in range(len(sorted_evals)):
        vsum += sorted_evals[i]
        transform = np.vstack((transform,sorted_evecs[i]))
        if vsum / total_variance >= pcnt_variance:
            break
        
    assert vsum / total_variance >= pcnt_variance
        
    return transform
    
# lag_vals is interval on which to apply lags (e.g. 1:14)
def lagmatrix(A,lag_vals):
    n = A.shape[1]
    if np.size(lag_vals) == 1:
        lag_vals = range(lag_vals)
    LM = np.empty([ A.shape[0], A.shape[1] * len(lag_vals) ])
    for i in range(len(lag_vals)):
        LM[ (i+1): , (n*i):(n*(i+1)) ] = A[:A.shape[0]-i-1,:]
    return LM

def rmse_cent(A,B):
    A = (A - np.mean(A)) / np.std(A)
    B = (B - np.mean(B)) / np.std(B)
    return np.mean((A-B)**2)
    


