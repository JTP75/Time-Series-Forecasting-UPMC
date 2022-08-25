import numpy as np
from scipy.interpolate import interp1d

def _downsample_(arr,npts):
    interpolated = interp1d(np.arange(len(arr)), arr, axis = 0, fill_value = 'extrapolate')
    downsampled = interpolated(np.linspace(0, len(arr), npts))
    return downsampled

def downsample(arr,ppd_init,ppd_desired):
    ndays = round(len(arr) / ppd_init)
    newsize = ndays * ppd_desired
    newarr = _downsample_(arr,newsize)
    return newarr

def getdaymat(X,ppd):
    return X.reshape(round(len(X)/ppd),ppd)

def PCA(A,pcnt_variance):
    # A should be centered (or standardized)
    # compute covariance matrix
    covariance_matrix = np.dot(A.T,A)

    # find largest eigenvalues
    n = A.shape[1]
    eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
    sorted_idcs = np.argsort(eigen_values)
    sorted_evals = np.flip(eigen_values[sorted_idcs])
    sorted_evecs = np.flip(eigen_vectors[sorted_idcs])

    sorted_evals /= np.sum(eigen_values)

    vsum = 0
    transform = np.empty([0,n])
    for i in range(len(sorted_evals)):
        vsum += sorted_evals[i]
        transform = np.vstack((transform,sorted_evecs[i]))
        if vsum >= pcnt_variance:
            break
    return transform
    
# lag_vals is interval on which to apply lags (e.g. 1:14)
def lagmatrix(A,lag_vals):
    n = A.shape[1]
    if np.size(lag_vals) == 1:
        lag_vals = range(lag_vals)
    LM = np.empty([ A.shape[0], A.shape[1] * len(lag_vals) ])
    for i in range(len(lag_vals)):
        LM[ (i+1): , (n*i):(n*(i+1)) ] = A[i+1:,:]
    return LM

        
    


