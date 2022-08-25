import support_fcns as f
import pandas
import numpy as np

# load and reformat
print("Loading data set...")
mf = pandas.read_excel(r'NEDOC_DATA.xlsx')
print("Load successful!\n")
score_arr_ = mf.Score.to_numpy()

score_arr_ = score_arr_.reshape(score_arr_.shape[0],1)
print("score_arr shape: ", score_arr_.shape)

score_arr = f.downsample(score_arr_,288,48)
print("score_arr shape after downsampling: ", score_arr.shape[0])

# get day matrix
daymat = f.getdaymat(score_arr,48)
print("daymat shape: ",daymat.shape)

# lagmatrix 
nlags = 14
lagmat = f.lagmatrix(daymat,nlags)      # first 14 rows contain NaNs
print("lagmat (pre-PCA) shape: ",lagmat.shape)
lagmat_cropped = lagmat[nlags:,:]
print("cropped lagmat (pre-PCA) shape: ",lagmat_cropped.shape)

# pca
mu1 = np.mean(lagmat_cropped,0)
sig1 = np.std(lagmat_cropped,0)

LMC_cent = (lagmat_cropped - mu1) / sig1
transform1 = f.PCA(LMC_cent,0.90)

X = np.dot(LMC_cent, transform1.T)
print("\nX shape",X.shape)













