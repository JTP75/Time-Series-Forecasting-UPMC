import numpy as np
import scipy
import pandas
import support_fcns as f
import tensorflow as tf
from scipy.interpolate import interp1d
import RNN_architecture as RNNA
import ClassDefs
import traceback as tb

# ===================================================================================================================================================================
print("")
RNNA.test_env()

# ===================================================================================================================================================================
print("")
print("Loading data set...")

try:
  mf = pandas.read_excel(r'NEDOC_DATA.xlsx')
except:
  print(tb.format_exc())
  
print("Load successful!")

# ===================================================================================================================================================================
print("")
fc = ClassDefs.Forecaster(mf)
fc.set_split(ClassDefs.Date(7,9,2021))
print("Today is", fc.td.dtstr(),"at index", fc.td_idx, "\n(i.e. this is the last day in the training set)")


# ===================================================================================================================================================================
print("")
print("Network data shapes:")
print("X shape:",fc.X.shape)
print("y shape:",fc.y.shape)

try:
  mdl = RNNA.build_net(fc.X.shape[1],fc.y.shape[1])
except:
  print(tb.format_exc())
  
mdl.summary()

# model parameters
minibatch_size = 64
max_epochs = 200
LR_base = 0.00144
dp = 93
df = 0.21244
steps_per_epoch = int(fc.X.shape[0]/minibatch_size)

LR_schedule = RNNA.LR_scheduler(LR_base,max_epochs,steps_per_epoch,df,dp)

try:
  mdl.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=LR_schedule),
    loss      = tf.keras.losses.Loss(),
    metric    = tf.keras.metrics.MeanSquaredError()
  )
except:
  print(tb.format_exc())
  exit()

'''

        solver, ...                                             % adam
        'MaxEpochs',            200, ...                        % max iters
        'GradientThreshold',    0.37957, ...                    % gradient boundary
        'InitialLearnRate',     0.00144, ...                    % initial learning rate
        'LearnRateSchedule',    "piecewise", ...                % how LR changes
        'LearnRateDropPeriod',  93, ...                         % how long between drops
        'LearnRateDropFactor',  0.21244, ...                    % lower LR by factor
        'MiniBatchSize',        64,...                          % nObs per iteration
        'Verbose',              true,...                        % whether to show info
        'Shuffle',              "every-epoch",...               % shuffle data
        'ExecutionEnvironment', mydevice...                     % gpu
'''



print("\n\nExecution Successful!\n\n")


