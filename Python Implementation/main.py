import numpy as np
import scipy
import pandas
import support_fcns as f
import tensorflow as tf
from scipy.interpolate import interp1d
import RNN_architecture as RNNA
import ClassDefs
import traceback as tb
import os
import sys

# process sys args and test environment
# ===================================================================================================================================================================
print("")
os.environ['KMP_DUPLICATE_LIB_OK']='True'
RNNA.test_env()

load_data = True
train_ntw = True
restart_training = False


for i in range(0,len(sys.argv)-1,2):
  key = sys.argv[i]
  val = sys.argv[i+1]
  if key == "Loadset":
    load_data = val
  elif key == "Trainmdl":
    train_ntw = val
  elif key == "Restart":
    restart_training = val

# load data set
# ===================================================================================================================================================================
print("")

if load_data:
  print("Loading data set...")

  try:
    mf = pandas.read_excel(r'NEDOC_DATA.xlsx')
  except:
    print(tb.format_exc())
    
  print("Load successful!")
else:
  print("Set-load routine aborted.")

# instantiate forecaster object
# ===================================================================================================================================================================
print("")
fc = ClassDefs.Forecaster(mf)
fc.set_split(ClassDefs.Date(7,9,2021))
print("Today is", fc.td.dtstr(),"at index", fc.td_idx, "\n(i.e. this is the last day in the training set)")
print("Network data shapes:")
print("X shape:",fc.X.shape)
print("y shape:",fc.y.shape)

# construct network (architecture)
# ===================================================================================================================================================================
print("")

try:
  mdl = RNNA.build_net(fc.X.shape[1],fc.y.shape[1])
except:
  print(tb.format_exc())
  
mdl.summary()

# construct network (options)
# ===================================================================================================================================================================
print("")
minibatch_size = 64     # how many samples are processed at once (larger values will train in fewer iterations)
max_epochs = 200        
LR_base = 0.00144       
dp = 93
df = 0.21244
steps_per_epoch = int(fc.X.shape[0]/minibatch_size)
LR_schedule = RNNA.LR_scheduler(LR_base,max_epochs,steps_per_epoch,df,dp)   # learning rate schedule (constant multiple decay)
checkpoint_path = "training_0/cp-{epoch:04d}.ckpt"      # file location (relative) of trained network parameters & checkpoints
checkpoint_dir = os.path.dirname(checkpoint_path)       # directory of above ^^
cp_callback = tf.keras.callbacks.ModelCheckpoint(       # checkpoint object
  filepath=checkpoint_path, 
  verbose=1, 
  save_weights_only=True,
  save_freq=500
)

mdl.save_weights(checkpoint_path.format(epoch=0))   # save initial weights

if not restart_training:
  latest = tf.train.latest_checkpoint(checkpoint_dir)
  mdl.load_weights(latest)
else:
  print("network parameters have been re-initialized")

try:
  print("\nCompiling network object")
  mdl.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=LR_schedule),
    loss = tf.keras.losses.MeanSquaredError(),
    metrics = tf.keras.metrics.RootMeanSquaredError()
  )
except:
  print("\nError in compiling routine:\n")
  print(tb.format_exc())
  exit()

if train_ntw:
  print("\nTraining network...")
  try:
    mdl.fit(
      x=fc.X,
      y=fc.y,
      batch_size=minibatch_size,
      callbacks=[cp_callback],
      epochs=max_epochs,
      verbose=1,
      workers=8,
      use_multiprocessing=True,
      shuffle=True
    )
  except:
    print("\nError in training routine:\n")
    print(tb.format_exc())
    exit()
  mdl.save('training_0/TrainedModel_00')
else:
  print("\nTraining aborted.")



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

# evaluate
# ===================================================================================================================================================================
output = mdl.predict(
  fc.Xts,
  batch_size=minibatch_size,
  verbose=1
)
yp = np.dot(output,fc.transformR)
yp = fc.sigR * yp + fc.muR
ya = fc.sigR * np.dot(fc.yts,fc.transformR) + fc.muR

assert yp.shape == ya.shape, f"predicition and actual shapes not the same."
shp = yp.shape

yp = np.reshape(yp,(shp[0]*shp[1],1))
ya = np.reshape(ya,(shp[0]*shp[1],1))
print(f.mse_cent(ya,yp))

print("\n\nExecution Successful!\n\n")


# MT: 0.3680
# TR: 0.3531