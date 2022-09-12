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

# ===================================================================================================================================================================
print("")
os.environ['KMP_DUPLICATE_LIB_OK']='True'
RNNA.test_env()

load_data = True
train_ntw = True


for i in range(0,len(sys.argv)-1,2):
  key = sys.argv[i]
  val = sys.argv[i+1]
  if key == "Loadset":
    load_data = val
  elif key == "Trainmdl":
    train_ntw = val

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

# Include the epoch in the file name (uses `str.format`)
checkpoint_path = "training_0/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
  filepath=checkpoint_path, 
  verbose=2, 
  save_weights_only=True,
  save_freq=5*minibatch_size
)

# Save the weights using the `checkpoint_path` format
mdl.save_weights(checkpoint_path.format(epoch=0))
latest = tf.train.latest_checkpoint(checkpoint_dir)
reload = True
if reload:
  mdl.load_weights(latest)

try:
  print("\nCompiling network object")
  mdl.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=LR_schedule),
    loss = tf.keras.losses.MeanSquaredError(),
    metrics = tf.keras.metrics.MeanSquaredError()
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
      verbose=2,
      workers=4,
      use_multiprocessing=True
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



print("\n\nExecution Successful!\n\n")


