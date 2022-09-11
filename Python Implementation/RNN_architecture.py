import numpy as np
import tensorflow as tf
from tensorflow import keras

def test_env():
    print('TensorFlow version', tf.__version__)
    gpu_count = len(tf.config.list_physical_devices('GPU'))
    print("Available GPU count:", gpu_count)

# cmds:   
# conda activate tf
# conda deactivate
# python "C:\Users\pacel\Desktop\ML_Work\UPMC\UPMC_ML_Research\Python Implementation\RNN_architecture.py"

def build_net(feat_count, resp_count):
    # layers
    input_layer = tf.keras.Input(name='input', shape=(feat_count,1,1))
    flatten_layer = tf.keras.layers.Flatten(name='flatten')
    reshape1 = tf.keras.layers.Reshape((feat_count,1),name="resh1")
    reshape2 = tf.keras.layers.Reshape((191,1),name="resh2")
    reshape3 = tf.keras.layers.Reshape((107,1),name="resh3")
    reshape4 = tf.keras.layers.Reshape((188*2,1),name="resh4")
    reshape5 = tf.keras.layers.Reshape((188*2,1),name="resh5")
    reshape6 = tf.keras.layers.Reshape((41*2,1),name="resh6")
    reshape7 = tf.keras.layers.Reshape((41*2,1),name="resh7")
    gru1 = tf.keras.layers.GRU(191,name='gru1',recurrent_initializer='HeNormal')
    gru2 = tf.keras.layers.LSTM(107,name='gru2',recurrent_initializer='HeNormal')
    drop1 = tf.keras.layers.Dropout(0.22651,name='drop1')
    drop2 = tf.keras.layers.Dropout(0.22651,name='drop2')
    drop3 = tf.keras.layers.Dropout(0.22651,name='drop3')
    drop4 = tf.keras.layers.Dropout(0.22651,name='drop4')
    bil1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(188,recurrent_initializer='HeNormal'),name='bil1')
    bil2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(188,recurrent_initializer='HeNormal'),name='bil2')
    bil3 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(41,recurrent_initializer='HeNormal'),name='bil3')
    bil4 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(41,recurrent_initializer='HeNormal'),name='bil4')
    bil5 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(9,recurrent_initializer='HeNormal'),name='bil5')
    fullconnect_layer = tf.keras.layers.Dense(resp_count,name='output')
    
    mdl = tf.keras.Sequential()
    mdl.add(input_layer)
    mdl.add(flatten_layer)
    mdl.add(reshape1)
    mdl.add(gru1)
    mdl.add(reshape2)
    mdl.add(gru2)
    mdl.add(drop1)
    mdl.add(reshape3)
    mdl.add(bil1)
    mdl.add(reshape4)
    mdl.add(bil2)
    mdl.add(drop2)
    mdl.add(reshape5)
    mdl.add(bil3)
    mdl.add(reshape6)
    mdl.add(bil4)
    mdl.add(drop3)
    mdl.add(reshape7)
    mdl.add(bil5)
    mdl.add(drop4)
    mdl.add(fullconnect_layer)
    
    return mdl


'''
MATLAB architecture:

        sequenceInputLayer([feat_count 1 1],'Name','input')
        flattenLayer('Name','flatten')
        
        gruLayer(191,'Name','gru1','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
        lstmLayer(107,'Name','gru2','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
        dropoutLayer(0.22651,'Name','drop1')
        bilstmLayer(188,'Name','bil1','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
        bilstmLayer(188,'Name','bil2','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
        dropoutLayer(0.22651,'Name','drop2')
        bilstmLayer(41,'Name','bil3','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
        bilstmLayer(41,'Name','bil4','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
        dropoutLayer(0.22651,'Name','drop3')
        bilstmLayer(9,'OutputMode',"last",'Name','bil5','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
        dropoutLayer(0.22651,'Name','drop4')
        
        fullyConnectedLayer(resp_count,'Name','fcOut')
        regressionLayer('Name','output')
'''

def LR_scheduler(LR_base,max_epochs,steps_per_epoch,drop_factor,drop_period):
    ndrops = int(max_epochs / drop_period)
    boundaries = []
    values = [LR_base]
    for i in range(ndrops):
        boundaries.append( (i+1) * drop_period * steps_per_epoch )
        values.append( LR_base * drop_factor**(i+1) )
    return tf.keras.optimzers.schedules.PiecewiseConstantDecay(boundaries=boundaries,values=values)
            