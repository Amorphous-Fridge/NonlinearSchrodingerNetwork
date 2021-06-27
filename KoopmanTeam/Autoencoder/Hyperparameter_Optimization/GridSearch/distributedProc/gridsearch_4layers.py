import numpy as np
import os
import sys
from contextlib import redirect_stdout  #Used for writing model architecture to datafiles
import matplotlib.pyplot as plt         
from datetime import date               #Used for datafiles
import tensorflow as tf
import scipy
sys.path.append('/fslhome/mccutler/Koopman-Quantum/shared-git/NonlinearSchrodingerNetwork/KoopmanTeam/Autoencoder/Hyperparameter_Optimization')


#Some GPU configuration
#Always uses the 1st GPU avalible (if avalible) unless 1st line is uncommented, in which case no GPU is used

tf.config.set_visible_devices([], 'GPU') #uncomment to set tensorflow to use CPU
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) != 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

#PARAMETER SETUP
STATE_DIMENSION = 4    #Treating two complex dimensions as 4 real dimensions for now
                          #Vector will be [real1, imag1, real2, imag2]
ANTIKOOPMAN_DIMENSION = 3
os.environ['STATE_DIM'] = str(STATE_DIMENSION)
os.environ['ANTIK_DIM'] = str(ANTIKOOPMAN_DIMENSION)

initial_state = tf.keras.Input(shape = STATE_DIMENSION)
antikoop_state = tf.keras.Input(shape = ANTIKOOPMAN_DIMENSION)
    
#import all functions from QuantumAutoencoder
from QuantumAutoencoder import *

numLayers = 2
nn = np.asarray([8,16,32,64,128,256,512])
N = len(nn)
nni = np.arange(0,7)

modelnum = int(os.getenv('WIDTH'))
widths=[] #TODO: change for loop
for i2 in nni[numLayers//2:N]:
    for i3 in nni[numLayers//2:N]:
        for i1 in range(0,i2):
            for i4 in range(0,i3):
                if ((abs(i2-i3)>1)or(abs(i1-i4)>1))or\
                   ((abs(i2-i3)>0)and(abs(i1-i4)>0)):
                    continue
                widths.append((nn[i1],nn[i2],nn[i3],nn[i4])) #TODO:change num inputs

(width1,width2,width3,width4) = widths[modelnum] #TODO: change # outputs

##########################################ENCODER#################################################################### TODO:Change layer #'s
encoding_layer_1 = tf.keras.layers.Dense(width1, activation="selu", name='encoding_layer_1')(initial_state)
encoding_layer_2 = tf.keras.layers.Dense(width2, activation="selu", name='encoding_layer_2')(encoding_layer_1)
encoding_layer_3 = tf.keras.layers.Dense(width3, activation="selu", name='encoding_layer_3')(encoding_layer_2)
encoding_layer_4 = tf.keras.layers.Dense(width4, activation="selu", name='encoding_layer_4')(encoding_layer_3)
encoded_state = tf.keras.layers.Dense(ANTIKOOPMAN_DIMENSION, activation="selu", name='bottleneck')(encoding_layer_4)
#####################################################################################################################

#########################################DECODER#####################################################################
decoding_layer_1 = tf.keras.layers.Dense(width4, activation = "selu", name='decoding_layer_1')(antikoop_state)
decoding_layer_2 = tf.keras.layers.Dense(width3, activation = "selu", name='decoding_layer_2')(decoding_layer_1)
decoding_layer_3 = tf.keras.layers.Dense(width2, activation = "selu", name='decoding_layer_3')(decoding_layer_2)
decoding_layer_4 = tf.keras.layers.Dense(width1, activation = "selu", name='decoding_layer_4')(decoding_layer_3)
decoded_state = tf.keras.layers.Dense(STATE_DIMENSION, activation = "selu", name='decoded_layer')(decoding_layer_4)
#####################################################################################################################
#Model declarations
Phi = tf.keras.Model(inputs=initial_state, outputs = encoded_state, name='Phi')
Phi_inv = tf.keras.Model(inputs = antikoop_state, outputs = decoded_state, name='Phi_inv')
Autoencoder = tf.keras.models.Sequential([Phi, Phi_inv], name='Autoencoder')
#COMPILE AND TRAIN
Autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = .001), loss=L2_loss, metrics = ['mse','mae'], run_eagerly=False)
history = Autoencoder.fit(generate_pure_bloch(4096), steps_per_epoch=50,epochs=100) #remove generate_pure_bloch and replace with dataset
                                                     #generate_pure_bloch=4096,steps_per_epoch=50,epochs=100

#SAVE TO A FILE
write_history(history, [Autoencoder, Phi, Phi_inv], datadir='./Results4L/', batch_size='4096',modelnum=str(modelnum)) #TODO:change datadir

