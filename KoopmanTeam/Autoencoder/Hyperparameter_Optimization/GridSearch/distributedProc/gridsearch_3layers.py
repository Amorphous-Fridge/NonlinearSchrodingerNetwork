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

numLayers = 3
nn = np.asarray([8,16,32,64,128,256,512])
N = len(nn)
nni = np.arange(0,7)

modelnum = int(os.getenv('WIDTH'))
widths=[]
for i2 in nni[numLayers//2:N]:
    for i1 in range(0,i2):
        for i3 in range(0,i2):
            widths.append((nn[i1],nn[i2],nn[i3]))

(width1,width2,width3) = widths[modelnum]

##########################################ENCODER####################################################################
encoding_layer_1 = tf.keras.layers.Dense(width1, activation="selu", name='encoding_layer_1')(initial_state)
encoding_layer_2 = tf.keras.layers.Dense(width2, activation="selu", name='encoding_layer_2')(encoding_layer_1)
encoding_layer_3 = tf.keras.layers.Dense(width3, activation="selu", name='encoding_layer_3')(encoding_layer_2)
encoded_state = tf.keras.layers.Dense(ANTIKOOPMAN_DIMENSION, activation="selu", name='bottleneck')(encoding_layer_3)
#####################################################################################################################

#########################################DECODER#####################################################################
decoding_layer_1 = tf.keras.layers.Dense(width3, activation = "selu", name='decoding_layer_1')(antikoop_state)
decoding_layer_2 = tf.keras.layers.Dense(width2, activation = "selu", name='decoding_layer_2')(decoding_layer_1)
decoding_layer_3 = tf.keras.layers.Dense(width1, activation = "selu", name='decoding_layer_3')(decoding_layer_2)
decoded_state = tf.keras.layers.Dense(STATE_DIMENSION, activation = "selu", name='decoded_layer')(decoding_layer_3)
#####################################################################################################################

#Learning Rate Scheduler
def scheduler(epoch):
    if epoch<250:
        return 0.001
    if epoch<750:
        return 0.0001
    else:
        return 0.00001

#Model declarations
Phi = tf.keras.Model(inputs=initial_state, outputs = encoded_state, name='Phi')
Phi_inv = tf.keras.Model(inputs = antikoop_state, outputs = decoded_state, name='Phi_inv')
Autoencoder = tf.keras.models.Sequential([Phi, Phi_inv], name='Autoencoder')
#COMPILE AND TRAIN
Autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = .001), loss=L2_loss, metrics = ['mse','mae'], run_eagerly=False)
callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
history = Autoencoder.fit(generate_pure_bloch(4096), steps_per_epoch=5,epochs=2,callbacks=[callback]) #remove generate_pure_bloch and replace with dataset
                                                     #generate_pure_bloch=4096,steps_per_epoch=50,epochs=100

#SAVE TO A FILE
write_history(history, [Autoencoder, Phi, Phi_inv], datadir='./Results3L/', batch_size='4096',modelnum=str(modelnum))
#SAVE THE MODEL! :D
Phi.save('/fslhome/mccutler/Koopman-Quantum/shared-git/NonlinearSchrodingerNetwork/KoopmanTeam/Autoencoder/Hyperparameter_Optimization/GridSearch/distributedProc/SavedModels/Phi_'+width1+'-'+width2+'-'+width3)
Phi_inv.save('/fslhome/mccutler/Koopman-Quantum/shared-git/NonlinearSchrodingerNetwork/KoopmanTeam/Autoencoder/Hyperparameter_Optimization/GridSearch/distributedProc/SavedModels/Phi_inv_'+width1+'-'+width2+'-'+width3)
Autoencoder.save('/fslhome/mccutler/Koopman-Quantum/shared-git/NonlinearSchrodingerNetwork/KoopmanTeam/Autoencoder/Hyperparameter_Optimization/GridSearch/distributedProc/SavedModels/Autoencoder_'+width1+'-'+width2+'-'+width3)

