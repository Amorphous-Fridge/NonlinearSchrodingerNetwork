import tensorflow as tf
import numpy as np
import os

import sys

sys.path.append('../../..')
from utils import *
sys.path.remove('../../..')


#Some GPU configuration
#Always uses the 1st GPU avalible (if avalible) unless 1st line is uncommented, in which case no GPU is used

#tf.config.set_visible_devices([], 'GPU') #uncomment to set tensorflow to use CPU
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) != 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
elif len(physical_devices) == 0:
    print("Warning: No GPU detected.  Running tensorflow on CPU")
    

#Import autoencoder functions, needed for comrpessing the data into a space the nonlinear function can train on
Phi = tf.keras.models.load_model('../../../Autoencoder/Autoencoder_Trials/models/trial25e1000Phi.h5', compile=False)
Phi_inv = tf.keras.models.load_model('../../../Autoencoder/Autoencoder_Trials/models/trial25e1000Phi_inv.h5', compile=False)

def L2_loss(y_true, y_pred):
    return tf.norm(y_true-y_pred, ord=2, axis=-1)
    
    
def get_multiple_evolutions(datadir, num_evolutions, train_evolutions, ideal_pre_compress_3D=False, pre_compress_phi=False):
    '''Read in the evolution of multiple initial conditions at once.  Each initial condition should 
    have its own csv file and be named in the format "evolution{}.csv", where {} indicates an integer.
    Note that there will be 1 fewer timesteps per evolution file than are present in the evolution file,
    since the last step does not have an evolution.
    
    PARAMS:
    -------
    str datadir: Path to directory containing evolution files.
    int num_evolutions: The number of initial conditions to read in.
    int train_evolutions: The number of initial conditions that should be treated as 
                          training evolutions
    
    RETURNS:
    --------
    (tf.data.Dataset, tf.data.Dataset): A tuple of the training and validation datasets, respectivley.
    '''
    
    pre_evolution = []
    post_evolution = []
    
    #Read training data
    for i in range(train_evolutions):
        with open(datadir+'evolution{}.csv'.format(i), 'r') as f:
            data = np.genfromtxt(f, delimiter=',', skip_header=1)[:,1:]
            
            if ideal_pre_compress_3D:
                data = [ideal_phi_3D(x) for x in data]
            elif pre_compress_phi:
                data = Phi(data)
            
            pre_evolution.append(data[0])
            
            for k in data[1:-1]:
                pre_evolution.append(k)
                post_evolution.append(k)
            
            post_evolution.append(data[-1])
            
    train = tf.data.Dataset.from_tensor_slices((pre_evolution, post_evolution))
    
    
    #Read test data
    pre_evolution = []
    post_evolution = []
    
    for i in range(train_evolutions, num_evolutions):
        with open(datadir+'evolution{}.csv'.format(i), 'r') as f:
            data = np.genfromtxt(f, delimiter=',', skip_header=1)[:,1:]
            
            if ideal_pre_compress_3D:
                data = [ideal_phi_3D(x) for x in data]
            elif pre_compress_phi:
                data = Phi(data)
            
            
            pre_evolution.append(data[0])
            
            for k in data[1:-1]:
                pre_evolution.append(k)
                post_evolution.append(k)
            
            post_evolution.append(data[-1])
            
    test = tf.data.Dataset.from_tensor_slices((pre_evolution, post_evolution))
    
    return train.shuffle(1000000, reshuffle_each_iteration=True), test.shuffle(1000000, reshuffle_each_iteration=True)
    




inputs = tf.keras.Input(shape=3)

#Set dataset sizes to test
dataset_sizes = (2000, 3000, 4000, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000)

for datasize in dataset_sizes:


    train_data, val_data = get_multiple_evolutions('../../../QuantumTeam/data/50000inits_dt0p01/', datasize, int(0.9*datasize), pre_compress_phi=True)
    train_data = train_data.batch(100000)
    val_data = val_data.batch(100000)


    #Re-declare model each time to re-initilize parameters
    nonlinear_layer_1 = tf.keras.layers.Dense(64, activation='selu', name='nonlinear_layer_1')(inputs)
    nonlinear_layer_2 = tf.keras.layers.Dense(256, activation='selu', name='nonlinear_layer_2')(nonlinear_layer_1)
    nonlinear_layer_3 = tf.keras.layers.Dense(512, activation='selu', name='nonlinear_layer_3')(nonlinear_layer_2)
    nonlinear_layer_4 = tf.keras.layers.Dense(512, activation='selu', name='nonlinear_layer_4')(nonlinear_layer_3)
    nonlinear_layer_5 = tf.keras.layers.Dense(256, activation='selu', name='nonlinear_layer_5')(nonlinear_layer_4)
    nonlinear_layer_6 = tf.keras.layers.Dense(64, activation='selu', name='nonlinear_layer_6')(nonlinear_layer_5)
    evolved = tf.keras.layers.Dense(3, activation='selu', name='evolved_state_layer')(nonlinear_layer_6)

    NonlinearEvolution = tf.keras.Model(inputs=inputs, outputs=evolved)



    #First 250 epochs w/ high learning rate
    NonlinearEvolution.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.001), loss=L2_loss, metrics=['mse', 'mae'])
    history = NonlinearEvolution.fit(train_data, validation_data=val_data, epochs=250)
    write_history(history, NonlinearEvolution,loss='L2_loss',batch_size='100000', other_info={'dataset':'50000inits_dt0p1 (First {} evolutions)'.format(datasize),'validation':'{}inits'.format(int(0.9*datasize))})

    #Second 250 epochs w/ lower learning rate
    NonlinearEvolution.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.0001), loss=L2_loss, metrics=['mse', 'mae'])
    history = NonlinearEvolution.fit(train_data, validation_data=val_data, epochs=250)
    append_history(history, params={'Learning Rate':.0001})
    
    #Save the model and label it w/ amount of data it was trained on
    NonlinearFunction.save('./models/{}inits_dt0p1.h5'.format(datasize))
