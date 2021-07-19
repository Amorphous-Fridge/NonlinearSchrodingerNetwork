import tensorflow as tf
import numpy as np
import os

import sys

from utils import *

####Theese are the only values that need to be configured####
DATADIR='/fslhome/wilhiteh/datasets/'
TIMESTEP = 0.1
TIMERANGE = 50
MAXPOINTS = 5000000
MEMFIT = False
#############################################################

points_per_file = int(np.ceil(TIMERANGE/TIMESTEP))
if MEMFIT:
    num_files = int(np.floor(MAXPOINTS/points_per_file))
    dataset = '0t{}_{}inits_dt{}_memfit_compressed/'.format(TIMERANGE,num_files,str(TIMESTEP).replace('.','p'))
else:
    num_files = 50000
    dataset = '0t{}_{}inits_dt{}_compressed/'.format(TIMERANGE,num_files,str(TIMESTEP).replace('.','p'))

timestep = str(TIMESTEP).replace('.','p')

#Set dataset sizes to test
dataset_sizes = (2000, 3000, 4000, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000)

size = int(os.getenv('SLURM_ARRAY_TASK_ID'))

datasize = dataset_sizes[size]

EPOCHS = int(os.getenv('EPOCHS'))

NAME = '0t{}_dt{}_{}inits'.format(TIMERANGE,timestep,datasize)
#NAME = 'NONMEMFITTEST'

#Some GPU configuration
#Always uses the 1st GPU avalible (if avalible) unless 1st line is uncommented, in which case no GPU is used

#tf.config.set_visible_devices([], 'GPU') #uncomment to set tensorflow to use CPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) != 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
elif len(physical_devices) == 0:
    print("Warning: No GPU detected.  Running tensorflow on CPU")
    

#Import autoencoder functions, needed for comrpessing the data into a space the nonlinear function can train on
Phi = tf.keras.models.load_model('/fslhome/wilhiteh/Autoencoder/trial25e1000Phi.h5', compile=False, custom_objects={'Functional':tf.keras.models.Model})
Phi_inv = tf.keras.models.load_model('/fslhome/wilhiteh/Autoencoder/trial25e1000Phi_inv.h5', compile=False, custom_objects={'Functional':tf.keras.models.Model})

def L2_loss(y_true, y_pred):
    return tf.norm(y_true-y_pred, ord=2, axis=-1)
    
 
def get_evolution_data(evolution_file):

    data = tf.data.experimental.CsvDataset(evolution_file, [tf.float32, tf.float32, tf.float32], select_cols=[1,2,3],header=True)

    data = data.map(lambda a,b,c: tf.stack([a,b,c]), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    pre_evolution = data.take(tf.data.experimental.cardinality(data)-1)
    post_evolution = data.skip(1).take(tf.data.experimental.cardinality(data)-1)

    ds = tf.data.Dataset.zip((pre_evolution,post_evolution))

    return ds


def interleave_evolutions(datadir, max_evolution, training_evolutions):
    
    training_filenames = [datadir+'evolution{}.csv'.format(x) for x in range(training_evolutions)]
    validation_filenames = [datadir+'evolution{}.csv'.format(x) for x in range(training_evolutions, max_evolution)]
    training_dataset = tf.data.Dataset.from_tensor_slices(training_filenames)
    validation_dataset = tf.data.Dataset.from_tensor_slices(validation_filenames)
    training_dataset = training_dataset.interleave(lambda x: get_evolution_data(x),num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(1000000, reshuffle_each_iteration=True)  
    validation_dataset = validation_dataset.interleave(lambda x: get_evolution_data(x),num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(1000000, reshuffle_each_iteration=True)  
    return training_dataset, validation_dataset


#This function leads to much quicker training but
##THE DATASET MUST FIT INTO MEMORY
def get_multiple_evolutions(datadir, num_evolutions, train_evolutions):

    pre_evolution = []
    post_evolution = []

    for i in range(train_evolutions):

        with open(datadir+'evolution{}.csv'.format(i), 'r') as f:
            data = np.genfromtxt(f, delimiter=',', skip_header=1)[:,1:]

            pre_evolution.append(data[0])
            for k in data[1:-1]:
                pre_evolution.append(k)
                post_evolution.append(k)
            post_evolution.append(data[-1])

    train = tf.data.Dataset.from_tensor_slices((pre_evolution, post_evolution))

    pre_evolution = []
    post_evolution = []

    for i in range(train_evolutions, num_evolutions):

        with open(datadir+'evolution{}.csv'.format(i), 'r') as f:
            data = np.genfromtxt(f, delimiter=',', skip_header=1)[:,1:]

            pre_evolution.append(data[0])
            for k in data[1:-1]:
                pre_evolution.append(k)
                post_evolution.append(k)
            post_evolution.append(data[-1])

    test = tf.data.Dataset.from_tensor_slices((pre_evolution, post_evolution))

    return train.shuffle(1000000, reshuffle_each_iteration=True), test.shuffle(1000000, reshuffle_each_iteration=True)


inputs = tf.keras.Input(shape=3)

if MEMFIT:
    train_data, val_data = get_multiple_evolutions(DATADIR+dataset,datasize,int(0.9*datasize))
else:
    print('MEMFIT false, using interleaving solution')
    train_data, val_data = interleave_evolutions(DATADIR+dataset,datasize,int(0.9*datasize))

train_data = train_data.batch(100000)
val_data = val_data.batch(100000)

if not MEMFIT:
    print('Enabling prefetching')
    train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)
    val_data = val_data.prefetch(tf.data.experimental.AUTOTUNE)

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
history = NonlinearEvolution.fit(train_data, validation_data=val_data, epochs=EPOCHS)
write_history(history, NonlinearEvolution,loss='L2_loss',batch_size='100000', 
              other_info={'dataset':dataset+' (First {} evolutions)'.format(datasize),
              'validation':'{}inits'.format(int(0.1*datasize))}, 
              savedname=NAME+'.data',
              datadir='/fslhome/wilhiteh/datafiles/')


#Second 250 epochs w/ lower learning rate
NonlinearEvolution.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.0001), loss=L2_loss, metrics=['mse', 'mae'])
history = NonlinearEvolution.fit(train_data, validation_data=val_data, epochs=EPOCHS)
append_history(history, trial=NAME+'.data', 
               params={'Learning Rate':.0001}, datadir='/fslhome/wilhiteh/datafiles/')

#Save the model and label it w/ amount of data it was trained on
NonlinearEvolution.save('/fslhome/wilhiteh/models/'+NAME+'.h5')
