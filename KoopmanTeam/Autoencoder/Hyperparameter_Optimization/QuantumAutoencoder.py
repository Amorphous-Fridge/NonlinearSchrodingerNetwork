from contextlib import redirect_stdout  #Used for writing model architecture to datafiles
import matplotlib.pyplot as plt         
from datetime import date               #Used for datafiles
import tensorflow as tf
import numpy as np
import scipy
import os
import sys
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
ANTIKOOPMAN_DIMENSION = 2

#DATA GENERATION
def generate_pure_bloch(batch_size=16):
    '''Generate random pure states on the bloch sphere.
    These are two complex dimensional vectors with an L2 norm of 1.
    Note that the state dimension of the Bloch sphere is always 4.
    '''
    bloch_state_dimension = 4
    while True:
        states = np.empty([batch_size, bloch_state_dimension])
        for i in range(batch_size):
            x1,y1,x2,y2 = np.random.random(4)
            norm = np.sqrt(x1*x1 + y1*y1 + x2*x2 + y2*y2)
            states[i] = 1/norm * np.array([x1,y1, x2,y2])
        yield (states, states) #autoencoder, so data and label are the same thing

#fix one component to zero, do random select, repeat through each component?
#fix some epsilon instead of zero?
#would be easy to exclude from training set at least (check that no component is zero or maybe within some epsilon of zero, else re-draw)
def generate_pure_bloch_val(batch_size=4096):
    bloch_state_dimension = 4
    epsilon_max = 1e-5
    while True:
        states = np.empty([batch_size, bloch_state_dimension])
        for i in range(batch_size//16):
            fixed1, fixed2, fixed3 = np.random.uniform(low=-1, high=1, size=3)
            epsilon = epsilon_max * np.random.uniform(low = -1, high = 1, size = 1)
            norm = np.sqrt(fixed1*fixed1 + fixed2*fixed2 + fixed3*fixed3 + epsilon*epsilon)
            states[16*i] = 1/norm * np.array([epsilon, fixed1, fixed2, fixed3])
            states[16*i+1] = 1/norm * np.array([fixed1, epsilon, fixed2, fixed3])
            states[16*i+2] = 1/norm * np.array([fixed1, fixed2, epsilon, fixed3])
            states[16*i+3] = 1/norm * np.array([fixed1, fixed2, fixed3, epsilon])
            states[16*i+4] = -1/norm * np.array([epsilon, fixed1, fixed2, fixed3])
            states[16*i+5] = -1/norm * np.array([fixed1, epsilon, fixed2, fixed3])
            states[16*i+6] = -1/norm * np.array([fixed1, fixed2, epsilon, fixed3])
            states[16*i+7] = -1/norm * np.array([fixed1, fixed2, fixed3, epsilon])
            
            fixed1, fixed2, fixed3 = np.random.uniform(low=0.5-epsilon_max, high=0.5+epsilon_max, size=3)
            epsilon = np.sqrt(1-fixed1*fixed1 - fixed2*fixed2 - fixed3*fixed3)
            states[16*i+8] = np.array([epsilon, fixed1, fixed2, fixed3])
            states[16*i+9] = np.array([fixed1, epsilon, fixed2, fixed3])
            states[16*i+10] = np.array([fixed1, fixed2, epsilon, fixed3])
            states[16*i+11] = np.array([fixed1, fixed2, fixed3, epsilon])
            states[16*i+12] = -1 * np.array([epsilon, fixed1, fixed2, fixed3])
            states[16*i+13] = -1 * np.array([fixed1, epsilon, fixed2, fixed3])
            states[16*i+14] = -1 * np.array([fixed1, fixed2, epsilon, fixed3])
            states[16*i+15] = -1 * np.array([fixed1, fixed2, fixed3, epsilon])
          
            
        yield(states, states)

def generate_pure_bloch_test(batch_size=4096):
    bloch_state_dimension = 4
    epsilon_max = 1e-5
    while True:
        states = np.empty([batch_size, bloch_state_dimension])
        
        for i in range(batch_size):
            x1, y1, x2, y2 = np.random.uniform(low=-1, high=1, size=4)
            norm = np.sqrt(x1*x1 + y1*y1 + x2*x2 + y2*y2)
            state = 1/norm * np.array([x1, y1, x2, y2])
            #Remove any elements from our validation set
            state[np.abs(state)<=epsilon_max] += 3*epsilon_max
            state[np.abs(state-0.5)<=epsilon_max] += 3*epsilon_max
            states[i] = state
            
        yield(states, states)

###Input layers for the encoder and decoder, respectivley
#initial_state = tf.keras.Input(shape = STATE_DIMENSION)
#antikoop_state = tf.keras.Input(shape = ANTIKOOPMAN_DIMENSION)
#
###########################################ENCODER####################################################################
#encoding_layer_1 = tf.keras.layers.Dense(16, activation="selu", name='encoding_layer_1')(initial_state)
#encoding_layer_2 = tf.keras.layers.Dense(64, activation="selu", name='encoding_layer_2')(encoding_layer_1)
#encoding_layer_3 = tf.keras.layers.Dense(128, activation="selu", name='encoding_layer_3')(encoding_layer_2)
#encoding_layer_4 = tf.keras.layers.Dense(64, activation="selu", name='encoding_layer_4')(encoding_layer_3)
#encoding_layer_5 = tf.keras.layers.Dense(16, activation="selu", name='encoding_layer_5')(encoding_layer_4)
#encoded_state = tf.keras.layers.Dense(ANTIKOOPMAN_DIMENSION, activation="selu", name='bottleneck')(encoding_layer_5)
######################################################################################################################
#
##########################################DECODER#####################################################################
#decoding_layer_1 = tf.keras.layers.Dense(16, activation = "selu", name='decoding_layer_1')(antikoop_state)
#decoding_layer_2 = tf.keras.layers.Dense(64, activation = "selu", name='decoding_layer_2')(decoding_layer_1)
#decoding_layer_3 = tf.keras.layers.Dense(128, activation = "selu", name='decoding_layer_3')(decoding_layer_2)
#decoding_layer_4 = tf.keras.layers.Dense(64, activation = "selu", name='decoding_layer_4')(decoding_layer_3)
#decoding_layer_5 = tf.keras.layers.Dense(16, activation = "selu", name='decoding_layer_5')(decoding_layer_4)
#decoded_state = tf.keras.layers.Dense(STATE_DIMENSION, activation = "selu", name='decoded_layer')(decoding_layer_5)
######################################################################################################################



#Model declarations
#Phi = tf.keras.Model(inputs=initial_state, outputs = encoded_state, name='Phi')
#Phi_inv = tf.keras.Model(inputs = antikoop_state, outputs = decoded_state, name='Phi_inv')
#
#Autoencoder = tf.keras.models.Sequential([Phi, Phi_inv], name='Autoencoder')

#UTILITY FUNCTIONS

def L2_loss(y_true, y_pred):
    '''The L2 norm of the input vector
    and the autoencoded vector'''
    return tf.norm(y_true-y_pred, ord = 2)


def get_relative_phase(vector):
    '''Returns the relative phase between
    the two complex components of a two
    complex dimensional vector
    Assumes the vector is passed in as a 
    four dimensional real row vector of form
    [real1, imag1, real2, imag2]
    '''
    

    #Tensorflow likes to return a list of a single
    #element sometimes, which breaks this function
    #This does not happen during training, only when
    #manually run on a single vector
    if vector.shape[0] == 1:
        vector = vector[0]
    
    return tf.atan2(vector[1],vector[0]) - tf.atan2(vector[3],vector[2])
    


def autoencoding_loss(y_true, y_pred):
    '''
    Autoencoding loss accounting for magnitude of
    input/output vector and the relative phase
    of the two complex components of the
    input/output vectors (we don't care if the 
    autoencoder rotates both components, so long
    as it rotates them both equally)
    '''
    y_true_L2 = tf.norm(y_true, ord=2)
    y_pred_L2 = tf.norm(y_pred, ord=2)
    
    return tf.abs(y_true_L2 - y_pred_L2) + tf.abs(get_relative_phase(y_true) - get_relative_phase(y_pred))


#def predict_single_state(state, encoder = Phi, decoder = Phi_inv):
    #'''Outputs the prediction of a single 
    #state.  Primarily for sanity checks.
    #'''
    #encoded = encoder(np.array([state,]))
    #decoded = decoder(encoder(np.array([state,])))
    #input_norm = np.linalg.norm(state, ord=2)
    #output_norm = np.linalg.norm(decoded.numpy(), ord=2)
    #input_rel_phase = get_relative_phase(state).numpy()
    #output_rel_phase = get_relative_phase(decoded.numpy()).numpy()
    #print('Initial State:{}\nEncoded State:{}\nDecoded State:{}\nInput Norm:{}\nOutput Norm:{}\nInput Relative Phase:{}\nOutput Relative Phase:{}\nNorm Difference:{}\nPhase Difference:{}\nLoss:{}'.format(
            #state, encoded.numpy(), decoded.numpy(), input_norm, output_norm,
            #input_rel_phase, output_rel_phase, np.abs(input_norm-output_norm), 
            #np.abs(input_rel_phase-output_rel_phase), 
            #np.abs(input_norm-output_norm)+np.abs(input_rel_phase-output_rel_phase)))
          #
    #return None

###################DATA WRITING FUNCTIONS#####################

##############################################################
def write_history(history, model, loss = 'autoencoding_loss', 
                  optimizer='Adam', lr='.001', 
                  batch_size='1024', datadir='./Results/'):
    '''Writes training history to a datafile
    This will create a new trial datafile.  If the model has 
    had additional training, append_history should be used instead.
    
    PARAMS:
    -------
    history - The history callback returned by the model.fit method in keras
    model - The model (or list of models) which we want to write the architecture of to the datafile
    string loss - The loss function the model was trained on
    string optimizer - The optimizer the model was compiled with
    string lr - The learning rate the model was initially compiled with
    string spe - The number of steps per epoch
    string batch_size - The number of samples trained on per step
    string datadir - Directory where the datafiles are stored
    '''
    
    rundatadir = datadir
    filename = 'trial'+str(len(os.listdir(rundatadir)))

    with open(rundatadir+filename+'.data', 'w') as f:
        f.write(str(date.today())+'\n')
        for key in history.history.keys():
            f.write(key+',')
            for epoch in range(history.params['epochs']):
                f.write(str(history.history[key][epoch])+',')
            f.write('\n')
        f.write("Loss,{}\nOptimizer,{}\nLearning Rate,{}\nSteps Per Epoch,{}\nBatch Size,{}\nEpochs,{}\n".format(loss,optimizer,lr,history.params['steps'],batch_size, history.params['epochs']))
        f.write('\n')
        with redirect_stdout(f):
            if type(model) == list:
                for i in model:              
                    i.summary()
            else:
                model.summary()
    return rundatadir+filename+'.data'

#####################################################################
#####################################################################

def append_history(history, trial, datadir='./Results/', params_update = True, 
                   params = {'Loss':None, 'Optimizer':None, 'Learning Rate':None, 'Batch Size':None}):
    '''Appends new training data to trial datafile.
    This will only work with datafiles written using write_history (or files of identical form)
    
    PARAMS:
    -------
    history - The history callback containing the new run's data
    int trial - The trial number to append the data to.  If we want to append data to 
            trial43.data, this would be 43
    str datadir - The directory containing the datafile
    bool params_update - Boolean indicating if we should update parameters (loss used, optimizer, etc.)
                    in addition to adding the new loss data
    dict params - Dictionary containing updated parameter values
                  If parameter is not included, the previous value 
                  written for that parameter will be repeated
    '''
    
    filename = 'trial'+str(trial)+'.data'
    
    
    newlines = []
    keys = history.history.keys()
    
    
    for k in ['Loss', 'Optimizer', 'Learning Rate', 'Batch Size']:
        if k not in params.keys():
            params[k] = None
    params['Epochs'] = history.params['epochs']
    params['Steps Per Epoch'] = history.params['steps']
   
      
    #Read old data and add in new data as it is read
    with open(datadir+filename, 'r') as f:
        for line in f.readlines():
            
            tag = line.split(',')[0]
            
            if tag in keys:
                newdata = [str(x) for x in history.history[tag]]
                newlines.append(line.split(',')[:-1] + newdata ) #[:-1] to drop the newline
            elif (tag in params.keys() and params_update == True):
                if params[tag] is None:
                    newlines.append(line.strip().split(',')[:] + [line.strip().split(',')[-1]])
                else:
                    newdata = [str(params[tag])]
                    newlines.append(line.strip().split(',')[:] + newdata)
            else:
                newlines.append(line)
    
    #Write the old data with the appended data
    with open(datadir+filename, 'w') as f:
        for el in newlines:
            if type(el) == list:
                f.write(','.join(el))
                f.write('\n')
            else:
                f.write(el)
    return
    
    
#####################################################################
#####################################################################
    
def loss_plot(trial, datadir='./datafiles/', savefig = True,
              figdir = './figures/', logplot=False,
              metric = None, mark_runs = False, 
              skip_epochs=0, mark_lowest = True):
    '''Creates a plot of the loss/metric for the given trial number
    '''
    
    if metric == None:
        metric = 'loss'

    losses = []
    runs = []
    
    #Read in the data
    with open(datadir+'trial'+str(trial)+'.data', 'r') as f:
        for line in f.readlines():
            if line.split(',')[0] == metric:
                losses = [float(x) for x in line.strip().split(',')[1:]]
                if not mark_runs:
                    break
            elif (line.split(',')[0] == 'Epochs' and mark_runs == True):
                runs = ['0.']+line.strip().split(',')[1:]
                runs = [float(runs[i-1])+float(runs[i-2]) for i in range(2, len(runs))]
                break
            
    
    fig, ax = plt.subplots(1,1, figsize = (8,8))


    ax.plot(range(len(losses[skip_epochs:])), losses[skip_epochs:])
    if mark_runs:
        for i in runs:
            ax.plot([i,i], ax.get_ylim(), c='black', ls=':')
    if mark_lowest:
        lowest = min(losses)
        ax.plot(losses.index(lowest), lowest, 'go')
 #       ax.text(ax.get_xlim()[1]-0.05*ax.get_xlim()[0], ax.get_ylim()[1]-0.05*ax.get_ylim()[0], 'Lowest loss: {}'.format(lowest))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    if logplot:
        ax.set_yscale('log')
    ax.set_title('Trial '+str(trial)+' Loss')

    if savefig:
        fig.savefig(figdir+'trial{}.png'.format(trial))
    
    return None

#####################################################################
#####################################################################

