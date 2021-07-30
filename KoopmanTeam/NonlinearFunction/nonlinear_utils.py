import matplotlib.pyplot as plt         
import tensorflow as tf
import numpy as np
import os
import pandas as pd
import qutip as qt


import sys

sys.path.append('..')
from utils import *
sys.path.remove('..')



#######################################################
##################Loss Functions#######################
#######################################################

def L2_loss(y_true, y_pred):
    return tf.norm(y_true-y_pred, ord=2, axis=-1)

def L2_with_unity_loss(y_true, y_pred):
    '''L2 loss augmented with extra loss for not having a norm of 1'''
    return tf.norm(y_true-y_pred, ord=2, axis=-1) + tf.abs(1. - tf.norm(y_pred, ord=2, axis=-1))

########################################################
###################End Loss Functions###################
########################################################


########################################################
###############Data Loading Functions###################
########################################################


#Train on entire evolutions, not just step-to-step?
def get_multiple_evolutions(datadir, num_evolutions, train_evolutions, pre_compress_phi=False):
    '''Read in the evolution of multiple initial conditions at once.  Each initial condition should 
    have its own csv file and be named in the format "evolution{}.csv", where {} indicates an integer 
    from 0 to num_evolutions-1.
    Note that there will be 1 fewer timesteps per evolution file than are present in the evolution file,
    since the last step does not have an evolution.
    
    This is the fastest data loading function (though it has a slow startup), but the dataset must fit into
    memory in order to use it.
    
    PARAMS:
    -------
    str datadir: Path to directory containing evolution files.
    int num_evolutions: The number of initial conditions to read in.
    int train_evolutions: The number of initial conditions that should be treated as 
                          training evolutions
    bool pre_compress_phi: Whether we should use the currently loaded Phi function
                           to compress the dataset.  If the dataset itself has already been
                           run through the encoder, this should be false.
    
    RETURNS:
    --------
    (tf.data.Dataset, tf.data.Dataset): A tuple of the training and validation datasets, respectivley.
                                        These datasets have already been shuffled, no need to shuffle them again.
    '''
    
    pre_evolution = []
    post_evolution = []
    
    #Read training data
    
    #We read all the evolutions up to train evolutions, then convert that array
    ##into a tensorflow dataset.  Trying to make each evolution a dataset and then
    ##appending them all together generally leads to a recursion error
    for i in range(train_evolutions):

        with open(datadir+'evolution{}.csv'.format(i), 'r') as f:
            data = np.genfromtxt(f, delimiter=',', skip_header=1)[:,1:]
            

            if pre_compress_phi:
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
      
    
    #return train, test
    return train.shuffle(1000000, reshuffle_each_iteration=True), test.shuffle(1000000, reshuffle_each_iteration=True)

########################################################
##################End Data Loading######################
########################################################


########################################################
################Prediction Functions####################
########################################################

def predict_single_timestep(state, compression_phi=True, decompression_phi_inv=True, NonlinearF=None):
    
    #If no nonlinear function is passed, use the globally declared one
    if NonlinearF==None:
        NonlinearF = NonlinearEvolution


    if compression_phi:
        assert (len(state) == 4), "State must be a four dimensional vector to be compressed"
        state = Phi(np.array([state,]))
        if decompression_phi_inv:
            return Phi_inv(NonlinearEvolution(state)).numpy()[0]
        else:
            return NonlinearEvolution(state).numpy()[0]
    
    assert (len(state)==3), "State must be three dimensional if not being compressed"
        
    return NonlinearF(np.array([state,])).numpy()[0]


def predict_evolution(initial_cond, save_prediction=False, save_compressed=False, 
                      saved_name='./predicted_evolutions/newtest.csv', 
                      saved_compressed_name='./predicted_evolutions/newtest_compressed.csv', 
                      timesteps=1000, NonlinearF=None):
    '''Use NonlinearF to predict an evolution given an initial condition.  
    The initial condition is compressed using Phi and the state is then evolved for the 
    specified number of timesteps by NonlinearF without any decompression between steps.  
    Each evolved state is then decompressed seperatley.
    
    PARAMS:
    -------
    str initial_cond: Path to file containing initial condition (read as first point in a csv)
    list initial_cond: The initial condition to evolve
    bool save_prediction: Whether or not to save the prediction
    str saved_name: Name/path to file containing predicted evolution (if save_prediction is True)
    bool save_compressed: Whether or not to save the evolution before it is decompressed
    str save_compressed_name: Name/path to file containing compressed evolution prediction
    int timesteps: Number of timesteps to evolve for
    '''
    
    #If no nonlinear function passed, use global one
    if NonlinearF==None:
        NonlinearF = NonlinearEvolution
    
    if type(initial_cond) == str:
        initial_cond = np.genfromtxt(initial_cond, delimiter=',', skip_header=1)[0,1:]
    
    if len(initial_cond)==4:
        initial_cond=Phi(np.array([initial_cond,])).numpy()[0]
    
    learned_evolution=np.empty([timesteps, 3])
    learned_evolution[0] = initial_cond
    
    for i in range(1,timesteps):
        learned_evolution[i] = predict_single_timestep(learned_evolution[i-1], compression_phi=False, decompression_phi_inv=False, NonlinearEvolution=NonlinearEvolution)
    
    if save_compressed:
        pd.DataFrame(learned_evolution).to_csv(saved_compressed_name)
    
    learned_evolution=Phi_inv(learned_evolution).numpy()
    
    if save_prediction:
        pd.DataFrame(learned_evolution).to_csv(saved_name)
        
    return learned_evolution

########################################################
###############End Prediction Functions#################
########################################################


#Create images to be used for animations of the evolution
#VERY SLOW, took about an hour to create 1000 images on my laptop

#After running, use ffmpeg to stitch the frames together (wrapped inside create_animation.sh)

def create_animation_frames(truth, predicted, framedir='./anim/', timesteps=500, view=[-60,30]):
    '''Use QuTip to generate frames for an animation of an evolution on
    the Bloch sphere, with one frame being equal to one timestep.  
    The animation frames can then be stitched together into a video 
    using ffmpeg (actual command is wrapped inside create_animation.sh).
    
    truth and predicted should be of the form np.array([timesteps,4]).  
    Using np.genfromtxt on a pandas generated csv creates the proper structure.
    
    This function is very slow, it can take over an hour to generate 1000 frames.
    
    PARAMS:
    -------
    
    np.array truth, predicted: Matrices holding the true and predicted evolutions 
                               respectivley.
    str framedir: Directory to save created frames to.
    int timesteps: Number of timesteps to create frames for.
                   Must be less than min(len(truth),len(predicted))
    [int, int]: view: Sets the viewing angle for the animation.
    '''

    b = qt.Bloch()
    
    b.view = view

    b.vector_color = ['g', '#000000']
    b.point_color = ['g', '#000000']
    b.point_marker = ['o']


    for i in range(timesteps):
    
        #Clear vectors, keep points
        b.vectors = []

        #Plot true evolution in green

        state = np.array([truth[i,1]+truth[i,2]*1j, truth[i,3]+truth[i,4]*1j])
        state = qt.Qobj(state)
        b.add_states([state])
        b.add_states([state], kind='point')


        #Plot learned evolution in black

        state = np.array([predicted[i,1]+predicted[i,2]*1j, predicted[i,3]+predicted[i,4]*1j])
        state = qt.Qobj(state)
        b.add_states([state])
        b.add_states([state], kind='point')
        
        b.save(dirc=framedir)


def predict_and_load(initial_condition_file, timerange, timestep, inits, save_prediction=True, 
                     save_compressed=True, timesteps=1000, modeldir=None, 
                     prediction_dir='./predicted_evolutions/', subtitle=''):
    '''Creates/loads prediction for given parameters and loads the corresponding
    model.  Models should be named 0t(timerange)_dt(timestep)_(inits)inits(subtitle).h5 .
    
    PARAMS:
    -------
    str initial_condition_file: Path to file containing initial condition and true evolution data
    int timerange, timestep, inits: Parameters selecting our model based on the naming convention
    bool save_prediction/compressed: Whether we should save the prediction/compressed prediction file.
                                     Follows same naming convention used for model.
    int timesteps: Selects how many timesteps the evolution should go through
    
    RETURNS:
    --------
    NonlinearEvolution, true_evolution, predicted_evolution
    '''
    
    
    if modeldir==None:
        modeldir = './dataspace_tests/models/'
    
    NonlinearEvolution = tf.keras.models.load_model(modeldir+'0t{}_dt{}_{}inits{}.h5'.format(timerange,str(timestep).replace('.','p'),inits,subtitle), compile=False)

    
    if os.path.exists(prediction_dir+'0t{}_dt{}_{}inits{}.csv'.format(timerange,str(timestep).replace('.','p'),inits,subtitle)):
        data = np.genfromtxt(initial_condition_file, delimiter=",", skip_header=1)
        learned = np.genfromtxt(prediction_dir+'0t{}_dt{}_{}inits{}.csv'.format(timerange,str(timestep).replace('.','p'),inits,subtitle), delimiter=',', skip_header=1)
        return NonlinearEvolution, data, learned
    else:
        if save_prediction==True:
            saved_name = prediction_dir+'0t{}_dt{}_{}inits{}.csv'.format(timerange,str(timestep).replace('.','p'),inits,subtitle)
        if save_compressed==True:
            saved_compressed_name = prediction_dir+'0t{}_dt{}_{}inits{}_compressed.csv'.format(timerange,str(timestep).replace('.','p'),inits,subtitle)
        predict_evolution(initial_condition_file, save_prediction=save_prediction, saved_name=saved_name, save_compressed=save_compressed, saved_compressed_name=saved_compressed_name, timesteps=timesteps, NonlinearF=NonlinearEvolution)
        data = np.genfromtxt(initial_condition_file, delimiter=",", skip_header=1)
        learned = np.genfromtxt(saved_name, delimiter=',', skip_header=1)
        return NonlinearEvolution, data, learned

    
#TODO - change all the boolean switches into a dictionary for selection
def plot_dynamics(truth, predicted, savefig=False, timerange=500, timestart=0,
                  phispace=False, truespace=False, expectation=False, figname='dynamics.png', 
                  fontsize=16):
    '''Plot the dynamics (or state) of some evolution.  
    
    PARAMS:
    -------
    np.array truth,predicted: The true evolution and the predicted evolution.  Should be read from
                              a csv generated by pd.DataFrame with np.genfromtxt, or have an otherwise
                              identical structure to such an array
    bool phispace, truespace, expectation: Selects which space to plot.  Truespace refers to the 
                                           uncompressed 4 dimensional space which the Bloch sphere lives in, 
                                           phispace refers to the space in which NonlinearF learns its dynamics.
                                           Expectation refers to the space which QuTip uses to plot points on the 
                                           Bloch sphere.  If all of these are set to False, default to plotting the 
                                           'ideal' compressed space (r1, r2, relative phase).
    '''
    
    FONTSIZE=fontsize
    plt.rc('font', size=FONTSIZE)
    plt.rc('axes', titlesize=FONTSIZE,labelsize=FONTSIZE)
    plt.rc('legend', fontsize=FONTSIZE)
    plt.rc('xtick', labelsize=FONTSIZE)
    plt.rc('ytick', labelsize=FONTSIZE)
    plt.rc('figure', titlesize=FONTSIZE)
    
    
    
    if truespace:
        fig, ax = plt.subplots(4,1,figsize=(18,32))
        
        ax[0].plot(truth[timestart:timerange,0], truth[timestart:timerange,1], label='True')
        ax[0].plot(predicted[timestart:timerange,0], predicted[timestart:timerange,1], ls='--', label='Predicted')
        
        ax[1].plot(truth[timestart:timerange,0], truth[timestart:timerange,2])
        ax[1].plot(predicted[timestart:timerange,0], predicted[timestart:timerange,2], ls='--')
        
        ax[2].plot(truth[timestart:timerange,0], truth[timestart:timerange,3])
        ax[2].plot(predicted[timestart:timerange,0], predicted[timestart:timerange,3], ls='--')
        
        ax[3].plot(truth[timestart:timerange,0], truth[timestart:timerange,4])
        ax[3].plot(predicted[timestart:timerange,0], predicted[timestart:timerange,4], ls='--')
        
        ax[0].legend()
        ax[0].set_title(r'$\alpha$')
        ax[1].set_title(r'$\beta$')
        ax[2].set_title(r'$\gamma$')
        ax[3].set_title(r'$\delta$')
        ax[3].set_xlabel('Timestep')
        
    elif expectation:
        fig, ax = plt.subplots(3,1, figsize=(18,24))

        xs = []
        ys = []
        zs = []

        xspred = []
        yspred = []
        zspred = []

        for i in range(timestart, timerange):
            state = np.array([truth[i,1]+truth[i,2]*1j, truth[i,3]+truth[i,4]*1j])
            state = qt.Qobj(state)

            predstate = np.array([predicted[i,1] + predicted[i,2]*1j, predicted[i,3] + predicted[i,4]*1j])
            predstate = qt.Qobj(predstate)

            for t in [state]:
                xs.append(qt.expect(qt.sigmax(), t))
                ys.append(qt.expect(qt.sigmay(), t))
                zs.append(qt.expect(qt.sigmaz(), t))


            for t in [predstate]:
                xspred.append(qt.expect(qt.sigmax(), t))
                yspred.append(qt.expect(qt.sigmay(), t))
                zspred.append(qt.expect(qt.sigmaz(), t))


        ax[0].plot(np.arange(timestart,timerange), xs)
        ax[1].plot(np.arange(timestart,timerange), ys)
        ax[2].plot(np.arange(timestart,timerange), zs)

        ax[0].plot(np.arange(timestart,timerange), xspred)
        ax[1].plot(np.arange(timestart,timerange), yspred)
        ax[2].plot(np.arange(timestart,timerange), zspred)

        ax[0].set_title(r'$\langle\psi|\hat{\sigma_x}|\psi\rangle$')
        ax[1].set_title(r'$\langle\psi|\hat{\sigma_y}|\psi\rangle$')
        ax[2].set_title(r'$\langle\psi|\hat{\sigma_z}|\psi\rangle$')
    
    elif not phispace:
        fig, ax = plt.subplots(3,1,figsize=(18,24))
        
        
        r1 = np.sqrt(truth[timestart:timerange,1]**2 + truth[timestart:timerange,2]**2)
        r2 = np.sqrt(truth[timestart:timerange,3]**2 + truth[timestart:timerange,4]**2)
        theta = np.arctan2(truth[timestart:timerange,4],truth[timestart:timerange,3]) - np.arctan2(truth[timestart:timerange,2],truth[timestart:timerange,1])

        r1l = np.sqrt(predicted[timestart:timerange,1]**2 + predicted[timestart:timerange,2]**2)
        r2l = np.sqrt(predicted[timestart:timerange,3]**2 + predicted[timestart:timerange,4]**2)
        thetal = np.arctan2(predicted[timestart:timerange,4],predicted[timestart:timerange,3]) - np.arctan2(predicted[timestart:timerange,2],predicted[timestart:timerange,1])
        
        ax[0].plot(truth[timestart:timerange,0], r1, label='True')
        ax[0].plot(predicted[timestart:timerange,0], r1l, ls='--', label='Predicted')

        ax[1].plot(truth[timestart:timerange,0],r2)
        ax[1].plot(predicted[timestart:timerange,0],r2l, ls='--')

        ax[2].plot(truth[timestart:timerange,0],theta)
        ax[2].plot(predicted[timestart:timerange,0],thetal, ls='--')

        ax[0].legend()
        ax[0].set_title(r'$r_1 = \sqrt{\alpha^2+\beta^2}$')
        ax[1].set_title(r'$r_2 = \sqrt{\gamma^2+\delta^2}$')
        ax[2].set_title(r'$\theta = \arctan\left(\frac{\delta}{\gamma}\right) - \arctan\left(\frac{\beta}{\alpha}\right)$')

        ax[2].set_xlabel('Timestep')
        
        
    else:
        fig, ax = plt.subplots(3,1,figsize=(18,24))
        
        
        compressed_truth = Phi(truth[:,1:]).numpy()
        compressed_pred = Phi(predicted[:,1:]).numpy()

        ax[0].plot(truth[timestart:timerange,0], compressed_truth[timestart:timerange,0], label='True')
        ax[0].plot(predicted[timestart:timerange,0], compressed_pred[timestart:timerange,0], ls='--',label='Predicted')

        ax[1].plot(truth[timestart:timerange,0],compressed_truth[timestart:timerange,1])
        ax[1].plot(predicted[timestart:timerange,0],compressed_pred[timestart:timerange,1], ls='--')

        ax[2].plot(truth[timestart:timerange,0],compressed_truth[timestart:timerange,2])
        ax[2].plot(predicted[timestart:timerange,0],compressed_pred[timestart:timerange,2], ls='--')

        ax[0].legend()
        ax[0].set_title(r'$\phi|\alpha\rangle[0]$')
        ax[1].set_title(r'$\phi|\alpha\rangle[1]$')
        ax[2].set_title(r'$\phi|\alpha\rangle[2]$')
        ax[2].set_xlabel('Timestep')
        
    if savefig:
        fig.savefig(figname, bbox_inches='tight', transparent=False)

def dataset_integrity_check(datadir, timerange, timestep):
    timestep = str(timestep).replace('.','p')
    datafiles = datadir+'dt{}_tests/0t{}/datafiles/'.format(timestep, timerange)
    for file in os.listdir(datafiles):
        with open(datafiles+file, 'r') as f:
            if (f.readlines()[1].strip().split(',')[-1]) == '':
                print('Check '+file)


#This metric essentially means that our prediction will deviate from
##the true value by normalized_val_loss per unit of time
def normalized_val_loss(datafile, timestep):
    val_losses = []
    with open(datafile, 'r') as f:
        for line in f.readlines():
            if line.strip().split(',')[0] == 'val_loss':
                val_losses = [float(x) for x in line.strip().split(',')[1:]]
    
    return min(val_losses)/timestep


def dataset_spatial_density(dataset, num_evolutions, ham, pretty_print=True):
    '''(Tries to) measure the 'spatial desnity' of points in our dataset.
    Assumes that every evolution is a complete circle, starts from the procession axis, 
    then checks how far we must walk along the Bloch sphere before reaching another evolution
    track.  Essentially just projects every evolution onto the procession axis, projects them
    back onto the same arc on the Bloch sphere (to line the points up), and takes some measurments 
    there.
    
    Not convinced that it actually works though.
    '''
    
    #if type(ham) == str:
    #    ham = np.genfromtxt(ham, delimiter=',', skip_header=1)[:,1:]
    
    
    #Gives axis of rotation on Bloch sphere
    ##Derived from eq. 16.104 in Merzbacher
    procession_z = 0.5*(ham[0,0]-ham[1,1])
    procession_y = -0.5j*(ham[1,0]-ham[0,1])
    procession_x = 0.5*(ham[1,0]+ham[0,1])
    
    procession_axis = np.real(np.array([procession_x, procession_y, procession_z]))
    procession_axis = 1/np.linalg.norm(procession_axis) * procession_axis
    
    
    #Read in 1 point from every evolution as a quantum state
    points = np.empty([num_evolutions,4])
    for i in range(num_evolutions):
        points[i] = np.genfromtxt(dataset+'evolution{}.csv'.format(i), skip_header=1, delimiter=',')[0,1:]
    
    #Get the spherical representation of the points
    ##Theta is relative phase, phi is 'angle between' the probabilities
    r1s = np.sqrt(points[:,0]**2 + points[:,1]**2)
    r2s = np.sqrt(points[:,2]**2 + points[:,3]**2)
    phis = np.arctan2(r2s,r1s)
    thetas = np.arctan2(points[:,3],points[:,2]) - np.arctan2(points[:,1],points[:,0])
    
    #Project spherical coords onto cartesian axes
    states_x = np.cos(thetas)*np.sin(phis)
    states_y = np.sin(thetas)*np.sin(phis)
    states_z = np.cos(phis)
    
 
    states_xyz = np.column_stack((states_x, states_y, states_z))

    
    #Project each state onto the axis of rotation
    ##Just getting the scalar projection, don't really care about the 
    ##specific direction of the projected vector
    projections = np.inner(procession_axis, states_xyz)
    
    
    
    #Projections are on (flat) axis of procession, but the evolutions exist on a sphere
    ##The flat projection will artificially cause points to 'bunch up' near the ends
    ##giving the appearance of high density even if the points are evenly spaced on the 
    ##sphere.  So we convert the points on this projected axis into angles of the corresponding
    ##position on the arc
    arc_angles = np.arccos(projections)
    
    arc_angles = np.sort(arc_angles)
    
    
    darc_angles = np.abs(arc_angles[1:] - arc_angles[:-1])
    
    
    if pretty_print:
        head = '{} evolutions from dataset {}:'.format(num_evolutions,dataset)
        print(head)
        print(f'{"":-<{len(head)}}')
        print('Average arc distance: {}'.format(np.mean(darc_angles)))
        print('Standard deviation: {}'.format(np.std(darc_angles)))
        print('Minimum arc distance: {}'.format(np.min(darc_angles)))
        print('Maximum arc distance: {}'.format(np.max(darc_angles)))
        print('Smallest angle: {}'.format(arc_angles[0]))
        print('Largest angle: {}'.format(arc_angles[-1]))
    
    return np.mean(darc_angles), np.std(darc_angles)

def dataset_temporal_density(dataset, num_points, ham, pretty_print=True):
    '''(Attempt to) measure the 'temporal density' of evolutions in the dataset.
    Essentially just chooses the first evolution in the dataset, walks around that
    evolution track, and checks how far apart each point is.  Since we get shifted
    slightly with each complete rotation (due to not sampling at an integer multiple 
    of the period), we can't just compare adjacent points in the dataset, we first 
    're-align' them.
    
    Not convinced it works yet though.
    '''
    
    #Gives axis of rotation on Bloch sphere
    ##Derived from eq. 16.104 in Merzbacher
    procession_z = 0.5*(ham[0,0]-ham[1,1])
    procession_y = -0.5j*(ham[1,0]-ham[0,1])
    procession_x = 0.5*(ham[1,0]+ham[0,1])
    
    procession_axis = np.real(np.array([procession_x, procession_y, procession_z]))
    procession_axis = 1/np.linalg.norm(procession_axis) * procession_axis
    
    #Read in 1 point from every evolution as a quantum state
    states = []
    points = np.genfromtxt(dataset+'evolution0.csv', skip_header=1, delimiter=',')[0:num_points,1:]
    
    #Get the spherical representation of the points
    ##Theta is relative phase, phi is 'angle between' the probabilities
    r1s = np.sqrt(points[:,0]**2 + points[:,1]**2)
    r2s = np.sqrt(points[:,2]**2 + points[:,3]**2)
    phis = np.arctan2(r2s,r1s)
    thetas = np.arctan2(points[:,3],points[:,2]) - np.arctan2(points[:,1],points[:,0])
    
    #Project spherical coords onto cartesian axes
    states_x = np.cos(thetas)*np.sin(phis)
    states_y = np.sin(thetas)*np.sin(phis)
    states_z = np.cos(phis)

    states_xyz = np.column_stack((states_x, states_y, states_z))


    #Project our states into the plane defined with the procession axis as its normal
    ##This is just v - proj(v,w) * w with some extra numpy magic sprinkled in to make it work
    perpstates = states_xyz - np.inner(procession_axis, states_xyz).reshape(num_points,1) * np.tile(procession_axis, (num_points,1))

    
    #Define one of our vectors to be the 'zero angle'
    refrence_vec = 1/np.linalg.norm(perpstates[0]) * perpstates[0]
    
    thetas = np.arccos(np.inner(refrence_vec, perpstates)/np.linalg.norm(perpstates))
    
    thetas = np.sort(thetas)
    
    dthetas = np.abs(thetas[1:] - thetas[:-1])
    
    if pretty_print:
        head = '{} evolutions from dataset {}:'.format(num_points,dataset)
        print(head)
        print(f'{"":-<{len(head)}}')
        print('Average arc distance: {}'.format(np.mean(dthetas)))
        print('Standard deviation: {}'.format(np.std(dthetas)))
        print('Minimum arc distance: {}'.format(np.min(dthetas)))
        print('Maximum arc distance: {}'.format(np.max(dthetas)))
        
    
    return 




#Still quite slow, will take about 42 hours for the full 25000000 points

def get_sum_pair_triples(dataset, num_inits, start_init=0,epsilon=1e-2):
    valid_points = []
    for i in range(start_init,num_inits+start_init):
        k=0
        evolution = np.genfromtxt(dataset+'evolution{}.csv'.format(i), delimiter=',', skip_header=1)[:,1:]
        for v in evolution:
            vpws = evolution + v
            for vpw in vpws[k:]: 
                diff = np.linalg.norm(evolution-vpw, axis=-1)
                if (len(diff[diff<epsilon]) > 0):
                    w = vpw-v
                    vpwtilde = evolution[diff<epsilon][0]
                    valid_points.append([v[0], v[1], v[2], w[0], w[1], w[2], vpwtilde[0], vpwtilde[1], vpwtilde[2], i])
            k+=1
        if(i%200 == 0):
            print('Finished evolution {} of {}'.format(i-start_init,num_inits))
    return np.array(valid_points)

def get_sum_pair_triples_bloch(dataset, num_inits, max_points_per_init=400, start_init=0,epsilon=1e-2):
    valid_points = []
    for i in range(start_init, num_inits+start_init):
        k=0
        evolution = np.genfromtxt(dataset+'evolution{}.csv'.format(i), delimiter=',', skip_header=1)[:,1:]
        for v in evolution:
            vpws = evolution + v
            norms = np.linalg.norm(vpws,axis=-1).reshape([len(vpws),1])
            vpws = 1/norms * vpws
            for vpw in vpws[k:]:
                #Disallow v+0
                if np.linalg.norm(vpw-v) < epsilon:
                    continue
                diff = np.linalg.norm(evolution-vpw, axis=-1)
                if (len(diff[diff<epsilon]) > 0):
                    w = vpw*norms[diff<epsilon][0] - v
                    print(w)
                    vpwtilde = evolution[diff<epsilon][0]
                    valid_points.append([v[0], v[1], v[2], v[3], w[0], w[1], w[2], w[3], vpwtilde[0], vpwtilde[1], vpwtilde[2], vpwtilde[3], i])
            k+=1
        if i%200 == 0:
            print('Finished evolution {} of {}'.format(i-start_init,num_inits))
    return np.array(valid_points)
                    

def read_sum_pair_triples(file,max_evolve,num_triples=-1,compressed=True):
    '''Reads in either num_triples (if specified) or all the triples
    from up to (and including) max_evolve, whichever is less.  If max_evolve
    is higher than the maximum evolution file, then we just return everything in the file
    (or num_triples, if specified)
    '''
    
    if compressed:
        vs = np.genfromtxt(file, delimiter=',', skip_header=1, usecols=(1,2,3))
        ws = np.genfromtxt(file, delimiter=',', skip_header=1, usecols=(4,5,6))
        vpws = np.genfromtxt(file, delimiter=',', skip_header=1, usecols=(7,8,9))
        evolutions = np.genfromtxt(file,delimiter=',', skip_header=1, usecols=(10))
    else:
        vs = np.genfromtxt(file, delimiter=',', skip_header=1, usecols=(1,2,3,4))
        ws = np.genfromtxt(file, delimiter=',', skip_header=1, usecols=(5,6,7,8))
        vpws = np.genfromtxt(file, delimiter=',', skip_header=1, usecols=(9,10,11,12))
        evolutions = np.genfromtxt(file,delimiter=',', skip_header=1, usecols=(13))    
    
    max_triple = np.searchsorted(evolutions, max_evolve+1)
    if max_triple < num_triples or num_triples==-1:
        num_triples = max_triple
    
    return vs[:num_triples], ws[:num_triples], vpws[:num_triples]


def linear_recovery_metric(file, num_evolutions, average=False, timerange=-1, timestep=-1, num_inits=-1):
    
    
    if (timerange, timestep, num_inits) != (-1,-1,-1):
        NonlinearEvolution = predict_and_load(None, timerange, timestep, num_inits, load_prediction=False)
    

    
    ayys, bs, apbs = read_sum_pair_triples(file, num_evolutions, compressed=False)
  #  for k in range(len(vs)):
  #      a = np.array([vs[k],])
  #      b = np.array([ws[k],])
  #      apb = np.array([vpws[k],])

    
    alpha = 1/np.linalg.norm(ayys+bs, axis=-1)
    alpha = alpha.reshape(len(alpha),1)
    
    
    apbs_compressed = Phi(apbs).numpy()
    Fapb = NonlinearEvolution(apbs_compressed).numpy()
    apb_evolved = Phi_inv(Fapb).numpy()
    
    ayys_compressed = Phi(ayys).numpy()
    Fa = NonlinearEvolution(ayys_compressed).numpy()
    a_evolved = Phi_inv(Fa).numpy()
    
    bs_compressed = Phi(bs).numpy()
    Fb = NonlinearEvolution(bs_compressed).numpy()
    b_evolved = Phi_inv(Fb).numpy()
    
    linear_diff = np.linalg.norm(apb_evolved - alpha*(a_evolved+b_evolved), axis=-1)
        
    if average==True:
        avg = np.average(linear_diff)
        #variance = np.sqrt(np.sum((samples-avg)**2) / (num_test_points-1))
        return avg
    else:
        return np.max(linear_diff)
 

