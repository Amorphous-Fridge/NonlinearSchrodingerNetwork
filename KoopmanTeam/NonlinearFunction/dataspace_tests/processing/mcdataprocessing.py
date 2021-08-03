import os
import numpy as np
import pandas as pd
import tensorflow as tf

'''
This file contains all of the functions for processing the dataset, both
before and after training.

The four processing options are:

COMPRESSING: Pre-compresses the dataset with the loaded Phi model.  This is 
             a necessary pre-processing step for the data loading functions in
             searches.py to work.  
             Creates a new dataset in DATADIR which has the same name as the original
             dataset, but with '_compressed/' appended.

CONVERTING: Converts a dataset from csv files to faster tfrecords files.
            Will eventually be required pre-processing for datasets that do not fit 
            into memory, but the loading functions for those datasets are broken right 
            now anyways.  Note that once these loading functions are up and running, the
            dataset will need to be compressed BEFORE being converted to tfrecords.
            Creates a new dataset in DATADIR which has the same name as the original 
            dataset, but with 'TFRecords/' appended.

TRIPLE_CHECKING: This seaches the dataset for points v and w such that a(v+w) is also
                 approximaltey in the dataset.  a is the norm of v+w.  This yields points
                 that we can test on our function to check how well it recovers the linearity
                 of the Schroedinger equation.
                 Creates a new 'dataset' in DATADIR (though its just a single csv file) which 
                 shares the same name as the original dataset with '_sum_pair_triples_bloch' appended.

TRIPLE_CHECKING_COMPRESSED: Similar to TRIPLE_CHECKING, but operates on a compressed dataset.
                            Used for checking nonlinearity of the evolution function, rather than
                            linearity of the entire end-to-end system.
                            Creates a new dataset in DATADIR which shares the same name as the original
                            dataset with '_sum_pair_triples' appended

Functions for reading the files created by TRIPLE_CHECKING(_COMPRESSED) are also in this file.

Which processing option we use is jankily selected with a bunch of booleans.


The parameters TIMESTEP, MAXEVOLVE, and TIMERANGE are used to specify which dataset
we wish to process, following the usual naming convention.
'''



##############VARIABLE DECLARATIONS#########################
DATADIR='/fslhome/wilhiteh/datasets/'
DATADIR='/fslhome/mccutler/Koopman-Quantum/bkp/datasets_mc/'


TIMERANGE = int(os.getenv('SLURM_ARRAY_TASK_ID'))

#TIMERANGE=5  #uncomment for manual selection
TIMESTEP=0.1
MAXEVOLVE=50000

#Have a generic compressing and triple checking job
##pre-written, set this flags accordingly if you want to use them
COMPRESSING=False
TRIPLE_CHECKING=False
TRIPLE_CHECKING_COMPRESSED=True
CONVERTING=False

#Checking through all the evolutions takes a long time
##and we get quite a few points with just the first several thousant,
##so here we can cut off the processing early (or resume it with the 
##STARTING_EVOLUTION param)
if TRIPLE_CHECKING_COMPRESSED or TRIPLE_CHECKING:
    STARTING_EVOLUTION=0
    EVOLUTIONS_TO_PROCESS=5000
###########################################################


################UTILITY DECLARATIONS#######################

points_per_file = int(np.ceil(TIMERANGE/TIMESTEP)) 
num_files = MAXEVOLVE
dataset = '0t{}_{}inits_dt{}'.format(TIMERANGE,num_files,str(TIMESTEP).replace('.','p'))


Phi = tf.keras.models.load_model('/fslhome/mccutler/Koopman-Quantum/shared-git/NonlinearSchrodingerNetwork/KoopmanTeam/Autoencoder/Autoencoder_Trials/models/trial25e1000Phi.h5', compile=False, custom_objects={'Functional':tf.keras.models.Model})
Phi_inv = tf.keras.models.load_model('/fslhome/mccutler/Koopman-Quantum/shared-git/NonlinearSchrodingerNetwork/KoopmanTeam/Autoencoder/Autoencoder_Trials/models/trial25e1000Phi_inv.h5', compile=False, custom_objects={'Functional':tf.keras.models.Model})


def compress_evolution(evolution_path, save_path):
    evolution = np.genfromtxt(evolution_path, delimiter=',', skip_header=1)[:,1:]
    compressed_evolution = Phi(evolution).numpy()
    pd.DataFrame(compressed_evolution).to_csv(save_path)


def get_sum_pair_triples(dataset, save_file,num_inits, start_init=0, epsilon=1e-2):
    valid_points = []

    for i in range(start_init, start_init+num_inits):
        k=0
        evolution=np.genfromtxt(dataset+'evolution{}.csv'.format(i), delimiter=',', skip_header=1)[:,1:]
        for v in evolution:
            vpws = evolution+v
            for vpw in vpws[k:]:
                diff = np.linalg.norm(evolution-vpw,axis=-1)
                if (len(diff[diff<epsilon]) > 0):
                    w = vpw-v
                    vpwtilde = evolution[diff<epsilon][0]
                    valid_points.append([v[0], v[1], v[2], w[0], w[1], w[2], vpwtilde[0], vpwtilde[1], vpwtilde[2], i])
                        
            k+=1

        if i%500==0:
            print('Finished the first {} evolutions'.format(i))

    pd.DataFrame(valid_points).to_csv(save_file)
    return valid_points

def get_sum_pair_triples_bloch(dataset, save_file, num_inits, max_points_per_evolve=400, start_init=0, epsilon=1e-2):
    valid_points = []
    for i in range(start_init, num_inits+start_init):
        k=0
        evolution = np.genfromtxt(dataset+'evolution{}.csv'.format(i), delimiter=',', skip_header=1)[:,1:]
        for v in evolution:
            vpws = evolution + v
            norms = np.linalg.norm(vpws, axis=-1).reshape([len(vpws),1])
            vpws = 1/norms * vpws
            for vpw in vpws[k:]:
                if np.linalg.norm(vpw-v) < epsilon:
                    continue
                diff = np.linalg.norm(evolution-vpw, axis=-1)
                if(len(diff[diff<epsilon]) > 0):
                    w = vpw*norms[diff<epsilon][0] - v
                    vpwtilde = evolution[diff<epsilon][0]
                    valid_points.append([v[0], v[1], v[2], v[3], w[0], w[1], w[2], w[3], vpwtilde[0], vpwtilde[1], vpwtilde[2], vpwtilde[3], i])
                    break #Only allow each v to be used once
            k+=1
            if len(valid_points) >= i*max_points_per_evolve:
                break
        if i%500==0:
            print('Finished the first {} evolutions'.format(i))
    pd.DataFrame(valid_points).to_csv(save_file)
    return valid_points


def TFRecord_convert(datadir, processed_dir):
    def _floats_feature(ex):
        return tf.train.Feature(float_list=tf.train.FloatList(value=ex))

    for file in [x for x in os.listdir(datadir) if x.endswith('.csv')]:
        data = np.genfromtxt(datadir+file, delimiter=',', skip_header=1)[:,1:]
        with tf.io.TFRecordWriter(processed_dir+file.replace('.csv','.tfrecord')) as writer:
            pre_evolve=data[:-1]
            post_evolve=data[1:]
            
            for i in range(len(data)-1):
                sample = tf.train.Example(features=tf.train.Features(feature={
                    'pre_evolve': _floats_feature(pre_evolve[i]),
                    'post_evolve': _floats_feature(post_evolve[i])
                }))
                writer.write(sample.SerializeToString())

    return

##########################################################


###################PROCESSING COMMANDS#########################

if COMPRESSING:
    if not(os.path.isdir(DATADIR+dataset+'_compressed/')):
        os.mkdir(DATADIR+dataset+'_compressed/')

    for i in range(num_files):
        compress_evolution(DATADIR+dataset+'/evolution{}.csv'.format(i), DATADIR+dataset+'_compressed/evolution{}.csv'.format(i))
        if i%200==0:
            print('Finished {} of {} evolutions'.format(i,num_files))

if CONVERTING:
    if not(os.path.isdir(DATADIR+dataset+'_compressed_TFRecord')):
        os.mkdir(DATADIR+dataset+'_compressed_TFRecord')

    TFRecord_convert(DATADIR+dataset+'_compressed/', DATADIR+dataset+'_compressed_TFRecord/')

elif TRIPLE_CHECKING_COMPRESSED:
    if not(os.path.isdir(DATADIR+dataset+'_compressed/')):
        print("Error: dataset {} does not appear to have a compressed version in {}".format(dataset,DATADIR))
        #We didn't import sys so this will lead to a crash,
        ##but we're trying to exit anyways so eh
        sys.exit()

    if not(os.path.isdir(DATADIR+dataset+'_sum_pair_triples/')):
        os.mkdir(DATADIR+dataset+'_sum_pair_triples/')
    
    get_sum_pair_triples(DATADIR+dataset+'_compressed/', DATADIR+dataset+'_sum_pair_triples/e{}t{}.csv'.format(STARTING_EVOLUTION,STARTING_EVOLUTION+EVOLUTIONS_TO_PROCESS),
                         EVOLUTIONS_TO_PROCESS, start_init=STARTING_EVOLUTION)

elif TRIPLE_CHECKING:
    if not(os.path.isdir(DATADIR+dataset+'/')):
        print("Error: dataset {} does not appear to exist in {}".format(dataset,DATADIR))
        #We didn't import sys so this will lead to a crash,
        ##but we're trying to exit anyways so eh
        sys.exit()

    if not(os.path.isdir(DATADIR+dataset+'_sum_pair_triples/')):
        os.mkdir(DATADIR+dataset+'_sum_pair_triples/')
    
    get_sum_pair_triples_bloch(DATADIR+dataset+'/', DATADIR+dataset+'_sum_pair_triples/e{}t{}_Bloch.csv'.format(STARTING_EVOLUTION,STARTING_EVOLUTION+EVOLUTIONS_TO_PROCESS),
                         EVOLUTIONS_TO_PROCESS, start_init=STARTING_EVOLUTION)


##############################################################
