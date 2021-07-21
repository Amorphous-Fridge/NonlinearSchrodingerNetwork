import os
import numpy as np
import pandas as pd
import tensorflow as tf



##############VARIABLE DECLARATIONS#########################
DATADIR='/fslhome/wilhiteh/datasets/'
CAPBYEVOLVE = True


TIMERANGE = int(os.getenv('SLURM_ARRAY_TASK_ID'))

DATADIR='/fslhome/wilhiteh/datasets/'
#TIMERANGE=5  #uncomment for manual selection
TIMESTEP=0.05
MAXPOINTS=5000000
MAXEVOLVE=50000

#Have a generic compressing and triple checking job
##pre-written, set this flags accordingly if you want to use them
COMPRESSING=False
TRIPLE_CHECKING=True
if TRIPLE_CHECKING:
    STARTING_EVOLUTION=0
    EVOLUTIONS_TO_PROCESS=10000
###########################################################


################UTILITY DECLARATIONS#######################

points_per_file = int(np.ceil(TIMERANGE/TIMESTEP)) 
if CAPBYEVOLVE:
    num_files = MAXEVOLVE
else:
    num_files = int(np.floor(MAXPOINTS/points_per_file))
dataset = '0t{}_{}inits_dt{}_memfit'.format(TIMERANGE,num_files,str(TIMESTEP).replace('.','p'))


Phi = tf.keras.models.load_model('/fslhome/wilhiteh/Autoencoder/trial25e1000Phi.h5', compile=False, custom_objects={'Functional':tf.keras.models.Model})
Phi_inv = tf.keras.models.load_model('/fslhome/wilhiteh/Autoencoder/trial25e1000Phi_inv.h5', compile=False, custom_objects={'Functional':tf.keras.models.Model})


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
                    valid_points.append([v[0], v[1], v[2], w[0], w[1], w[2], vpwtilde[0], vpwtilde[1], vpwtilde[2]])
        k+=1
    if i%500==0:
        print('Finished the first {} evolutions'.format(i))
    pd.DataFrame(valid_points).to_csv(save_file)
    return valid_points

##########################################################


###################PROCESSING COMMANDS#########################

if COMPRESSING:
    if not(os.path.isdir(DATADIR+dataset+'_compressed/')):
        os.mkdir(DATADIR+dataset+'_compressed/')

    for i in range(num_files):
        compress_evolution(DATADIR+dataset+'/evolution{}.csv'.format(i), DATADIR+dataset+'_compressed/evolution{}.csv'.format(i))
        if i%200==0:
            print('Finished {} of {} evolutions'.format(i,num_files))

elif TRIPLE_CHECKING:
    if not(os.path.isdir(DATADIR+dataset+'_compressed/')):
        print("Error: dataset {} does not appear to have a compressed version in {}".format(dataset,DATADIR))
        #We didn't import sys so this will lead to a crash,
        ##but we're trying to exit anyways so eh
        sys.exit()

    if not(os.path.isdir(DATADIR+dataset+'_sum_pair_triples/')):
        os.mkdir(DATADIR+dataset+'_sum_pair_triples/')
    
    get_sum_pair_triples(DATADIR+dataset+'_compressed/', DATADIR+dataset+'_sum_pair_triples/e{}t{}.csv'.format(STARTING_EVOLUTION,STARTING_EVOLUTION+EVOLUTIONS_TO_PROCESS),
                         EVOLUTIONS_TO_PROCESS, start_init=STARTING_EVOLUTION)

##############################################################
