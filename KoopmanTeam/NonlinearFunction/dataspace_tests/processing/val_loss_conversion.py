import os
import numpy as np
import pandas as pd
import tensorflow as tf

'''
Hastily thrown together script to 'convert' validation loss from
the compressed space to the uncompressed space.  Since our validation
data is loaded in a very predictable fashion, we know what points are 
used for calculating the validation loss in the compressed space.  We
just load in the model that was trained, have it make predictions on the
validation data (the same data it used before!) and then decompress those 
predictions and compare them with the true evolution.  We then take the 
average L2 loss across all those points.  The true points are never 
compressed, we just grab them straight from the dataset.
'''



##############VARIABLE DECLARATIONS#########################
DATADIR='/fslhome/wilhiteh/datasets/'


TIMESTEP = 0.075

DATADIR='/fslhome/wilhiteh/datasets/'
#TIMERANGE=5  #uncomment for manual selection
TIMERANGE = int(os.getenv('SLURM_ARRAY_TASK_ID'))
MAXEVOLVE=50000
#############################################################



inits = (2000,3000,4000,5000,7500,10000,12500,15000,17500,20000,25000,30000,35000,40000,45000,50000)



val_losses = []
for datasize in inits:

    num_files = MAXEVOLVE
    dataset = '0t{}_{}inits_dt{}_memfit'.format(TIMERANGE,num_files,str(TIMESTEP).replace('.','p'))

    NAME = '0t{}_dt{}_{}inits'.format(TIMERANGE, str(TIMESTEP).replace('.','p'), datasize)

    Phi = tf.keras.models.load_model('/fslhome/wilhiteh/Autoencoder/trial25e1000Phi.h5', compile=False, custom_objects={'Functional':tf.keras.models.Model})
    Phi_inv = tf.keras.models.load_model('/fslhome/wilhiteh/Autoencoder/trial25e1000Phi_inv.h5', compile=False, custom_objects={'Functional':tf.keras.models.Model})
    NonlinearEvolution = tf.keras.models.load_model('/fslhome/wilhiteh/models/{}.h5'.format(NAME), compile=False, custom_objects={'Functional':tf.keras.models.Model})


    start_init = int(0.9*datasize)
    points_per_file = int( np.ceil(TIMERANGE/TIMESTEP) )
    val_data = np.empty([(datasize-start_init)*points_per_file, 4])
    val_predictions = np.empty([(datasize-start_init)*points_per_file,4])

    for evolve in range(start_init, datasize):
        i = evolve-start_init
        val_data[i*points_per_file:(i+1)*points_per_file] = np.genfromtxt(DATADIR+dataset+'/evolution{}.csv'.format(evolve), 
                                                                        skip_header=1, delimiter=',')[:,1:]
       
    for k in range(int( (datasize*points_per_file)/100000 )):
        val_predictions[k*100000:(k+1)*100000] = Phi_inv(NonlinearEvolution(Phi(val_data[k*100000:(k+1)*100000]))).numpy()

    losses = np.linalg.norm(val_predictions[:-1]-val_data[1:], ord=2, axis=-1)

    val_loss = np.average(losses)
    val_losses.append(val_loss)

pd.DataFrame(val_losses).to_csv('./0t{}_dt{}_val_loss_converted'.format(TIMERANGE, str(TIMESTEP).replace('.','p')))
#with open('./{}_val_loss_converted'.format(NAME),'w') as f:
#    f.write(str(val_loss))



