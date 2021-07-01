from contextlib import redirect_stdout  #Used for writing model architecture to datafiles
import matplotlib.pyplot as plt         
from datetime import date               #Used for datafiles

import numpy as np
import os

def write_history(history, model, loss = 'autoencoding_loss', 
                  optimizer='Adam', lr='.001', 
                  batch_size='1024', datadir='./datafiles/',
                  other_info={}):
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
    dict other_info - dictionary containing any additional information to be added
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
        for k in other_info.keys():
            f.write("{},{}".format(k, other_info[k]))
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

def append_history(history, trial=None, datadir='./datafiles/', params_update = True, 
                   params = {'Loss':None, 'Optimizer':None, 'Learning Rate':None, 'Batch Size':None}):
    '''Appends new training data to trial datafile.
    This will only work with datafiles written using write_history (or files of identical form)
    
    PARAMS:
    -------
    history - The history callback containing the new run's data
    int trial - The trial number to append the data to.  If we want to append data to 
            trial43.data, this would be 43.  If none specified, updates the most recent trial
    str datadir - The directory containing the datafile
    bool params_update - Boolean indicating if we should update parameters (loss used, optimizer, etc.)
                    in addition to adding the new loss data
    dict params - Dictionary containing updated parameter values
                  If parameter is not included, the previous value 
                  written for that parameter will be repeated
    '''
    
    if trial==None:
        trial = max([int(x.strip('.data').strip('trial')) for x in os.listdir(datadir)])
    
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
    
    
def loss_plot(trial, datadir='./datafiles/', savefig = True,
              figdir = './figures/', logplot=False,
              metric = None, mark_runs = False, 
              skip_epochs=0, mark_lowest = True, 
              validation=True, size = (12,12)):
    '''Creates a plot of the loss/metric for the given trial number
    '''
    
    if metric == None:
        metric = 'loss'

    losses = []
    vals = []
    runs = []
    
    #Read in the data
    with open(datadir+'trial'+str(trial)+'.data', 'r') as f:
        for line in f.readlines():
            if line.split(',')[0] == metric:
                losses = [float(x) for x in line.strip().split(',')[1:]]
                if (not mark_runs) and (not validation):
                    break
            elif line.split(',')[0] == 'val_'+metric:
                vals = [float(x) for x in line.strip().split(',')[1:]]
                if not mark_runs:
                    break
            elif (line.split(',')[0] == 'Epochs' and mark_runs == True):
                runs = ['0.']+line.strip().split(',')[1:]
                runs = [float(runs[i-1])+float(runs[i-2]) for i in range(2, len(runs))]
                break
            
    
    fig, ax = plt.subplots(1,1, figsize = size)


    ax.plot(range(len(losses[skip_epochs:])), losses[skip_epochs:], label=metric)
    
    if validation:
        ax.plot(range(len(vals[skip_epochs:])), vals[skip_epochs:], label='validation', ls='--')
        ax.legend()
    
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

    