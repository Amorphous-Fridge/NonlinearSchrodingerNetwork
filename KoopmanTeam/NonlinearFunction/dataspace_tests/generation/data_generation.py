import numpy as np
import math
from math import e
import pandas as pd
import random
from scipy.linalg import expm
import cmath
import os

'''
This file handles the generation of the datasets.  A dataset with the specified 
parameters is created and the hamiltonian that generated it is saved under the same name
to a seperate hamiltonians directory.  Currently the hamiltonian is hardcoded on the 
last line of code in this file.

PARAMS TO CONFIGURE:
--------------------

DATADIR: Directory to save the dataset to.  This is not the name of the 
         dataset, just the directory that is resides in (naming is handled 
         automatically based on the parameters of the dataset)
         The hamiltonians for our datasets are stored in a seperate 
         subdirectory of DATADIR called 'hamiltonians/'.

TIMERANGE*: Sets period of time (starting from 0) which the system 
            should be evolved through.

TIMESTEP*: Sets the timestep for the evolution

MAXEVOLVE: The total number of evolutions we wish to generate
           NOTE: The number of evolutions actually trained on is set
           in searches.py, so this value should be whatever the largest 
           number of initial conditions we expect to use is.

*One of TIMERANGE or TIMESTEP can be set via the slurm array ID to make generating
multiple evolutions at once easier.  If this is done, make sure that the parameter is not
reset to a fixed value in the PARAMS TO CONFIGURE section

'''



TIMERANGE = int(os.getenv('SLURM_ARRAY_TASK_ID'))


##############PARAMS TO CONFIGURE###########3
DATADIR='/fslhome/wilhiteh/datasets/'
#TIMERANGE=5  #uncomment for manual selection, otherwise taken care of by slurm ID
TIMESTEP=0.025
MAXEVOLVE=50000
#############################################


points_per_file = int(np.ceil(TIMERANGE/TIMESTEP))
num_files = MAXEVOLVE
dataset = '0t{}_{}inits_dt{}/'.format(TIMERANGE,num_files,str(TIMESTEP).replace('.','p'))

if not (os.path.isdir(DATADIR+dataset)):
    os.mkdir(DATADIR+dataset)
if not(os.path.isdir(DATADIR+'hamiltonians/'+dataset)):
    os.mkdir(DATADIR+'hamiltonians/'+dataset)


def check_hermitian(A):
    Adag = A.getH()
    if not (np.array_equal(np.matmul(Adag,A),np.matmul(A,Adag))):
        return False
    return True

def generate_hermitian(a,b,c,d):
    return np.matrix([[a,c+d*1j],[c-d*1j,b]])

def generate_random_hermitian(n):
    A = np.matrix(np.zeros((n,n),np.cdouble))
    constants = np.zeros(n*n)
    for i in range(0,n*n):
        constants[i] = random.random()
    constant_iter = 0
    for i in range(0,n-1):
        A[i,i] = constants[constant_iter]
        constant_iter += 1
        for j in range(i+1,n):
            A[i,j] = constants[constant_iter] + 1j*constants[constant_iter+1]
            A[j,i] = constants[constant_iter] - 1j*constants[constant_iter+1]
            constant_iter += 2
    A[n-1,n-1] = constants[constant_iter]
    return A


def get_magnitude(v):
    sum_of_squares = 0
    for i in v:
        sum_of_squares += i**2
    return cmath.sqrt(sum_of_squares)

def evolve(v,A,t):
    A = -A*1j*t
    A = expm(A)
    return np.matmul(A,v)

def generate_evolution_matrix(v,A,n,t_step=1):
    evolution_matrix = np.array([v,evolve(v,A,t_step)])
    for i in range(2,n):
        evolution_matrix = np.vstack( (evolution_matrix,np.array([evolve(v,A,i*t_step)])) )
    return evolution_matrix

# unspecified dimension defaults to 2x2 hamiltonian
def generate_data_2d(filename,ham_filename,num_evolutions,normal_sampling=True,H=None,t_step=1):
    # choose random coefficients for hermitian hamiltonian
    #if one is not provided by H
    if type(H) != np.ndarray:
        a=random.random()
        b=random.random()
        c=random.random()
        d=random.random()
        H = generate_hermitian(a,b,c,d)

    # generate random 2 dimensional vector
    if normal_sampling: #Normal dist. is spherically symmetric, better for bloch sphere
        vec = np.random.randn(2)
    else:
        vec = 2*np.random.rand(2)-1

    # change vector to have norm of 1
    vec = vec/get_magnitude(vec)

    # choose number of evolutions
    num_rows = num_evolutions
    evolution_matrix = generate_evolution_matrix(vec,H,num_rows,t_step=t_step)

    #extract real and imaginary parts out of evolution matrix
    split_evolution_matrix = np.zeros((evolution_matrix.shape[0],4))
    for row in range(0,evolution_matrix.shape[0]):
        split_evolution_matrix[row, 0] = np.real(evolution_matrix[row, 0])
        split_evolution_matrix[row, 1] = np.imag(evolution_matrix[row, 0])
        split_evolution_matrix[row, 2] = np.real(evolution_matrix[row, 1])
        split_evolution_matrix[row, 3] = np.imag(evolution_matrix[row, 1])
    
    # export data as csv
    mat_df = pd.DataFrame(split_evolution_matrix)
    mat_df.to_csv(filename)
    ham_df = pd.DataFrame(H)
    ham_df.to_csv(ham_filename)
    
    
    
#Using the old data loader (squish everything into memory at once), we can run up to 5000000 points in just over 8GB of memory
    
for i in range(num_files):
    generate_data_2d(DATADIR+dataset+'evolution{}.csv'.format(i), DATADIR+'hamiltonians/'+dataset+'ham.csv', points_per_file, H=np.array([[0.5566942775987375+0*1j, 0.9560186831757848+0.784645945006308j],[0.9560186831757848-0.784645945006308j,0.0872675534068238+0*1j]]), t_step=TIMESTEP, normal_sampling=True)
