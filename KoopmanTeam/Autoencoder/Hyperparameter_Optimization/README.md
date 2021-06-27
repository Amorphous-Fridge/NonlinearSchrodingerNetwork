#Hyperparameter Optimization

##Grid Search Over Layers and Neurons

8-512 neurons
3-6 layers

All scripts are specfic to a certain number of layers (making this generalized wasn't really worth the time I think)
I was only able to run the scripts with CPUs on the supercomputer (still working on trying to get GPUs to behave). These can take up to an hour to run
		for a single model.
Tests all possible combinations of widths given a certain number of layers and a list of constraints.
Constraints on widths:
	General 'pyramid' shape (increases and then decreases in both the encoder and the decoder)
	Decoder and encoder are symmetric
	Light assymetries in the encoder/decoder are allowed (one of the layers can be a factor of 2 off from the opposite layer)
		e.g.: layer widths for 5 layers:	8, 16, 256, 32, 8 okay
							128, 256, 512, 256, 128 okay
							8, 16, 128, 32, 16 bad (two are off by 2x)
							8, 64, 128, 64, 32 bad (off by more than 2x)
		***note that on the 6 layer models, the middle two layers were required to have the same widths. This was not required for the 4 layer model
				I did this because it reduced the nubmer of possible models from 255 to 115
	Neurons (widths) allowed to be powers of 2 between 8 and 512

I added another parameter for the write function so that it was easier to track which file was created with which widths.

##QuantumAutoencoder.py
	Reads in the AntiKoopman and Quantum dimensions from an environmenet variable (set by the hyperparameter search script). Defaults to 3,4 respectively
	Contains all of the functions required to run the hyperparameter scripts. may need to update paths in scripts to reflect location of this
	Would still take a very long time to run (not measured) without GPUs.

#GRIDSEARCH SCRIPTS

##multiprocessing
	Uses Python's multiprocessing library to speed up. Mary Lou (supercomputer) allows for up to 16 processors (I think)

##distributed
	This uses a job array of the same size as the total number of possible permutations of widths, given a number of layers and sever constraints (above)
	This is currently the best version that exists, as it is easy to restart if the jobs stop without losing all of the progress.
	Note that you submit the submit<number of layers>.sh file, which submits a job array of Sbatch<num layers>layers.sh files, each of which runs
		one instance of gridsearch_<num layers>layers.py
