## DATASPACE SEARCH

- ./teststatus.ods contains info on what tests have been run

- ./searches/ contains the scripts related to actually running a search
    - IMPORTANT NOTE: Unlike previous tests, these new search files assume that the dataset has already been run through a compression network.
    - There are settings to configure both in the python script and the BASH script.

- ./processing/ contains scripts related to processing datasets.
    - We can compress a dataset with a neural net, yielding a new pre-compressed dataset
    - We can find vectors v and w in the dataset such that v+w is also (almost) in the dataset

- ./generation/ contains scripts relevant to actually generating a new dataset
