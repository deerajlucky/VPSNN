

# VPSNN
Spiking/Artificial Variable Projection Neural Networks

# Running Online

The example trains a spiking VPNN, with temporal deattenuation spike encoder in latent space, on ECG heartbeats of [MITBIH Arrhythmia] dataset (https://physionet.org/content/mitdb/1.0.0/) in order to perform binary discrimination between ectopic and normal beats. Training set was selected from 23 records (numbered from 100 to 124) while the test set includes the rest of available records in the dataset. More information about the splitting strategy of records in training and test sets can be found [here](
https://archive.physionet.org/physiobank/database/html/mitdbdir/intro.htm#selection).


# Running and testing on local machines

Before running [the example notebook](notebooks/vpsnn_cpu.ipynb), you need to install the python environment using:

>	conda env create -f environment.yml 

