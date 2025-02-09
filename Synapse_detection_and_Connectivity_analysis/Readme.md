Synapse detection and connectivity analysis code for the LICONN project.

## Requirements

- Miniconda - conda 24.4.0
- Pytorch 1.12.1

The code was tested on Ubuntu 20.04 and Debian GNU/Linux 12

All jupyter notebooks were executed on a classical laptop without GPU and don't equire any extra RAM.
Deep-learning parts (training and prediction) require single GPU (e.g. NVIDIA A10, NVIDIA A40 or similar). The amount of RAM requested for training was normally ~500 Gb, which can be reduced by using smaller batch size. The amount of RAM required for prediction step depends on the size of the volume and normally is below 100 Gb.
Connectivity analysis scripts require large amounts of RAM (~ 200 Gb) and a large amount of disk space to store the results (~50 Gb).
These scripts (liconn_computeConectivityMatrix.py and liconn_analyzeConnectivity.py) were executed using HPC Cluster.

## Contents:
### Scripts:
 1. Immunostaining-based synapse detection
    - [immuno_synapse_detection.ipynb](./Scripts/Immunostaining-based%20synapse%20detection/immuno_synapse_detection.ipynb)
	--> Jupyter Notebook used to run synapse detection and validation based on real immunostaining signal. More detailed instructions on how to use the code are written in the notebook.
    - [detect_gephyrin.ipynb](./Scripts/Immunostaining-based%20synapse%20detection/detect_gephyrin.ipynb)
	--> Jupyter Notebook used to detect inhibitory post-synapses based on gephyrin immunostaining.

    Expected run time: ~10 minutes

 2. DL-based synapse detection
    
    Synapse prediction part:
    
    The following 3 scripts are used to train the network to predict bassoon(shank2) immunostaining from the LICONN structural channel. 
    The first step is to convert datasets into .zarr format used during training. Next, the model is trained, and the saved checkpoints can be used to predict immunostainings on the test data.
	- [liconn_convert_data.py](./Scripts/DL-based%20synapse%20detection/liconn_convert_data.py)
    - [liconn_train.py](./Scripts/DL-based%20synapse%20detection/liconn_train.py)
	- [liconn_predcit.py](./Scripts/DL-based%20synapse%20detection/liconn_predict.py)

    Expected run time:
    Training time depends on the number of iterations. Training for 10K iterations typically takes ~ 3 hours. Prediction times depend on the volume size, for single-tiled data it usually takes less than 10 minutes.
    
    Synapse detection part:
    - [dl_synapse_detection.ipynb](./Scripts/DL-based%20synapse%20detection/dl_synapse_detection.ipynb)
	    --> Jupyter Notebook used to detect synapses from the predicted immunostaining channels (outputs of the liconn_predict.py scripts)
    
    Expected run time:  ~ 15 minutes


 3. Connectivity analysis
    - [liconn_computeConectivityMatrix.py](./Scripts/Conectivity%20analysis/liconn_computeConnectivityMatrix.py)
	--> Script used to compute the connectivity matrix based on the volume shown in Fig2a.
    - [liconn_analyzeConnectivity.py](./Scripts/Conectivity%20analysis/liconn_analyzeConnectivity.py)
	--> Script used to analyze the connectivity matrix computed in the previous step.

Expected run time: around 30 minutes for each script, since big datasets need to be loaded into the memory

 4. Utils
    - [nml2json.ipynb](./Scripts/Utils/nml2json.ipynb)
	--> Jupyter Notebook used to convert .nml (webknossos format) manual annotations to .json for later use. 
Expected run time: minutes.

### Data:
This folder contains test datasets needed to run the scripts.

Some of the datasets, which exceed the size limit, can be accessed and downloaded from the Institute of Science and Technology Austria’s data repository:
https://doi.org/10.15479/AT:ISTA:18697 (https://research-explorer.ista.ac.at/record/18697 )

To run the scripts using demo datasets, put the downloaded files into the Data folder.

To run a script on your data, simply place it in the Data folder and change the paths in the scripts. 

## Installation:
In order to install the required libraries and external packages, create a conda environment following the instructions below:
```
conda env create -f liconn.yml
conda activate liconn
pip install git+https://github.com/funkelab/funlib.learn.torch
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```
Installation time is ~8 minutes






