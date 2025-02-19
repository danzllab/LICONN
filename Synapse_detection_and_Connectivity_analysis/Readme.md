Synapse detection and connectivity analysis code for LICONN.

## Requirements

- Miniconda - conda 24.4.0
- Pytorch 1.12.1

The code was tested on Ubuntu 20.04 and Debian GNU/Linux 12

All jupyter notebooks were executed on a classical laptop without GPU and don't require extra RAM beyond that available on standard laptops.
Deep-learning parts (training and prediction) were executed using an HPC Cluster and require a GPU (e.g. NVIDIA A10, NVIDIA A40 or similar). The amount of RAM requested for training was usually around 500 GB, which can be reduced by using smaller batch size. The amount of RAM required for the prediction step depends on the size of the volume and usually is below 100 GB.
Connectivity analysis scripts require large amounts of RAM (~ 200 GB) and a large amount of disk space to store the results (~50 GB).
These scripts (liconn_computeConectivityMatrix.py and liconn_analyzeConnectivity.py) were executed using HPC Cluster.

## Contents:
### Scripts:
 1. Immunostaining-based synapse detection
    - [immuno_synapse_detection.ipynb](./Scripts/Immunostaining-based%20synapse%20detection/immuno_synapse_detection.ipynb)
	--> Jupyter Notebook for synapse detection and validation based on real immunostaining signal. More detailed instructions on how to use the code are written in the notebook.
    - [detect_gephyrin.ipynb](./Scripts/Immunostaining-based%20synapse%20detection/detect_gephyrin.ipynb)
	--> Jupyter Notebook for inhibitory post-synapses based on gephyrin immunostaining.

    Cost matrix calculation in the Hungarian algorithm for matching detections to ground truth was calculated as in the Synful implementation by the Funke lab: https://github.com/funkelab/synful
    
    Expected run time: ~10 minutes

 2. DL-based synapse detection
    
    DL-based synapse detection is implemented using the Gunpowder library (https://github.com/funkelab/gunpowder). 

    The method uses U-Net architecture presented in the following paper: "Label-free prediction of three-dimensional fluorescence images from transmitted light microscopy" by Ounkomol et al. in Nature Methods, 2018 (https://www.nature.com/articles/s41592-018-0111-2) with code available at: https://github.com/AllenCellModeling/pytorch_fnet

    Synapse prediction part:
    
    The following 3 scripts are used to train the network to predict Bassoon (Shank2) immunostaining from the LICONN structural channel. 
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
	--> Script used to compute the connectivity matrix based on the volume shown in Fig2a of the LICONN manuscript.
    - [liconn_analyzeConnectivity.py](./Scripts/Conectivity%20analysis/liconn_analyzeConnectivity.py)
	--> Script used to analyze the connectivity matrix computed in the previous step.

Expected run time: around 30 minutes for each script, since big datasets need to be loaded into the memory

 4. Utils
    - [nml2json.ipynb](./Scripts/Utils/nml2json.ipynb)
	--> Jupyter Notebook used to convert .nml (webknossos format) manual annotations to .json for later use. 
Expected run time: minutes.

### Data:
This folder contains test datasets needed to run the scripts.

Datasets which exceed the size limit of the Git repository can be accessed and downloaded from the Institute of Science and Technology Austria’s data repository:
https://doi.org/10.15479/AT:ISTA:18697 (https://research-explorer.ista.ac.at/record/18697 )

To run the scripts using demo datasets, put the downloaded files into the Data folder.

To run a script on your data, simply place the data in the Data folder and change the paths in the scripts. 

## Installation:
In order to install the required libraries and external packages, create a conda environment following the instructions below:
```
conda env create -f liconn.yml
conda activate liconn
pip install git+https://github.com/funkelab/funlib.learn.torch
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```
Installation time is ~8 minutes

### Notes:

1) 
The Deep-learning based synapse detection method uses U-Net architecture presented in the following paper: "Label-free prediction of three-dimensional fluorescence images from transmitted light microscopy" by Ounkomol et al. in Nature Methods, 2018 (https://www.nature.com/articles/s41592-018-0111-2) with code available at: https://github.com/AllenCellModeling/pytorch_fnet\ available under the following conditions:

This software license is the 2-clause BSD license plus clause a third clause that prohibits redistribution and use for commercial purposes without further permission.
Copyright © 2018. Allen Institute. All rights reserved.

1. Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

2. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

3. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.Redistributions and use for commercial purposes are not permitted without the Allen Institute’s written permission. For purposes of this license, commercial purposes are the incorporation of the Allen Institute's software into anything for which you will charge fees or other compensation or use of the software to perform a commercial service for a third party. Contact terms@alleninstitute.org for commercial licensing opportunities.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Please consider citing also that paper when using the code. 

2) 
The Deep-learning based synapse detection method uses the Gunpowder library with the code available at: https://github.com/funkelab/gunpowder under the following conditions:

MIT License

Copyright (c) 2020 Jan Funke

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

3) 
Synapse detection validation includes a step that optimally assigns ground truth synapses to detected synapses in a one-to-one manner. For this, an implementation of the Hungarian matching algorithm from the Synful repository (https://github.com/funkelab/synful) was partially adopted. The code is available under the following conditions:

MIT License

Copyright (c) 2020 Funke Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.







