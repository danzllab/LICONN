# Distortion analysis

Distortion analysis for LICONN, published in the manuscript "Light-microscopy based connectomic reconstruction of mammalian brain tissue"[1]. For details, see the Methods section of the LICONN manuscript. The code was adapted from previously published methods by the Chen group [2] with the relevant code available at: https://github.com/Yujie-S/Click-ExM_data_process_and_example.git. 

## Contents:
### Scripts:
- [main.m](./main.m/) --> the main distortion analysis that includes the following steps: loading data, pre-processing, non-rigid registration, visualisation of results
- [demons.m](./demons.m/) --> non-rigid registration via imregdemons.m [3]
- [error_calculation.m](./error_calculation.m/) --> calculating the measurement error
- [error_plot.m](./error_plot.m/) --> plotting the measurement error across different measurement lengths
### Demo datasets:
- pre.tif --> pre-expansion 16-bit image (initially aligned to the post-expansion image via similarity transformation) for testing the analysis
- post.tif --> post-expansion 16-bit image for testing the analysis
### Outputs:
- [outputs](./outputs/) --> the folder with the outputs of the distortion analysis: measurement error plot, distortion vector field, etc.

## Requirments:
The code was tested on Ubuntu 20.04 with MATLAB R2022b using 503 GB RAM.

## How to run the distortion analysis:
- In the main.m, adjust parameters such as expansion factor and path to the pre-/post- expansion images
- Run main.m and inspect the outputs such as the measurement error plot, and the distortion vector field

Expected run time: 1-2 minutes for each script on the demo dataset.

---
### References
[1] Tavakoli et al., Light-microscopy based dense connectomic reconstruction of mammalian brain tissue, bioRxiv 2024.03.01.582884; doi: https://doi.org/10.1101/2024.03.01.582884

[2] Sun, D., et al., Click-ExM enables expansion microscopy for all biomolecules. Nat Methods 18, 107–113 (2021). https://doi.org/10.1038/s41592-020-01005-2

[3] https://mathworks.com/help/images/ref/imregdemons.html
