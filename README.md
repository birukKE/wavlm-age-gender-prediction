# WavLM Age and Gender Prediction
Predicting speaker age and gender from speech using WavLM embeddings. 
This project extracts pre-trained WavLM features from audio and further trains 
classifiers for demographic prediction using a multi-task learning approach 
with a shared encoder backbone and task-specific classification heads.
The src folder contains the python source code and
the data folder contains the final extracted WavLM embeddings.

## Features
- Extract WavLM embeddings from speech audio
- Gender classification (male/female)
- Age group classification (teens to nineties)
- PyTorch-based training pipeline

