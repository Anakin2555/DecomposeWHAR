# DecomposeWHAR
The official code for the AAAI 2025 submission, titled "Decomposing and Fusing Intra- and Inter-Sensor Spatio-Temporal Signals for Multi-Sensor Wearable Human Activity Recognition."

## Directory Structure

- **Dataset/**: Contains scripts and data for preprocessing and handling datasets used in training and testing the models. Put the preprocessed data into the following directory structure:
  - opp/
    - opp_24_12/
    - opp_60_30/
       ... # preprocessed data of Opportunity
  - realdisp/
    - realdisp_40_20/
    - realdisp_100_50/
       ... # preprocessed data of Realdisp
  - skoda/
    - skoda_right_78_39/
    - skoda_right_196_98/
       ... # preprocessed data of Skoda
  
- **layers/**: Includes the core layer implementations used in our model.

- **models_layers/**: Contains scripts that define the neural network models, integrating various layers and modules.

- **utils/**: Utility functions that support data processing, model training, evaluation, and other common operations.

- **main.py**: The main script to run the training and evaluation of the models. This file orchestrates the entire pipeline from data loading to model inference.

- **modules.py**: Defines the main modules of our model.

- **utils_this.py**: Additional utility functions.

## Prerequisites
- Python: >= 3.8
- Pytorch: >= 2.2.0 + cu118
- Mamba-ssm: 1.2.0
- Causal-conv1d: 1.2.0


## Getting Started

1. **Setup**: Ensure that all required dependencies are installed.
2. **Run the model**: Execute the `main.py` script to start training or evaluating the model.
