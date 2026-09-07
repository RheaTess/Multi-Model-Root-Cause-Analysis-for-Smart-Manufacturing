# Multi-Model Root Cause Analysis for Smart Manufacturing

## Overview

This project implements a multi-model Root Cause Analysis (RCA) framework for smart manufacturing systems using deep learning. The framework analyzes fault-related data from three manufacturing subsystems:

- Coolant
- Hydraulics
- Probe

Each subsystem is modeled independently using a Multi-Layer Perceptron (MLP) and a 1D Convolutional Neural Network (1D CNN). The predictions from both models are combined using late fusion to support multi-label root-cause identification.

## Project Objectives

- Identify possible root causes from manufacturing fault data.
- Extract meaningful features from sensor and machine-state data.
- Support multiple simultaneous root causes using multi-label classification.
- Learn complementary feature representations using MLP and 1D CNN models.
- Combine model predictions using late fusion.
- Generate ranked and human-readable root-cause summaries.

## Dataset and Subsystems

The project uses CNC vertical lathe fault data from the causRCA dataset. The implementation processes the following subsystems independently:

| Subsystem | Description |
|---|---|
| Coolant | Cooling and pump/filter related machine conditions |
| Hydraulics | Hydraulic pressure, level, temperature and component conditions |
| Probe | Machine probe and position-related conditions |

The raw data contains time-based sensor values, binary machine states, alarms, and categorical operating information.

## Methodology

The overall pipeline is:

```text
Raw Fault Data
      |
      v
Subsystem-wise Data Processing
      |
      v
Feature Extraction
      |
      v
Root-Cause Label Encoding
      |
      v
Data Preprocessing
      |
      +-------------------+
      |                   |
      v                   v
     MLP               1D CNN
      |                   |
      +--------+----------+
               |
               v
          Late Fusion
               |
               v
     Ranked Root-Cause Candidates
               |
               v
   Human-Readable RCA Summary
```

## Feature Extraction

The raw fault files are transformed into fixed-length feature vectors.

Features are extracted by comparing conditions before and after fault events. Depending on the signal type, the implementation derives features such as:

- Mean
- Standard deviation
- Before/after changes
- Absolute and relative changes
- Binary state ratios
- State transitions
- Counter changes
- Categorical distributions
- Alarm-related features

This converts variable-length manufacturing signals into a structured feature matrix suitable for machine learning.

## Root-Cause Encoding

Manufacturing faults may have more than one associated root cause. Therefore, the project uses multi-label classification.

`MultiLabelBinarizer` converts root-cause labels into binary target vectors.

Example:

```text
Root causes:
[Cause_A, Cause_B]

Encoded target:
[1, 1, 0, 0]
```

This allows the model to assign probabilities independently to multiple root-cause classes.

## Data Preprocessing

The preprocessing pipeline includes:

### 1. Feature Reduction

Features with more than 80% missing values are removed to reduce sparsity and eliminate features with insufficient information.

### 2. Train-Test Split

The processed dataset is divided into training and testing sets.

### 3. Median Imputation

Remaining missing numerical values are replaced using median imputation. This provides a robust way to handle incomplete sensor features.

### 4. Feature Standardization

Features are standardized using `StandardScaler`:

```text
x_scaled = (x - mean) / standard_deviation
```

Standardization ensures that features with different numerical ranges contribute appropriately during neural network training.

## Models

### Multi-Layer Perceptron (MLP)

The MLP learns global relationships among the extracted manufacturing features.

The architecture uses fully connected layers with:

- Dense layers
- ReLU activation
- Batch normalization
- Dropout
- Sigmoid output layer

The model produces an independent probability for each possible root cause.

### 1D Convolutional Neural Network (CNN)

The 1D CNN learns local patterns from the ordered feature representation.

The architecture includes:

- Conv1D layers
- ReLU activation
- Batch normalization
- Max pooling
- Dropout
- Dense layers
- Sigmoid output layer

The CNN processes the engineered feature vector after reshaping it into a one-dimensional sequence.

## Training

Both models use:

- Optimizer: Adam
- Loss function: Binary Cross-Entropy
- Activation: ReLU for hidden layers
- Output activation: Sigmoid
- Batch normalization
- Dropout for regularization
- Mini-batch training

The models learn through forward propagation, loss calculation, backpropagation, gradient computation, and optimizer-based weight updates.

## Late Fusion

The MLP and CNN predictions are combined using equal-weight late fusion.

For each root cause:

```text
P_fused = 0.5 * P_MLP + 0.5 * P_CNN
```

where:

- `P_MLP` is the probability predicted by the MLP.
- `P_CNN` is the probability predicted by the 1D CNN.
- `P_fused` is the combined probability.

This allows both model architectures to contribute to the final root-cause evidence.

## Root-Cause Analysis Output

The final pipeline generates:

- Predicted root-cause probabilities
- Top-1 root-cause candidate
- Top-3 root-cause candidates
- Case-wise root-cause analysis
- Subsystem-wise candidate aggregation
- Human-readable RCA summaries

The final output is designed to make model predictions easier to interpret for manufacturing fault analysis.

## Project Structure

```text
Multi_Model_RCA_Smart_Manufacturing/
|
|-- Multi_Model_RCA_Smart_Manufacturing.ipynb
|-- README.md
|
|-- Dataset/
|    |-- causRCA fault data
|
```

## Installation

Install the required libraries:

```bash
pip install pandas numpy scikit-learn tensorflow
```

Additional dependencies may be required depending on the dataset setup and notebook environment.

## Running the Project

1. Clone this repository.
2. Open `Multi_Model_RCA_Smart_Manufacturing.ipynb`.
3. Configure the dataset path if required.
4. Run the notebook from top to bottom.

The notebook executes the following stages:

1. Dataset preparation
2. Coolant processing and model training
3. Hydraulics processing and model training
4. Probe processing and model training
5. MLP and CNN prediction fusion
6. Final root-cause ranking
7. Human-readable RCA generation

## Technologies Used

- Python
- Google Colab / Jupyter Notebook
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras

## Key Techniques

- Multi-label classification
- Feature engineering
- Missing-value filtering
- Median imputation
- Feature standardization
- Multi-Layer Perceptron
- 1D Convolutional Neural Network
- Late fusion
- Root Cause Analysis
- Explainable human-readable prediction summaries

## Future Improvements

Possible extensions include:

- Training on a larger number of manufacturing fault cases.
- Hyperparameter optimization.
- Alternative fusion strategies with learned weights.
- Temporal deep learning directly on raw sensor sequences.
- SHAP or other feature-attribution methods for model explainability.
- Real-time integration with manufacturing monitoring systems.

## Authors
 Rhea Tess Payyapilly


## Project Domain

**Artificial Intelligence | Machine Learning | Deep Learning | Smart Manufacturing | Root Cause Analysis**

