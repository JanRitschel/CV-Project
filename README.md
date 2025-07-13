# Glioma Detection from SRH images

This project trains a LeViT model on a dataset of annotated SRH images of glioma to investigate its' efficacy at indetifiying 6 types of tumor and specifically to investigate the accuracy of the 2 channels of an SRH model individually and in comparison to the full image.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Model](#model)
- [Results](#results)
- [Literature](#literature)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/JanRitschel/CV-Project.git
    ```

2. Navigate to the project repository

    ```bash
    cd /your/installation/path/core
    ```

3. Install the required packages

    ```bash
    pip install -r requirements.txt
    ```

4. Save the database in a directory of the following structure:

- /your/custom/path
  - studies
    - patient folders
  - meta
    - opensrh.json

## Usage

The code in this project can be used in three main ways.

1. Run cross validation to find the optimal learning rate and batch size:

    ```bash
    python cross_validator.py -db /your/custom/path
    ```

    This will print the optimal batch size and learning rate upon completion, which can be used for the other cases. </br>
    The function "cross_validate()" also returns these rates.

    The options for the hyperparamters are set in the core/global_vars.py file.

2. Train a model and save it

    ```bash
    python main.py -db /your/custom/path -lf save/results.txt -mf save/model.pth -lr learning_rate -bs batch_size
    ```

    This will train a model on a fixed random training subset with early stopping based on validation improvement.
    The resulting best weights are then saved in the save/model.pth file.
    The losses and accuracies are saved in the save/results.txt file

    The training subset is based on the SPLIT_SEED variable in the core/global_vars.py file.

3. Load a model and evaluate it

    ```bash
    python evaluator.py -db /your/custom/path -lf save/results.txt -mf load/model.pth
    ```

    This method loads a model from the load/model.pth file and evaluates it on a test dataset.

## Data

The data used in this project was provided by the University of Cologne and includes SRH images of XXXXXXX patients, split into patches with annotations per patch classifying them into 6 classes of tumor, normal and non-diagnostic.


## Model

## Results

In our experimentation we found the most effective learning rate and batch size to be 0.0001 and 16, which are the defaults for all other use cases.

## Literature

## Contributing

## License
