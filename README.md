# Glioma Detection from SRH images

This project trains a LeViT model on a dataset of annotated SRH images of glioma to investigate its efficacy at identifiying 6 types of tumor and specifically to investigate the accuracy of the 2 channels of an SRH model individually and in comparison to the full image.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Model](#model)
- [Results](#results)
- [Literature](#literature)
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

    The options for the hyperparameters are set in the core/global_vars.py file.

2. Train a model and save it

    ```bash
    python main.py -db /your/custom/path -lf save/results.txt -mf save/model.pth -lr learning_rate -bs batch_size
    ```

    This will train a model on a fixed random training subset with early stopping based on validation improvement.
    The resulting best weights are then saved to save/model.pth.
    The losses and accuracies are saved to save/results.txt.

    The training subset is determined by the SPLIT_SEED variable in the core/global_vars.py file.

3. Load a model and evaluate it

    ```bash
    python evaluator.py -db /your/custom/path -lf save/results.txt -mf load/model.pth
    ```

    This method loads a model from the load/model.pth file and evaluates it on a test dataset.

## Data

The data used in this project was provided by the University of Cologne and includes SRH images of 214 patients, split into patches with annotations per patch classifying them into 6 classes of tumor, normal and non-diagnostic.

To start the training process, some patches had to be removed from consideration as they were only annotated as unspecific tumors and could not be classified as a specific type of tumor.
Additionally, the images were resized to fit with the general LeViT structure.

## Model

### Algorithms

The Model used here is a slightly modified version of [LeViT][levit_link], a combination of a Convolutional Neural Network (CNN) with Vision Transformers in a relatively lightweight format.
The specific model was a LeViT 384 architecture.

The only modification from the original [Timm][timm_link] implementation is a difference in the input layers to account for the different image type and channel variation.

The optimizer was an Adam algorithm with a learning rate of 0.0001 and a weight decay of 0.01.
Our loss function was Crossentropy-loss.

### Hyperparameters

- Learning Rate: 1e-4
- Batch Size: 16
- Weight decay: 0.01
- Early Stopping Patience: 3
- Test split: 0.25
- Validation split: 0.2
- Split seed: 42

### Evaluation

The model was evaluated on the accuracy of its predictions on a test set.
Additionally we created a confusion matrix as shown in the [results](#results) section.

## Results

The first step of the project was to perform cross validation to obtain good hyperparameters.
In our experimentation we found the most effective learning rate and batch size to be 0.0001 and 16, which are the defaults for all further uses.

When applied to our dataset the most effective classifier was to use both channels, which resulted in an accuracy of about 86.06% after 14 epochs of training.
The first channel of the images, which shows mostly lipids, resulted in an accuracy of 64.42% after 9 epochs of training.
The second channel of the images, showing cellular structures, resulted in a very low accuracy of 26.06% after 4 epochs of training.

The confusion matrices are:

<div style="display: flex; justify-content: space-between;">
  <div>
    <img src="core\plots\confusion_matrix_binary_channel.png" alt="Image 1" width="80%"/>
    <p><em>Confusion matrix both channels</em></p>
  </div>
  <div>
    <img src="core\plots\confusion_matrix_channel0.png" alt="Image 2" width="80%"/>
    <p><em>Confusion matrix channel 0</em></p>
  </div>
  <div>
    <img src="core\plots\confusion_matrix_channel1.png" alt="Image 3" width="80%"/>
    <p><em>Confusion matrix channel 1</em></p>
  </div>
</div>

In addition ot the confusion matrices the UMAPs of each channel combination are:

<div style="display: flex; justify-content: space-between;">
  <div>
    <img src="core\plots\umap_projection_binary_channel.png" alt="Image 1" width="80%"/>
    <p><em>UMAP both channels</em></p>
  </div>
  <div>
    <img src="core\plots\umap_projection_channel0.png" alt="Image 2" width="80%"/>
    <p><em>UMAP channel 0</em></p>
  </div>
  <div>
    <img src="core\plots\umap_projection_channel1.png" alt="Image 3" width="80%"/>
    <p><em>UMAP channel 1</em></p>
  </div>
</div>

## Literature

[timm_link]: https://doi.org/10.5281/zenodo.4414861
[levit_link]: https://openaccess.thecvf.com/content/ICCV2021/html/Graham_LeViT_A_Vision_Transformer_in_ConvNets_Clothing_for_Faster_Inference_ICCV_2021_paper.html
[main_paper_link]: https://www.nature.com/articles/s41591-019-0715-9

- [Wightman, R. PyTorch Image Models (Version 1.0.11) [Computer software].][timm_link] - Package including the used model

- [Graham, B., El-Nouby, A., Touvron, H., Stock, P., Joulin, A., Jégou, H., & Douze, M. (2021). Levit: a vision transformer in convnet's clothing for faster inference. In Proceedings of the IEEE/CVF international conference on computer vision (pp. 12259-12269).][levit_link] - Original paper on LeViT models

- [Hollon, T. C., Pandian, B., Adapa, A. R., Urias, E., Save, A. V., Khalsa, S. S. S., ... & Orringer, D. A. (2020). Near real-time intraoperative brain tumor diagnosis using stimulated Raman histology and deep neural networks. Nature medicine, 26(1), 52-58.][main_paper_link] - Original paper on the application of CNN on SRH images

## License

This project is licensed under the Apache License 2.0 - See the LICENSE file for details.
