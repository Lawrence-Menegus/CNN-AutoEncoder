# Convolutional-Autoencoders-on-Flower-Dataset

<p>The provided program implements a Convolutional Autoencoder for reconstructing flower images. The program utilizes TensorFlow and Keras to preprocess data, build an encoder-decoder architecture, and train the model for image reconstruction tasks.</p>

### Key Features
#### Data Preparation:
Supports dataset extraction and loading on both Google Colab and local machines.
Preprocesses images with normalization and resizing for training and validation.
#### Model Architecture:
Encoder: Uses convolutional layers and max-pooling for feature extraction.
Decoder: Mirrors the encoder with upsampling layers for reconstruction.
Autoencoder: Combines encoder and decoder for end-to-end image reconstruction.
#### Training and Evaluation:
Compiles the model with the Adam optimizer and Mean Squared Error (MSE) loss.
Tracks loss during training for both training and validation datasets.
#### Visualization:
Displays input images, latent space embeddings, and reconstructed outputs for qualitative evaluation.

### Install the Package
pip install tensorflow matplotlib
<p>in the terminal of a Python environment</p> <p>Ensure you have TensorFlow and Matplotlib installed before running the program. For TensorFlow installation instructions, visit [TensorFlow Installation](https://www.tensorflow.org/install).</p>

### Running the Program
#### Dataset Preparation
Extract the flower dataset from the zip file to the appropriate directory.
Structure the dataset into training and validation directories.
#### On Google Colab
Mount Google Drive and set paths to the dataset accordingly.
Use the provided script to unzip and organize the dataset.
#### On Local Machine
Ensure the dataset path is correctly set in the script for training and validation directories.
Use the provided script to unzip the dataset.
#### Execution
Preprocess the dataset using either manual preprocessing or image_dataset_from_directory.
Build and compile the autoencoder model.
Train the model using the preprocessed training and validation datasets.
Visualize reconstructed images and embeddings.

## Contributor
<p>Lawrence Menegus</p>
