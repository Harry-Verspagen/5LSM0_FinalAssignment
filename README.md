# Final Assignment

This repository contains all the code required to reproduce the results I have optained for the Final Assignment for the course 5LSM0 Neural Networks for Computer Vision.


## Getting Started

### Dependencies

This code only requires basic dependencies such as torch, torchvision, matplotlib and numpy.

### File Descriptions

Here's a brief overview of the files you'll find in this repository:

- **run_container.sh:** Contains the script for running the container. In this file you have the option to enter your wandb keys if you have them and additional the hyperparameters can be adjusted for training the different models.

  
- **run_main:** Includes the code for building the Docker container. 
  

- **model.py:** This is the baseline model U-net.
- **unet2.py:** This is the U-net++ model.
- **model_smaller.py** This is the smaller U-net.
- **Small_model.py** This is the Enet model.

  
- **train.py:** Contains the code for training the neural network. Inside this file it is required to manually change whether to include different data transformations and the learning rate schedular. Futher the model must be chosen inside this file. Most other hyperparamters can be adjusted in the run_container.sh file.


- **plotresults.py:** Contains the code that was used to produce various plots for different models. Inside this file it is required to manually specify the data transformations used, the model used, and the location to the trained model.


- **utils.py:** The file that was provided for class selection, which is also used for coloring the labels.

### Author

- H.E.A.M. Verspagen
