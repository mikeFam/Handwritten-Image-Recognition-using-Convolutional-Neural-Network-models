# Handwritten Image Recognition using Convolutional Neural Network models
### Author: Sky Nguyen, Michael Pham Group : 23

## Description: 
  This is an image recognition program for recognizing handwritten digits images from the MNIST dataset. Four different achitectures of the Convolution Neural Network was implemented, a modified CNN model, LeNet, VGG and a simple CNN model. The objective of this program is to evaluate these different models in terms of the learning curve, accuracy and training time in respect of the application.

## Things to note: 
  1. Ensure 'scikit-learn' is installed (might require a restart after installation), instructions can be found on: https://anaconda.org/anaconda/scikit-learn 
  2. Our program does not use GPU for simplicity, thus the training time is very long.

## Instruction (how to run):
  1. Open Command Prompt 
  2. Redirect to the project folder "CS302-Python-2020-Group23" 
  3. Run command: python main.py 
  4. Enter the name of the model and the number of epochs to run 
  5. As the program is executing, the selected model will start the training and testing process. Information of the results of the training and the testing processes will appear as it continues to run. 
  6. Wait for the simulation to finished, plots will appear showing the loss factors in the training process and the improvement of accuracy in after the testing process
