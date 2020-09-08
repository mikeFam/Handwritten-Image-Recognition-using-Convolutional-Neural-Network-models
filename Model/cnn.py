import torch.nn as nn
import torch.nn.functional as F
import torch

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # loss_history is a list contain prediction error of the model through the training dataset
        # acc_history is a list contain accuracy percentage of the model through the test dataset
        self.loss_history = []
        self.acc_history = []

        # Declare convolution layer format : (number of channels in the input image,
        #                              number of channels produced by the convolution,
        #                              size of the convolving kernel,
        #                              Stride of the convolution)
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1)
        self.conv3 = nn.Conv2d(32, 32, 3, 1)
        self.conv4 = nn.Conv2d(32, 64, 3, 1)
        self.conv5 = nn.Conv2d(64, 64, 3, 1)
        self.conv6 = nn.Conv2d(64, 64, 3, 1)

        # Batch normalize layer format : ( input size of learnable parameter vectors)
        # This function help to smoothen the training progress by normalizing the hidden units
        # activation values so that the distribution of these activations remains same during training
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm2d(64)

        # Max pooling layer format: (the size of the window to take a max over)
        # This function reduces the dimensionality of the input and help to make the training progress more manageable
        self.maxpool1 = nn.MaxPool2d(2)
        self.maxpool2 = nn.MaxPool2d(2)

        # Dropout layer format: (probability)
        # This function will zero out entire channels with the set probability
        # in the function for reducing overfitting in the model.
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)

        # Fully connected layers format: (size of each input sample,
        #                                 size of each output sample)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 10)

    # Feed all declared layers through feedforward neural network
    # In this CNN model, we decide to use ReLU activation function
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = F.relu(x)
        x = self.maxpool2(x)
        x = self.dropout1(x)

        # Flatten layer will flatten the input into an 1D array before feed it to the fully connected layer
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        # Log_softmax function will turn logits into probabilities that sum to one
        # The function outputs a vector that contains all the probability this model predicts to the corresponding class
        output = F.log_softmax(x, dim=1)
        return output
