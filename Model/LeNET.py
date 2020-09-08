import torch.nn as nn
import torch.nn.functional as F
import torch

class LN(nn.Module):
    def __init__(self):
        super(LN, self).__init__()
        # loss_history is a list contain prediction error of the model through the training dataset
        # acc_history is a list contain accuracy percentage of the model through the test dataset
        self.loss_history = []
        self.acc_history =[]

        # Sequential function allow to make the forward function to be more concise

        # Declare convolution layer format : (number of channels in the input image,
        #                              number of channels produced by the convolution,
        #                              size of the convolving kernel,
        #                              Stride of the convolution)

        # Average pooling layer format: (the size of the window to take a average over)
        # This function reduces the dimensionality of the input and help to make the training progress more manageable

        # In this LeNET model, we decide to use Tanh activation function
        self.conv_model = nn.Sequential(nn.Conv2d(1, 6, 5),
                                        nn.Tanh(),
                                        nn.AvgPool2d(2, stride=2),
                                        nn.Conv2d(6, 16, 5),
                                        nn.Tanh(),
                                        nn.AvgPool2d(2, stride=2)
                                        )

        # Fully connected layers format: (size of each input sample,
        #                                 size of each output sample)
        self.fc1 = nn.Sequential(nn.Linear(256, 120),
                                 nn.Tanh(),
                                 nn.Linear(120, 84),
                                 nn.Tanh(),
                                 nn.Linear(84, 10)
                                 )

    def forward(self, x):
        x = self.conv_model(x)

        # Flatten layer will flatten the input into an 1D array before feed it to the fully connected layer
        x = torch.flatten(x, 1)
        x = self.fc1(x)

        # Log_softmax function will turn logits into probabilities that sum to one
        # The function outputs a vector that contains all the probability this model predicts to the corresponding class
        output = F.log_softmax(x, dim=1)
        return output
