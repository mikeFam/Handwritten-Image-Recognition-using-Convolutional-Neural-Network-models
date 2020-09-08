from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from PIL import Image
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

from Model.LeNET import LN
from Model.SimpleCNN import SimpleCNN_net
from Model.VGG import VGG_net
from Model.cnn import Net
from sklearn.metrics import confusion_matrix, classification_report


# functions to show an image
def imsave(img):
    npimg = img.numpy()
    npimg = (np.transpose(npimg, (1, 2, 0)) * 255).astype(np.uint8)
    im = Image.fromarray(npimg)
    im.save("./results/your_file.jpeg")

# train function for the application
def train_model(log_interval, model, device, train_loader, optimizer, epoch):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output = model(data)

        # Calculate the loss of the model with negative log-likelihood method
        loss = F.nll_loss(output, target)
        loss.backward()

        # Update the parameter based on the current gradient after every test
        optimizer.step()

        # Print out current status of training process
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

        # Save loss percentage value to the loss list created for each model
        if loss.item() >= 1:
            model.loss_history.append(1)
        else:
            model.loss_history.append(loss.item())

def test(model, device, test_loader):
    # Change the model into evaluation mode
    model.eval()

    # Average loss through testing
    test_loss = 0

    # Total number of correct prediction
    correct = 0

    # Create empty tensor to save all of the prediction that the model had made and their corresponding target as well
    all_preds = torch.tensor([])
    all_target = torch.tensor([], dtype=torch.int64)

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # Concatenate current target value and prediction value to the created tensor
            all_target = torch.cat(
                (all_target, target), dim=0)
            all_preds = torch.cat(
                (all_preds, output), dim=0)

            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()

            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)

            # sum up correct prediction
            correct += pred.eq(target.view_as(pred)).sum().item()

    # Create confusion matrix and classification report with all the predictions and target in the test dataset.
    cm = confusion_matrix(all_target, all_preds.argmax(dim=1))
    print('\nConfusion Matrix')
    print(cm)
    target_names = ['number 0', 'number 1', 'number 2', 'number 3', 'number 4', 'number 5', 'number 6', 'number 7',
                    'number 8', 'number 9']
    report = classification_report(all_target, all_preds.argmax(dim=1), digits=3, target_names=target_names)
    print('\nClassification report')
    print(report)

    # Calculate average lost of the model through testing
    test_loss /= len(test_loader.dataset)

    # Print result of the test
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    # Save accuracy value to the accuracy list created for each model
    model.acc_history.append(100. * correct / len(test_loader.dataset))

def main():
    gamma = 0.7
    log_interval = 10
    torch.manual_seed(1)
    save_model = True

    # Choose the model that want to train or test via change the corresponding model name to true
    # Default model: Simple CNN
    CNN = False
    LeNet = False
    Vgg = False

    # Allowing the user to choose which model they would like to run
    print('Please enter one of the following model: CNN, LeNet, Vgg and Simple_CNN')
    SelectModel = input("Model: ")
    if (SelectModel == 'CNN'):
        CNN = True
    elif (SelectModel == 'LeNet'):
        LeNet = True
    elif (SelectModel == 'Vgg'):
        Vgg = True
    else:
        CNN = False
        LeNet = False
        Vgg = False

    # Allow user to change the number of epochs to train
    epoches = int(input('Please enter the number of epochs: '))
    print('Executing model: {}, for {} epochs'.format(SelectModel, epoches))

    # Check whether you can use Cuda
    use_cuda = 0
    # use_cuda = torch.cuda.is_available()
    # torch.backends.cudnn.enabled = False


    # Use Cuda if you can
    device = torch.device("cuda" if use_cuda else "cpu")

    ######################   Torchvision    ###########################
    # Use data predefined loader
    # Pre-processing by using the transform.Compose
    # divide into batches
    # Batch size for training : 128
    # Batch size for testing : 1000
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=128, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=1000, shuffle=True, **kwargs)

    # get some random training images
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    img = torchvision.utils.make_grid(images)
    imsave(img)

    # #####################    Build your network and run   ############################
    if CNN:
        model = Net().to(device)
    elif LeNet:
        model = LN().to(device)
    elif Vgg:
        model = VGG_net().to(device)
    else:
        model = SimpleCNN_net().to(device)

    # Set the model optimizer to be ADAM with learning rate of 0.001
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Decays the learning rate of each parameter group by gamma every epoch
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    # Looping the the training function for the amount of set epoch
    # Every time the model finished training 1 epoch, test function with 10000 examples will be executed.
    for epoch in range(0, epoches + 1):
        train_model(log_interval, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    # Save train models.
    if save_model:
        if CNN:
            torch.save(model.state_dict(), "./results/CNN.pt")
        elif LeNet:
            torch.save(model.state_dict(), "./results/LeNet.pt")
        elif Vgg:
            torch.save(model.state_dict(), "./results/Vgg.pt")
        else:
            torch.save(model.state_dict(), "./results/SimpleCNN.pt")

    # Plotting loss graph after every batches of testing.
    plt.plot(model.loss_history)
    if CNN:
        plt.title('Loss factor after every batches of training (Modified CNN model)')
    elif LeNet:
        plt.title('Loss factor after every batches of training (LeNet model)')
    elif Vgg:
        plt.title('Loss factor after every batches of training (Vgg-10 model)')
    else:
        plt.title('Loss factor after every batches of training (Simple CNN model)')
    plt.ylabel('Loss Factor')
    plt.xlabel('Number of Batches')
    plt.show()
    plt.plot(model.acc_history)

    # Plotting accuracy graph after every epochs of testing.
    if CNN:
        plt.title('Percentage of accuracy after every epochs of testing (Modified CNN model)')
    elif LeNet:
        plt.title('Percentage of accuracy after every epochs of testing (LeNet model)')
    elif Vgg:
        plt.title('Percentage of accuracy after every epochs of testing (Vgg-10 model)')
    else:
        plt.title('Percentage of accuracy after every epochs of testing (Simple CNN model)')
    plt.ylabel('Percentage of Accuracy (%)')
    plt.xlabel('Number of Epochs')
    plt.xticks(np.arange(1, epoches + 1, 1.0))
    plt.show()

if __name__ == '__main__':
    main()
