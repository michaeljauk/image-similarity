import csv
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

import config
from src.siamese_network import SiameseNetwork
from src.contrastive_loss import ContrastiveLoss
import src.dataset as DS

# Use GPU if possible
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'


def show_plot(iteration, loss):
    plt.plot(iteration, loss)
    plt.show()


def hyperparameter_optimization(network, criterion, train_dataloader, lr_list, epoch_list, debug=True):
    with open('parameters.csv', 'w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(["Learning rate", "Epochs", "Accuracy"])
        for lr in lr_list:
            print("LR:",  lr)
            for epochs in epoch_list:
                print("Epoch", epochs)
                optimizer = optim.Adam(network.parameters(), lr=lr)
                train(network, criterion, optimizer, epochs, train_dataloader)
                print("train")
                accuracy = test(network, train_dataloader)
                print("test")
                writer.writerow([lr, epochs, accuracy.numpy()[0]])


def train(network, criterion, optimizer, epochs, train_dataloader, debug=True):
    counter = []
    loss_history = []
    iteration_number = 0

    for epoch in range(epochs):
        for i, data in enumerate(train_dataloader, 0):
            # Load data and move it to gpu if possible
            img0, img1, label = data
            img0 = img0.to(device)
            img1 = img1.to(device)
            label = label.to(device)

            # Clear out gradients
            optimizer.zero_grad()

            # Forward both images through network
            output1, output2 = network(img0, img1)

            # Calcuate contrastive loss and calc gradient
            loss_contrastive = criterion(output1, output2, label)
            loss_contrastive.backward()

            # Update parameters
            optimizer.step()

            print(i)

            if debug and i % 10 == 0:
                print("Epoch number {}\n Current loss {}\n".format(
                    epoch, loss_contrastive.data))
                iteration_number += 10
                counter.append(iteration_number)
                loss_history.append(loss_contrastive.data)

            del img0, img1, label

    # if debug:
        #show_plot(counter, loss_history)

# test accuracy of the network
# 0 -> lowest accuracy
# 1 -> highest accuracy


def test(network, test_dataloader):
    similarity = nn.CosineSimilarity()
    with torch.no_grad():
        sum_accuracy = 0
        for i, data in enumerate(test_dataloader):
            img0, img1, label = data
            x1, x2 = network(img0, img1)
            result = similarity(x1, x2)
            if label == 0:
                sum_accuracy += 1 - result
            else:
                sum_accuracy += result
        return sum_accuracy / len(test_dataloader)


def save_model(network, path):
    torch.save(network.state_dict(), path)


def load_model(path):
    network = SiameseNetwork()
    network = network.to(device)
    network.load_state_dict(torch.load(path))
    return network


def test_similarity(network, path1, path2):
    transformations = transforms.Compose([
        transforms.ToTensor()
    ])

    crit = nn.CosineSimilarity()

    img1 = cv2.imread(path1)
    img1 = cv2.resize(img1, (224, 224))
    img1 = transformations(img1)
    img1 = img1.unsqueeze(0)

    img2 = cv2.imread(path2)
    img2 = cv2.resize(img2, (224, 224))
    img2 = transformations(img2)
    img2 = img2.unsqueeze(0)

    x1, x2 = network(img1, img2)
    distance = crit(x1, x2)
    print(distance)


def do_initial_training():
    # Declare network and move to gpu if possible
    net = SiameseNetwork()
    net.to(device)

    # Loss function
    criterion = ContrastiveLoss()

    # Learning rate
    lr = 0.0005

    # Optimizer
    # In this case: Adam
    # TODO: Look if this is right
    optimizer = optim.Adam(net.parameters(), lr=lr)

    # epochs to train
    epochs = 2

    dataset = DS.get_dataset('default')

    # Initialize Train dataloader
    train_dataloader = DataLoader(
        dataset, shuffle=True, num_workers=0, batch_size=1)

    print("Start training")

    train(net, criterion, optimizer, epochs, train_dataloader)

    PATH = config.model_path

    save_model(net, PATH)

    # test_same(net, criterion)


def start_hyperparameter_optimization():
        # Declare network and move to gpu if possible
    net = SiameseNetwork()
    net.to(device)

    # Loss function
    criterion = ContrastiveLoss()

    # Learning rate
    #lr = 0.0005

    # Optimizer
    # In this case: Adam
    # TODO: Look if this is right
    #optimizer = optim.Adam(net.parameters(), lr=lr)

    # epochs to train
    #epochs = 2

    # Get OIDv4 Dataset
    # TODO: Test different configs
    # TODO: Test if the tool even works, sometimes there are weird pictures inside the same category
    # At least two classes need to be downloaded
    dataset = DS.get_dataset('caltech_256')

    # Initialize Train dataloader
    train_dataloader = DataLoader(
        dataset, shuffle=True, num_workers=0, batch_size=1)

    print("Start training")

    epochs = [1, 2, 3]
    learningrates = [1, 4, 10]

    hyperparameter_optimization(net, criterion, train_dataloader, learningrates, epochs)

    PATH = config.model_path

    save_model(net, PATH)


def main():
    # do_initial_training()
    # start_hyperparameter_optimization()
    net = SiameseNetwork()
    path1 = r"C:\Users\Simon\Documents\Temporary\5.jpg"
    path2 = r"C:\Users\Simon\Documents\Temporary\6.jpg"
    test_similarity(net, path1, path2)


if __name__ == "__main__":
    main()
