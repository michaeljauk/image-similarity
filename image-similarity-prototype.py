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

from datasets.Caltech256Dataset import Caltech256Dataset
from datasets.TotallyLooksLikeDataset import TotallyLooksLikeDataset

# Use GPU if possible
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
# device = 'cpu'


def show_plot(iteration, loss):
    plt.plot(iteration, loss)
    plt.show()


class SiameseNetwork(nn.Module):

    def __init__(self):
        super(SiameseNetwork, self).__init__()

        # Load pre-trained VGG-19; structure: https://images.app.goo.gl/MtYeQkBbpEtGfvQE8
        self.model = torch.hub.load(
            'pytorch/vision:v0.6.0', 'vgg19', pretrained=True)

        # Remove last two FC layers
        self.model.classifier = self.model.classifier[:-6]
        # print(self.model)

    def forward_once(self, x):
        output = self.model(x)
        return output

    def forward(self, x1, x2):
        output1 = self.forward_once(x1)
        output2 = self.forward_once(x2)
        return output1, output2


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function; using cosine similarity instead of euclidian distance

    TODO: look if this is right, as the cosine similarity calculates the angel between the two vectors; what we need?

    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.cosine_similarity = nn.CosineSimilarity()
        self.margin = margin

    def forward(self, output1, output2, label):
        """
        Calculates contrastive loss using cosine similarity though.

        output1: Output 1
        output2: Output 2
        label: Similarity label (0 if genuine, 1 if imposter)
        """

        cos_sim = self.cosine_similarity(output1, output2)
        # euclidean_distance = F.pairwise_distance(output1, output2)1
        loss_contrastive = torch.mean((1-label) * torch.pow(cos_sim, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - cos_sim, min=0.0), 2))

        return loss_contrastive


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

    if debug:
        show_plot(counter, loss_history)

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


def test_same(network, crit):
    """
    To be removed
    """

    transformations = transforms.Compose([
        transforms.ToTensor()
    ])

    crit = nn.CosineSimilarity()

    img1 = cv2.imread(
        r"D:\Dokumente\Schule\Schulstufe_13\SYP-U\PyTorch\image-similarity-prototype\cube_1.jpg")
    img1 = cv2.resize(img1, (224, 224))
    img1 = transformations(img1)
    img1 = img1.unsqueeze(0)
    img1 = img1.to(device)

    img2 = cv2.imread(
        r"D:\Dokumente\Schule\Schulstufe_13\SYP-U\PyTorch\image-similarity-prototype\cube_1.jpg")
    img2 = cv2.resize(img2, (224, 224))
    img2 = transformations(img2)
    img2 = img2.unsqueeze(0)
    img2 = img2.to(device)

    x1, x2 = network(img1, img2)
    distance = crit(x1, x2)
    print(distance)


def test_not_same(network, crit):
    """
    To be removed
    """
    transformations = transforms.Compose([
        transforms.ToTensor()
    ])

    img1 = cv2.imread(
        r"D:\Dokumente\Schule\Schulstufe_13\SYP-U\PyTorch\image-similarity-prototype\airpods.jpg")
    img1 = cv2.resize(img1, (224, 224))
    img1 = transformations(img1)
    img1 = img1.unsqueeze(0)
    img1 = img1.to(device)

    img2 = cv2.imread(
        r"D:\Dokumente\Schule\Schulstufe_13\SYP-U\PyTorch\image-similarity-prototype\cube_2.jpg")
    img2 = cv2.resize(img2, (224, 224))
    img2 = transformations(img2)
    img2 = img2.unsqueeze(0)
    img2 = img2.to(device)

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

    # Initialize dataset
    transformations = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # # !!! Has to be changed
    # root_dir = "C:\\Users\\Simon\\Documents\\Schule\\5. Schuljahr\\Image Similarity Detection\\256_ObjectCategories\\256_ObjectCategories"
    # dataset = Caltech256Dataset(
    #     root_dir=root_dir, transform=transformations, should_invert=False)

    # !!! Has to be changed
    root_dir = "C:\\Users\\Simon\\Documents\\Schule\\5. Schuljahr\\Image Similarity Detection\\TLL"
    dataset = TotallyLooksLikeDataset(
        root_dir=root_dir, transform=transformations)

    # Initialize Train dataloader
    train_dataloader = DataLoader(
        dataset, shuffle=True, num_workers=0, batch_size=1)

    print("Start training")

    train(net, criterion, optimizer, epochs, train_dataloader)

    PATH = "./model"

    save_model(net, PATH)

    # test_same(net, criterion)


def main():
    do_initial_training()


if __name__ == "__main__":
    main()
