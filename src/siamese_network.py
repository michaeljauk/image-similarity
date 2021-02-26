import torch
import torch.nn as nn

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
