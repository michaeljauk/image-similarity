import torch
import torch.nn as nn


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
        label: Similarity label (1 if genuine, 0 if imposter)
        """

        cos_sim = self.cosine_similarity(output1, output2)
        # euclidean_distance = F.pairwise_distance(output1, output2)1
        loss_contrastive = torch.mean((label) * torch.pow(cos_sim, 2) +
                                      (1-label) * torch.pow(torch.clamp(self.margin - cos_sim, min=0.0), 2))

        return loss_contrastive
