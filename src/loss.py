import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss
import torch.nn.functional as F




class FocalLoss( BCEWithLogitsLoss ):
    """ Implementation of Focal Loss

        From:
        Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár,"Focal Loss for Dense Object Detection"
        https://arxiv.org/abs/1708.02002

    """

    def __init__( self, gamma ):
        super().__init__(reduction='none')
        self.gamma = gamma

    # __init__

    def forward( self, input, target ):
        BCELoss = super().forward( input, target )
        pt = torch.exp( -BCELoss )

        if self.gamma == 0:
            loss = BCELoss
        else:
            loss = (1 - pt) ** self.gamma * BCELoss

        loss = loss.mean()

        return loss

    # forward

# class: FocalLoss
