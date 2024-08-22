import torch.nn as nn
from config import Config


def loss_fn(self, inputs, targets):
    inputs = inputs.view(-1, inputs.size(-1)).to(Config.device)
    targets = targets.view(-1).to(Config.device)
    loss = nn.functional.cross_entropy(inputs, targets).to(Config.device)
    return loss

