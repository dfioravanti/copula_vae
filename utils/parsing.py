
import torch.nn as nn


def choose_loss_function(loss='L2'):

    if loss == 'L1':
        return nn.L1Loss()
    elif loss == 'L2':
        return nn.MSELoss()
    elif loss == 'BCE':
        return nn.BCELoss()
    else:
        raise NotImplementedError("Unknown loss function")
