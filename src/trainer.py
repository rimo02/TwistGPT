import torch
from tqdm import tqdm


class Trainer:
    def __init__(self, model, optim, loss_fn, wandb=False):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optim

    def train_one_epoch(self, train_dl):
        loss_ = 0
        for input, target in tqdm(train_dl, total=len(train_dl)):
            out = self.model(input)
            loss = self.loss_fn(self, out, target)
            loss_ += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        print('Train Loss: ', loss_/len(train_dl))

    @torch.no_grad()
    def val_one_epoch(self, val_dl):
        loss_ = 0
        for input, target in tqdm(val_dl, total=len(val_dl)):
            out = self.model(input)
            loss = self.loss_fn(self, out, target)
            loss_ += loss.item()
            return loss_, self.model
        print('Valid Loss: ', loss_/len(val_dl))
