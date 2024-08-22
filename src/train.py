import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from config import Config
from dataset import CustomDataset, CustomSampler
from model import Decoder
from trainer import Trainer
from utils import loss_fn


def run():
    model = Decoder(Config).to(Config.device)
    optim = torch.optim.AdamW(
        model.parameters(), lr=Config.lr, weight_decay=Config.wd)
    trainer = Trainer(model, optim, loss_fn)
    val_min = np.Inf
    path = './'
    all_files = os.listdir()
    all_files = [os.path.join(path, x)
                 for x in all_files if x.endswith('.txt')]
    file = all_files[0]
    dataset = CustomDataset(file, Config)
    loader = DataLoader(dataset, sampler=CustomSampler(
        dataset, Config.block_size))

    for epoch_ in range(Config.epochs):
        print(f"{'='*50} EPOCH: {epoch_+1}/{Config.epochs} {'='*50}")

        trainer.train_one_epoch(loader)
        loss, out_model = trainer.val_one_epoch(loader)

        if loss <= val_min:
            print(
                "Validation_loss decreased ({:.4f}-->{:.4f}). Saving the model...".format(val_min, loss))
            torch.save(out_model.state_dict(),
                       f"Epoch= {epoch_+1}_{Config.model_name}.pt")
            val_min = loss


if __name__ == "__main__":
    run()
