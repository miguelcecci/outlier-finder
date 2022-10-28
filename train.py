import torch
import torchvision.datasets as datasets
from tqdm import tqdm
from torch import nn, optim
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd 

#  locals
from variational_autoencoder import VariationalAutoEncoder
from dataset_loader import DatasetLoader

class Outliers:

    def __init__(self, data_path):

        self.dataset = DatasetLoader(data_path)
        self.loader = DataLoader(
            self.dataset, batch_size=20, shuffle=True
        )

        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.INPUT_DIM = 784*3
        self.H_DIM = 200
        self.Z_DIM = 20
        self.NUM_EPOCHS = 14
        self.BATCH_SIZE = 32
        self.LR_RATE = 3e-4

        return

    def find(self):

        """
        """

        model = VariationalAutoEncoder(self.INPUT_DIM, self.H_DIM, self.Z_DIM).to(self.DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=self.LR_RATE)
        loss_fn = nn.BCELoss(reduction="sum")


        result_names = []
        result = []

        for epoch in range(self.NUM_EPOCHS):
            loop = tqdm(enumerate(self.loader))
            for i, (x, names) in loop:
                # forward pass

                x = x.to(self.DEVICE).view(x.shape[0], self.INPUT_DIM)

                x_reconstructed, mu, sigma = model(x)

                reconstruction_error = (x_reconstructed - x)

                reconstruction_loss = loss_fn(x_reconstructed, x)
                kl_div = -torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))

                if epoch == self.NUM_EPOCHS-1:
                    result.append(reconstruction_error.cpu().detach().numpy())
                    result_names.append(names)

                #backprop
                loss = reconstruction_loss + kl_div
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loop.set_postfix(loss=loss.item())

        result = list(map(lambda x: sum(x**2), np.concatenate(result, axis=0)))
        result_names = np.concatenate(result_names, axis=0)

        final_list = list(zip(result, result_names))

        return pd.DataFrame(final_list).sort_values(0, ascending=False)

#  return final_list
