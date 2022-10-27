import torch
import torch.nn.functional as F
from torch import nn

class VariationalAutoEncoder(nn.Module):
    

    # input image ->
    # hidden dimension ->
    # mean, std ->
    # parametrization trick ->
    # decoder ->
    # output image ->


    def __init__(self, input_dim, hidden_dimension=200, z_dimension=20):
        super().__init__()

        #encoder
        self.img_2hidden = nn.Linear(input_dim, hidden_dimension)

        self.hid_2mu = nn.Linear(hidden_dimension, z_dimension)
        self.hid_2sigma = nn.Linear(hidden_dimension, z_dimension)

        #decoder
        self.z_2hid = nn.Linear(z_dimension, hidden_dimension)
        self.hid_2img = nn.Linear(hidden_dimension, input_dim)

        self.relu = nn.ReLU()


    def encode(self, x):
        h = self.relu(self.img_2hidden(x))
        mu, sigma = self.hid_2mu(h), self.hid_2sigma(h)

        return mu, sigma

    def decode(self, z):

        h = self.relu(self.z_2hid(z))

        return torch.sigmoid(self.hid_2img(h))


    def forward(self, x):
        mu, sigma = self.encode(x)

        #parametrization trick
        epsilon = torch.randn_like(sigma)
        z_reparametrized = mu + sigma*epsilon

        x_reconstructed = self.decode(z_reparametrized)


        return x_reconstructed, mu, sigma
        


if __name__ == "__main__":
    x = torch.randn(4, 784)

    vae = VariationalAutoEncoder(input_dim=784)

    print(vae(x))
