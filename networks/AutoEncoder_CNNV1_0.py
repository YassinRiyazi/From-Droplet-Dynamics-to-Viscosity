"""
    Author: Yassin Riyazi
    Date: 04-08-2025

    Autoencoder network using CNN layers.

    Learned:
        Dilation in CNN: https://www.geeksforgeeks.org/machine-learning/dilated-convolution/
"""
import torch
import torch.nn as nn
# import torch.nn.functional as F
# from typing import Union, Optional, Tuple

class Encoder_CNN(nn.Module):
    """
    Encoder network for the autoencoder using CNN layers.
    This network compresses the input image into a lower-dimensional embedding.
    """
    def __init__(self,
                 embedding_dim: int = 100) -> None:
        """
        Initializes the encoder network.
        Args:
            embedding_dim (int): Dimension of the embedding space
        Returns:
            None: Initializes the encoder with convolutional layers followed by a fully connected layer.
        """
        super(Encoder_CNN, self).__init__() # type: ignore
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # 1x200x200 → 16x100x100
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # 32x50x50
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # 64x25x25
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # 128x13x13
            nn.ReLU(),
        )
        self.fc = nn.Linear(128 * 13 * 13, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 200, 200)   
        Returns:
            torch.Tensor: Embedding tensor of shape (batch_size, embedding_dim)
        """
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        embedding = self.fc(x)
        return embedding

class Decoder_CNN(nn.Module):
    """
    Decoder network for the autoencoder using CNN layers.
    This network reconstructs the image from the lower-dimensional embedding.
    """
    def __init__(self,
                 embedding_dim: int = 100) -> None:
        """
        Initializes the decoder network.
        Args:
            embedding_dim (int): Dimension of the embedding space
        Returns:
            None: Initializes the decoder with a fully connected layer followed by transposed convolutional layers.
        """
        super(Decoder_CNN, self).__init__() # type: ignore
        self.fc = nn.Linear(embedding_dim, 128 * 13 * 13)
        self.deconv = nn.Sequential(
                                        nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), # 64x25x25
                                        nn.ReLU(),
                                        nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=0),  # 32x50x50
                                        nn.ReLU(),
                                        nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=0), # 16x100x100
                                        nn.ReLU(),
                                        nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=0),   # 1x200x200
                                        nn.Sigmoid()
                                    )

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the decoder.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, embedding_dim)
        Returns:
            torch.Tensor: Reconstructed tensor of shape (batch_size, 1, 200, 200)
        """
        x = self.fc(x)
        x = x.view(x.size(0), 128, 13, 13)
        x = self.deconv(x)
        return x

class Autoencoder_CNN(nn.Module):
    def __init__(self,DropOut: bool = False ,
                 embedding_dim: int = 100) -> None:
        """
        Initializes the autoencoder network.
        Args:
            embedding_dim (int): Dimension of the embedding space
        Returns:
            None: Initializes the encoder and decoder networks.
        """

        super(Autoencoder_CNN, self).__init__() # type: ignore
        # self.resizer = ResizeTo200()
        self.encoder = Encoder_CNN(embedding_dim)
        self.decoder = Decoder_CNN(embedding_dim)

        self.DropOut = DropOut
        # Change dropout probability after initialization
        # print("Before:", model.dropout.p)
        # model.dropout.p = 0.2
        # print("After:", model.dropout.p)
        self.dropout = nn.Dropout(p=0.45)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the autoencoder.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 200, 200)
        Returns:
            torch.Tensor: Reconstructed tensor of shape (batch_size, 1, 200, 200)
        """
        # x = self.resizer(x)
        embedding = self.encoder(x)
        
        if self.DropOut:
            embedding = self.dropout(embedding)
            
        recon = self.decoder(embedding)
        return recon#, embedding
    
    def Embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts the embedding from the input tensor.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 200, 200)
        Returns:
            torch.Tensor: Embedding tensor of shape (batch_size, embedding_dim)
        """
        return self.encoder(x)



if __name__ == "__main__":
    # conv = nn.Sequential(
    #         # Input: (1, 130, 1280)
    #         nn.Conv2d(1, 16, kernel_size=(3,9), stride=2, padding=1),   # -> (16, 65, 640)
    #         nn.ReLU(),
    #         nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # -> (32, 33, 320)
    #         nn.ReLU(),
    #         nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # -> (64, 17, 160)
    #         nn.ReLU(),
    #         nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # -> (128, 9, 80)
    #         nn.ReLU(),
    #         nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),# -> (256, 5, 40)
    #         nn.ReLU(),
    #         nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),# -> (512, 3, 20)
    #         nn.ReLU(),
    #     )
    
    # randData = torch.randn(1, 1, 130, 1280)
    # output = conv(randData)
    # print(output.shape)  

    from calflops import calculate_flops

    batch_size = 1
    Sequence_length = 10
    LSTMEmbdSize = 201*201#4048  # input dimension

    model = LSTMModel(  LSTMEmbdSize = LSTMEmbdSize,  # input dimension
                        hidden_dim = 128 ,  # hidden dimension
                        num_layers = 2   ,  # number of LSTM layers
                        dropout = 0.2 )
    input_shape = (batch_size, Sequence_length, LSTMEmbdSize)
    with torch.inference_mode():
        flops, macs, params = calculate_flops(model=model, 
                                            input_shape=input_shape,
                                            output_as_string=True,
                                            output_precision=4)
        print("Alexnet FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))