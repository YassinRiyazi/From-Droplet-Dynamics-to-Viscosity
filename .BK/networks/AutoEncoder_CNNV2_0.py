"""
    Author: Yassin Riyazi
    Date: 14-09-2025

    Autoencoder network using CNN layers.
    Consideration:
        [] Images are grey scale (1 channel).
        [] Gap before latent space
        [] Height is always 130 pixels for drop come back later when patch embedding is implemented.
          
    Architecture:
        Encoder:
            Three down-sampling ResBlocks. Use stride 2 conv for down-sampling.
            Inception Block
            Squeeze and excitation block
            GAP

        Decoder:
            Decoder: MLP -> reshape -> three up-sampling ResBlocks. Use bilinear upsample + conv.


    TODO:
        A [] Do genetic algorithm to find best architecture.
        
        [] Input landscape and cropped images. Takes in a width scale down argument to reduce the input size.
        Block design:
            [] Test resnet and skip connection
            [] Test leaky relu
            [] latent expressiveness: Self-attention block at the bottleneck (like in Vision Transformers or modern VAEs).
            [] multiple conv blocks in each block [VGG style]
            [] Add skip connection [ResNet style]
            [] Squeeze-and-Excitation (SE) / Channel gating
                Lightweight channel-wise attention that improves feature selectivity in encoder/decoder.

        Latent:
            Orthogonality constraints
                Force latent subspaces (e.g., style vs. content) to be orthogonal using cosine penalties.
                Helps to keep different features in separate slots.

        LOSS:
            [] Loss = L1 + perceptual (VGG) on the final output. 
            Optionally add small latent dropout and a contrastive loss on z to improve invariance.
            Variational methods
                [] β-VAE: Add a β coefficient (>1) on the KL divergence term of the VAE loss:
                    L=Recon(x,x^)+β DKL(q(z∣x) ∣∣ p(z))
                    Larger β forces latents to be more independent (closer to isotropic Gaussian), which encourages disentanglement, though at the cost of reconstruction fidelity.
                [] FactorVAE / β-TCVAE: Explicitly penalize total correlation (a measure of statistical dependence between latent dimensions). This encourages latents to be independent.
                Start with β-VAE (β ~ 4–10) or FactorVAE if you want unsupervised disentanglement.

    Learned:
        Dilation in CNN: https://www.geeksforgeeks.org/machine-learning/dilated-convolution/
"""

import  torch
import  torch.nn                as      nn
import  torch.nn.functional     as      F
from    typing                  import  Tuple

class BasicBlock(nn.Module):
    """ResNet Basic Block (for ResNet-18/34 style)."""
    expansion = 1

    def __init__(self, in_channels:int, out_channels:int, stride:int=1, pool:bool=True):
        super().__init__() # type: ignore
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        # Adding pooling to reduce dimensions
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if pool else nn.Identity()
        # self.pool = nn.Identity()  # don’t pool if using stride=2

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = F.relu(out)
        # vv = out.shape
        out = self.pool(out)
        # print("BasicBlock output shape:", vv, "\t After MaxPool shape:", out.shape)
        return out

class ResNetLayer(nn.Module):
    """Stack multiple ResNet blocks sequentially."""
    def __init__(self, in_channels:int, num_blocks:int, stride:int=1, scaleFactor: int = 2, poolIndex: int = 3):
        super().__init__() # type: ignore
        layers: list[nn.Module] = []
        current_channels = in_channels
        pools_per_layer = max(1, num_blocks // poolIndex)  # E.g., pool every ~3 blocks; adjust to 3-4 total pools
        pool_count = 0
        for i in range(num_blocks):
            pool_this_block = (pool_count < pools_per_layer) and (i % poolIndex == 0)  # Pool sparingly
            if pool_this_block:
                stride_block = 2
                pool_count += 1
            else:
                stride_block = 1
            out_channels = current_channels * scaleFactor if i < num_blocks - 1 else current_channels
            layers.append(BasicBlock(current_channels, out_channels, stride=stride_block, pool=pool_this_block))
            current_channels = out_channels

        self.layer = nn.Sequential(*layers)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.layer(x)
        
class SqueezeExcite(nn.Module):
    """Squeeze-and-Excitation block."""
    def __init__(self, channels:int, reduction:int=16):
        super().__init__() # type: ignore
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # Squeeze: global average pooling
        w = F.adaptive_avg_pool2d(x, 1)
        # Excitation: FC → ReLU → FC → Sigmoid
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w  # channel-wise scaling

class GlobalAveragePooling(nn.Module):
    """Global Average Pooling (GAP)."""
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(x, (1, 1))  # shape (N, C, 1, 1)

class Encoder_CNN(nn.Module):
    """
    Encoder network for the autoencoder using CNN layers.
    This network compresses the input image into a lower-dimensional embedding.
    """
    def __init__(self,
                #  embedding_dim: int = 100,
                 In_channel: int = 1,
                 num_blocks: int = 7,
                 scaleFactor: int = 3) -> None:
        """
        Initializes the encoder network.
        Args:
            In_channel (int): Number of input channels
            num_blocks (int): Number of ResNet blocks in the encoder
        Returns:
            None: Initializes the encoder with convolutional layers followed by a fully connected layer.
        """
        super(Encoder_CNN, self).__init__() # type: ignore

        self.layer = ResNetLayer(in_channels=In_channel, num_blocks=num_blocks, stride=2, scaleFactor=scaleFactor)    
        # Add SE
        self.se = SqueezeExcite(channels=In_channel*scaleFactor**(num_blocks-1))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 200, 200)   
        Returns:
            torch.Tensor: Embedding tensor of shape (batch_size, embedding_dim)
        """
        x = self.layer(x)
        # print("ResNetLayer:", x.shape)

        x = self.se(x)
        # print("After SE:", x.shape)
        
        # x = self.gap(x)
        # print("After GAP:", x.shape)
        return x # x.view(x.size(0), -1)  # Flatten to (batch_size, channels)
    
class DecoderBlock(nn.Module):
    """Residual decoder block with up-sampling (mirror of BasicBlock)."""
    def __init__(self, in_channels: int, out_channels: int, upsample: bool = True):
        super().__init__() # type: ignore
        self.upsample = upsample

        # If upsample=True, first upsample then conv
        if upsample:
            self.up = nn.Upsample(scale_factor=2, mode="nearest")
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                                   padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                                   padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        if self.upsample:
            x = self.up(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual if not self.upsample else self.up(residual))
        out = F.relu(out)
        return out

class Decoder_CNN(nn.Module):
    """
    Decoder network mirroring the ResNet-style encoder.
    """
    def __init__(self, In_channel: int = 1, 
                 num_blocks: int = 7,
                 inSize:Tuple[int,int] = (4,20),
                 scaleFactor: int = 3) -> None:
        super(Decoder_CNN, self).__init__() # type: ignore

        # # Latent vector → feature map
        # self.fc = nn.Linear(1, inSize[0] * inSize[1])  # starting small spatial map

        # Mirror of encoder blocks, progressively reducing channels
        layers: list[nn.Module] = []
        in_channels = In_channel*scaleFactor**(num_blocks-1)
        self.InputShape = (int(in_channels), int(inSize[0]), int(inSize[1]))  # (C, H, W)

        for _ in range(num_blocks - 1-2, 0, -1):
            # print("Decoder in_channels:", in_channels, "out_channels:", in_channels // scaleFactor)
            layers.append(DecoderBlock(in_channels, in_channels // scaleFactor, upsample=True))
            in_channels = in_channels // scaleFactor


        self.layers = nn.Sequential(*layers)

        # Final reconstruction conv
        self.final_conv = nn.Conv2d(in_channels, In_channel, kernel_size=3, stride=1, padding=1)
        self.out_act = nn.Sigmoid()

        self.resize1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.resize2 = nn.AvgPool2d(kernel_size=2, stride=2)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = self.fc(x)
        # # print("After FC:", x.shape)
        x = x.view(x.size(0), *self.InputShape)  # reshape latent into feature map
        # print("After reshape:", x.shape)
        x = self.layers(x)
        x = self.final_conv(x)
        # x = self.resize1()
        x = self.out_act(x) # This was off before
        return x

class Autoencoder_CNN(nn.Module):
    def __init__(self,
                 In_channel: int = 1,
                 num_blocks: int = 9,
                 Image: Tuple[int, int] = (256, 256),
                 scaleFactor: int = 2) -> None:
        """
        Initializes the autoencoder network.
        Args:
            In_channel (int): Number of input channels
            num_blocks (int): Number of ResNet blocks in the encoder
        Returns:
            None: Initializes the encoder and decoder networks.
        """

        super(Autoencoder_CNN, self).__init__() # type: ignore
        self.encoder    = Encoder_CNN(In_channel=In_channel, num_blocks=num_blocks, scaleFactor=scaleFactor)

        sample_input    = torch.randn(1, In_channel, *Image)  # Example input to determine size
        self.inSize     = self.encoder.forward(sample_input)
        self.inSize     = self.inSize.shape[2:]
        # print("Encoder output shape (C, H, W):", inSize.shape[2:])

        self.decoder    = Decoder_CNN(In_channel=In_channel, num_blocks=num_blocks, inSize=self.inSize, scaleFactor=scaleFactor)

        # adding dropout
        self.DropOut = False
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
        x = self.encoder(x)
    

         # Apply dropout if enabled
         # Dropout is typically used during training to prevent overfitting.
         # During evaluation, dropout is disabled.
        if self.DropOut:
            x = self.dropout(x)

        x = self.decoder(x)
        return x
    
    def forwardD(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        print("Encoder output shape (C, H, W):", x.shape)
        x = self.decoder(x)
        print("Decoder output shape (C, H, W):", x.shape)
        return x
    
    def Embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts the embedding from the input tensor.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 200, 200)
        Returns:
            torch.Tensor: Embedding tensor of shape (batch_size, embedding_dim)
        """
        x = self.encoder(x)
        # x = x.view(x.size(0), -1)  # Flatten the output
        # x = x.contiguous()
        return x

if __name__ == "__main__":
    In_channel = 1
    num_blocks = 7
    with torch.no_grad():
        
        # x = torch.randn(10, In_channel, 256, 256)
        # x = Autoencoder_CNN().forwardD(x)
        # # print(x.shape)

        # Insize = (512, 1024)
        # x = torch.randn(10, In_channel, *Insize)
        # x = Autoencoder_CNN(num_blocks=9, Image=Insize).forwardD(x)
        # print(x.shape)

        Insize = (256, 1024)
        x = torch.randn(10, In_channel, *Insize)
        x = Autoencoder_CNN(num_blocks=9, Image=Insize).forwardD(x)
        print(x.shape)

