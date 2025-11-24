"""
    Author: Yassin Riyazi
    Date: 11-24-2025
    Description: Autoencoder network using CNN layers with Self-Attention mechanism.
"""
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_dim):
        super(SelfAttention,self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1)

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
        """
        m_batchsize, C, width, height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1) # B X (N) X C
        proj_key =  self.key_conv(x).view(m_batchsize, -1, width*height) # B X C x (N)
        energy =  torch.bmm(proj_query, proj_key) # B X (N) X (N)
        attention = self.softmax(energy) 
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height) # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
        
        out = self.gamma*out + x
        return out

class Encoder_Attention(nn.Module):
    """
    Encoder network for the autoencoder using CNN layers and Self-Attention.
    """
    def __init__(self, embedding_dim: int = 100) -> None:
        super(Encoder_Attention, self).__init__()
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
        self.attn = SelfAttention(128)
        self.fc = nn.Linear(128 * 13 * 13, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.attn(x)
        x = x.view(x.size(0), -1)
        embedding = self.fc(x)
        return embedding

class Decoder_Attention(nn.Module):
    """
    Decoder network for the autoencoder using CNN layers and Self-Attention.
    """
    def __init__(self, embedding_dim: int = 100) -> None:
        super(Decoder_Attention, self).__init__()
        self.fc = nn.Linear(embedding_dim, 128 * 13 * 13)
        self.attn = SelfAttention(128)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = x.view(x.size(0), 128, 13, 13)
        x = self.attn(x)
        x = self.deconv(x)
        return x

class Autoencoder_Attention(nn.Module):
    """
    Autoencoder with Self-Attention mechanism.
    """
    def __init__(self, DropOut: bool = False, embedding_dim: int = 100) -> None:
        super(Autoencoder_Attention, self).__init__()
        self.encoder = Encoder_Attention(embedding_dim)
        self.decoder = Decoder_Attention(embedding_dim)
        self.DropOut = DropOut
        self.dropout = nn.Dropout(p=0.45)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.encoder(x)
        
        if self.DropOut:
            embedding = self.dropout(embedding)
            
        recon = self.decoder(embedding)
        return recon

    def Embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

if __name__ == "__main__":
    from calflops import calculate_flops

    batch_size = 1
    input_dim = (1, 201, 201)
    embedding_dim = 1024

    model = Autoencoder_Attention(embedding_dim=embedding_dim)
    input_shape = (batch_size, *input_dim)
    
    print("Testing Autoencoder_Attention...")
    x = torch.randn(input_shape)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    with torch.inference_mode():
        flops, macs, params = calculate_flops(model=model, 
                                            input_shape=input_shape,
                                            output_as_string=True,
                                            output_precision=4)
        print("FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))
