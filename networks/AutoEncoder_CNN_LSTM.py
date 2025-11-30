"""
    Author:         Yassin Riyazi
    Date:           04-08-2025
    Description:    Train an embedding-based LSTM for time series data.

    TODO:
        -
"""
import  os
import  torch 
import  torch.nn                as      nn
from    typing                  import  Union

class LSTMModel(nn.Module):
    def __init__(self, LSTMEmbdSize: int, hidden_dim: int, num_layers: int, dropout: float, device: torch.device|None = None) -> None:
        """
        Initializes the LSTM model.
        Args:
            LSTMEmbdSize (int): Dimension of the input space
            hidden_dim (int): Dimension of the hidden state in LSTM
            num_layers (int): Number of LSTM layers
            dropout (float): Dropout rate for LSTM layers
        Returns:
            None: Initializes the LSTM layer and a fully connected layer.

        """
        super(LSTMModel, self).__init__() # type: ignore
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        
        self.lstm   = nn.LSTM(LSTMEmbdSize, hidden_dim, num_layers, dropout=dropout, batch_first=True, device=self.device)
        self.fc     = nn.Linear(hidden_dim, 1, device=self.device)  # output layer
        self.h      = None
        self.c      = None

    def forward(self, x: torch.Tensor,
                ) -> torch.Tensor:
        """
        Forward pass through the LSTM model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_dim)
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1)
        """
        
        
        if self.h is None or self.c is None:
            self.reset_states(x)
        
        out, _ = self.lstm(x, (self.h, self.c))
        # out = out[:, -1, :]  # take the last hidden state
        out = torch.mean(out, dim=1)  # average across time steps
        out = self.fc(out)
        out = torch.sigmoid(out)  # Add sigmoid to constrain to 0-1
        return out
    
    def reset_states(self,x: torch.Tensor) -> None:
        """
        Resets the hidden and cell states of the LSTM.
        Args:
            x (torch.Tensor): Input tensor to determine batch size  
        Returns:
            None: Resets the states to zero.
        """
        self.h = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size, device=self.device)
        self.c = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size, device=self.device)

if __name__ == "__main__":
    from calflops import calculate_flops # type: ignore

    batch_size = 1
    Sequence_length = 10
    LSTMEmbdSize = 1201*201#4048  # input dimension

    model = LSTMModel(  LSTMEmbdSize = LSTMEmbdSize,  # input dimension
                        hidden_dim = 128 ,  # hidden dimension
                        num_layers = 2   ,  # number of LSTM layers
                        dropout = 0.2,
                        device = torch.device("cpu"))
    input_shape = (batch_size, Sequence_length, LSTMEmbdSize)
    with torch.inference_mode():
        flops, macs, params = calculate_flops(model=model, # type: ignore
                                            input_shape=input_shape,
                                            output_as_string=True,
                                            output_precision=4)
        print("Alexnet FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))

class Encoder_LSTM(torch.nn.Module):
    def __init__(self,
                 address_autoencoder:Union[str,None],
                 proj_dim:int   ,
                 LSTMEmbdSize:int   = 2048,  # input dimension
                 hidden_dim:int  = 128 ,  # hidden dimension
                 num_layers:int  = 2   ,  # number of LSTM layers
                 dropout:float   = 0.2 ,  # dropout rate
                 Autoencoder_CNN: torch.nn.Module|None = None,
                 S4_size:int|None = None) -> None:
        """
        Initializes the LSTM encoder.
        Args:
            address_autoencoder (str): Path to the pre-trained autoencoder model
            input_dim (int): Dimension of the input space
            hidden_dim (int): Dimension of the hidden state in LSTM
            num_layers (int): Number of LSTM layers
            dropout (float): Dropout rate for LSTM layers
            sequence_length (int): Length of the input sequences

        Returns:
            None: Initializes the LSTM layer.
        """
        super(Encoder_LSTM, self).__init__() # type: ignore
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.autoencoder = address_autoencoder
        self.proj_dim = proj_dim
        self.Properties: dict[str, int|float] = {
                            "input_dim": proj_dim,
                            "hidden_dim": hidden_dim,
                            "num_layers": num_layers,
                            "dropout": dropout
                        }
        if self.autoencoder:
            self.load_autoencoder(address_autoencoder,Autoencoder_CNN)
        
        if S4_size is not None:
            # FIXME: Adjust input dimension to include S4 features
            # self.Properties["input_dim"] += S4_size
            LSTMEmbdSize += S4_size

        self.LN = nn.LayerNorm(LSTMEmbdSize, device=self.device)  # Layer normalization layer

        self.lstm = LSTMModel(LSTMEmbdSize=LSTMEmbdSize,
                              hidden_dim=hidden_dim,
                              num_layers=num_layers,
                              dropout=dropout).to(self.device)
        # self.proj   = nn.Linear(proj_dim, LSTMEmbdSize, device=self.device)
        # self.relu   = nn.ReLU()
        return None
    
    def _encoder(self,x:torch.Tensor) -> torch.Tensor:
        """
            For DF dataset there is no need to use the autoencoder.
        """
        with torch.no_grad():
            _shape = x.shape                                                # (batch_size, seq_length, channels, height, width)
            x = x.view(-1,1, _shape[3], _shape[4])                          # (batch_size * seq_length, channels, height, width)
            # breakpoint()
            x = self.autoencoder.Embedding(x)                               # (batch_size * seq_length, input_dim)
            x = x.view(_shape[0], _shape[1], self.Properties["input_dim"])  # (batch_size, seq_length, input_dim)
            x = x.contiguous()
        return x

    def forward(self,
                x: torch.Tensor,
                x_additional: torch.Tensor|None = None) -> torch.Tensor:
        """
        Forward pass through the encoder.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_dim)
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, hidden_dim)
        
        Note:
            SROF features (input[:,:,2:]) are pre-normalized with global mean/std from dataset.
            LayerNorm applies per-sample normalization, which may interfere with learned global distributions.
            Consider removing LayerNorm or applying it only during initial training phases.
        """
        if self.autoencoder:
            x = self._encoder(x)

        if x_additional is not None:
            x = torch.cat((x, x_additional), dim=-1)  # Concatenate along the feature dimension

        # x = self.relu(self.proj(x))

        # OPTION 1 (RECOMMENDED): Skip LayerNorm since data is already globally normalized
        # Features are pre-normalized in dataset with consistent mean/std across splits
        # x = self.LN(x)  # DISABLED: data already normalized
        
        # # OPTION 2 (ALTERNATIVE): Keep LayerNorm but be aware it re-normalizes per sample
        # # This can be beneficial for stability but loses global distribution info
        # x = self.LN(x)  # Per-sample normalization (mean=0, std=1 within each sample)
        
        out = self.lstm(x)  # (batch_size, 1)
        return out.squeeze(1)  # (batch_size,)
    
    def load_autoencoder(self,
                         address_autoencoder: str|None,
                         Autoencoder_CNN: torch.nn.Module|None) -> None:
        """
        Load the autoencoder model from a file.
        Args:
            address_autoencoder (str): Path to the autoencoder model file
        Returns:
            None: Loads the autoencoder model.
        """
        if Autoencoder_CNN is None:
            raise ValueError("Autoencoder_CNN model must be provided to load the autoencoder.")
        if address_autoencoder is None or os.path.isfile(address_autoencoder) is False:
            raise ValueError(f"Address of the autoencoder model must be correctly provided. {address_autoencoder} is invalid.")
        
        self.autoencoder = Autoencoder_CNN(embedding_dim = self.proj_dim).to(self.device)
        self.autoencoder.eval()
        
        # Load state dict and handle potential _orig_mod prefix from torch.compile
        state_dict = torch.load(address_autoencoder, map_location=self.device)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('_orig_mod.'):
                new_state_dict[k[10:]] = v
            else:
                new_state_dict[k] = v
                
        self.autoencoder.load_state_dict(new_state_dict)
        
        return None