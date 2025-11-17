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
if __name__ == "__main__":
    from AutoEncoder_CNNV2_0 import  Autoencoder_CNN as Autoencoder_CNNV2
    from AutoEncoder_CNNV1_0 import  Autoencoder_CNN as Autoencoder_CNNV1
else:
    from    .AutoEncoder_CNNV2_0    import  Autoencoder_CNN as Autoencoder_CNNV2
    from    .AutoEncoder_CNNV1_0    import  Autoencoder_CNN as Autoencoder_CNNV1
import  glob

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
                 sequence_length:int = 5,
                 Autoencoder_CNN: torch.nn.modules = None) -> None:
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
        self.Properties = {
                            "input_dim": proj_dim,
                            "hidden_dim": hidden_dim,
                            "num_layers": num_layers,
                            "dropout": dropout
                        }
        if self.autoencoder:
            self.load_autoencoder(address_autoencoder,Autoencoder_CNN)

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
                x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_dim)
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, hidden_dim)
        """
        if self.autoencoder:
            x = self._encoder(x)

        # x = self.relu(self.proj(x))

        # Apply LayerNorm directly (no flattening needed)
        x = self.LN(x)
        out = self.lstm(x)  # (batch_size, 1)
        return out.squeeze(1)  # (batch_size,)
    
    def load_autoencoder(self,
                         address_autoencoder: str|os.PathLike[str],
                         Autoencoder_CNN: torch.nn.modules) -> None:
        """
        Load the autoencoder model from a file.
        Args:
            address_autoencoder (str): Path to the autoencoder model file
        Returns:
            None: Loads the autoencoder model.
        """
        self.autoencoder = Autoencoder_CNN(embedding_dim = self.proj_dim).to(self.device)
        self.autoencoder.eval()
        if os.path.isfile(address_autoencoder):
            self.autoencoder.load_state_dict(torch.load(address_autoencoder, map_location=self.device))
        else:
            raise FileNotFoundError(f"Autoencoder model file not found at {address_autoencoder}")
        # self.autoencoder.requires_grad_(False)
        return None