import torch
from ptflops import get_model_complexity_info

if __name__ == '__main__':
    from AutoEncoder_CNNV1_0 import Autoencoder_CNN, Encoder_CNN
else:
    from .AutoEncoder_CNNV1_0 import Autoencoder_CNN, Encoder_CNN


dataSize = (1, 201, 201)
with torch.cuda.device(0):
    net:torch.nn.Module = Autoencoder_CNN(embedding_dim=1024)
    macs, params = get_model_complexity_info(net, dataSize, as_strings=True, backend='pytorch',
                                        print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    net:torch.nn.Module = Encoder_CNN(embedding_dim=1024)
    macs, params = get_model_complexity_info(net, dataSize, as_strings=True, backend='pytorch',
                                        print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))