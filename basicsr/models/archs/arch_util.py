import torch

class dw_conv(torch.nn.Module):
    def __init__(self, in_channels:int, out_channels:int,
                 kernel:int=3,stride:int=1,
                 padding:int=1,scale:int=2,
                 bias:bool=True,groups:int=1):
        super().__init__()
        dw_channel = out_channels * scale
        self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=dw_channel,
                                     kernel_size=1, stride=1,
                                     padding=0, groups=groups, bias=bias)
        self.conv2 = torch.nn.Conv2d(in_channels=dw_channel, out_channels=out_channels,
                                     kernel_size=kernel, stride=stride,
                                     padding=padding, groups=out_channels, bias=bias)
    def forward(self,x:torch.Tensor)->torch.Tensor:
        x=self.conv1(x)
        x=self.conv2(x)
        return x

class DoubleConv(torch.nn.Module):
    def __init__(self, in_channels:int, out_channels:int=3,kernel:int=3,
                 stride:int=1,padding:int=1,groups:int=1,
                 norm:bool=False,bias:bool=False):
        super(DoubleConv,self).__init__()
        self.conv = torch.nn.Sequential(
            dw_conv(in_channels=in_channels,out_channels= out_channels, kernel=kernel, stride=stride, padding=padding, bias=bias, groups=groups),
                LayerNorm(in_ch) if norm else torch.nn.Identity(),
                torch.nn.PReLU(),
            dw_conv(out_channels, out_channels, kernel=kernel, stride=stride, padding=padding, bias=bias, groups=groups),
                LayerNorm(out_channels) if norm else torch.nn.Identity(),
                torch.nn.PReLU(),
            )
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return self.conv(x)
class Mlp(torch.nn.Module):
    def __init__(self, in_features:int,
                 hidden_features:int=None,
                 out_features:int=None,
                 act_layer:torch.nn.Module=torch.nn.PReLU,
                 drop:float=0.0):
        super(Mlp,self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.act = act_layer()
        self.drop = torch.nn.Dropout(drop)
        self.fc1 = torch.nn.Conv2d(in_channels=in_features,
                                   out_channels= hidden_features,
                                   kernel_size=1,stride= 1)
        self.fc2 = torch.nn.Conv2d(in_channels=hidden_features,
                                   out_channels=out_features,
                                   kernel_size=1, stride=1)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x   

class LayerNorm(torch.nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self,
                 normalized_shape:list,
                 eps:float=1e-6,
                 data_format:str="channels_first"):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.ones(size=normalized_shape)
        )
        self.bias = torch.nn.Parameter(
            torch.zeros(size=normalized_shape)
        )
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        if self.data_format == "channels_last":
            return torch.nn.functional.layer_norm(input=x,normalized_shape=self.normalized_shape,
                                                  weight=self.weight,
                                                  bias=self.bias,eps= self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x