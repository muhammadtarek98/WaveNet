import torch
from thop import profile
from timm.models.layers import DropPath
from timm.models._registry import register_model
from basicsr.models.archs.arch_util import LayerNorm,Mlp

class ASFF(torch.nn.Module):
    def __init__(self, in_channels:int, height:int=2,reduction:int=4,bias:bool=False)->None:
        super(ASFF, self).__init__()
        self.height = height
        d = max(int(in_channels/reduction),4)       
        self.fcs = torch.nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=in_channels, out_channels=d,kernel_size= 1, padding=0, bias=bias), torch.nn.PReLU()))
        self.conv=torch.nn.Conv2d(in_channels=d*height,out_channels=height,kernel_size=1,stride=1,padding=0,bias=bias)

    def forward(self, inp_feats:torch.Tensor)->torch.Tensor:
        fusion_vectors = [fc(inp_feats[idx]) for idx,fc in enumerate(self.fcs)]
        fusion_vectors = torch.cat(fusion_vectors, dim=1)#N*(height*d)*H*W
        fusion_vectors = self.conv(fusion_vectors).softmax(dim=1).permute(1,0,2,3).unsqueeze(-3)#N*height*H*W
        return fusion_vectors

class WFB(torch.nn.Module):
    def __init__(self, in_dim:int,
                 out_dim:int,
                 height:int=6,
                 bias:bool=False,
                 proj_drop:float=0.0,
                 norm:torch.nn.Module=LayerNorm)->None:
        super(WFB,self).__init__()
        
        self.norm=norm(normalized_shape=in_dim)
        self.fcs = torch.nn.ModuleList([])
        self.height=height
        for i in range(self.height):
            self.fcs.append(torch.nn.Conv2d(in_channels=in_dim,
                                            out_channels= out_dim,
                                            kernel_size=1,
                                            stride=1,
                                            padding=0,bias=bias))

        self.f3=torch.nn.Conv2d(in_channels=in_dim,out_channels=out_dim,
                                kernel_size=3,stride=1,
                                padding=3//2,groups=out_dim//4,
                                bias=bias)
        self.f5=torch.nn.Conv2d(in_channels=in_dim,out_channels=out_dim,
                                kernel_size=5,stride=1,
                                padding=5//2,groups=out_dim//2,
                                bias=bias)
        self.f7=torch.nn.Conv2d(in_channels=in_dim,out_channels=out_dim,
                                kernel_size=7,stride=1,
                                padding=7//2,groups=out_dim,
                                bias=bias)
        self.reweight_u = Mlp(in_features=out_dim,hidden_features= out_dim // 4,
                              out_features=out_dim * 2)
        self.reweight_v= Mlp(in_features=out_dim, hidden_features=out_dim // 4,
                             out_features= out_dim * 2)
        self.reweight_w = Mlp(in_features=out_dim, hidden_features=out_dim // 4,
                              out_features=out_dim * 2)
        self.proj = torch.nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=1, stride=1,bias=True)
        self.proj_drop = torch.nn.Dropout(proj_drop)
    def forward(self, x:torch.Tensor)->torch.Tensor:
        x=self.norm(x)
        B, C, H, W = x.shape
        ex_ch=[fc(x) for fc in self.fcs]
        ##  cosine wave
        f3=self.f3(ex_ch[3]*torch.cos(ex_ch[4]))
        ##  sine wave
        f5=self.f5(ex_ch[5]*torch.sin(ex_ch[6]))
        ## tanh wave
        f7=self.f7(ex_ch[7]*torch.tanh(ex_ch[8]))
        u=ex_ch[0]+f3
        v=ex_ch[1]+f5 
        w=ex_ch[2]+f7
        # wave-FC
        u=torch.nn.functional.adaptive_avg_pool2d(u,output_size=1)
        u=self.reweight_u(u).reshape(B, C, 2).permute(2, 0, 1).softmax(dim=0).unsqueeze(-1).unsqueeze(-1)
        x1=ex_ch[0]*u[0]+f3*u[1]
        del f3
        v=torch.nn.functional.adaptive_avg_pool2d(v,output_size=1)
        v=self.reweight_v(v).reshape(B, C, 2).permute(2, 0, 1).softmax(dim=0).unsqueeze(-1).unsqueeze(-1)
        x2=ex_ch[1]*v[0]+f5*v[1]
        del f5
        w=torch.nn.functional.adaptive_avg_pool2d(w,output_size=1)
        w=self.reweight_w(w).reshape(B, C, 2).permute(2, 0, 1).softmax(dim=0).unsqueeze(-1).unsqueeze(-1)
        x3=ex_ch[2]*w[0]+f7*w[1]
        del f7
        del ex_ch
        x=x1+x2+x3
        del x1,x2,x3
        x = self.proj(x)
        x = self.proj_drop(x)           
        return x

class WTB(torch.nn.Module):
    def __init__(self, norm_dim:int,
                 dim:int,
                 height:int=6,
                 mlp_ratio:float=2.0,
                 bias:bool=False,
                 attn_drop:float=0.0,
                 drop_path:float=0.0,
                 act_layer:torch.nn.Module=torch.nn.PReLU,
                 norm_layer:torch.nn.Module=LayerNorm):
        super(WTB,self).__init__()
        self.wfb = WFB(in_dim=dim,out_dim=dim,
                       height=height,bias=bias,
                       norm=norm_layer)
        self.drop_path = DropPath(drop_prob=drop_path) if drop_path > 0.0 else torch.nn.Identity()
        self.norm2 = norm_layer(normalized_shape=dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        
    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = x+ self.drop_path(self.wfb(x)) 
        x = x + self.drop_path(self.mlp(self.norm2(x))) 
        return x
## Resizing modules
class Downsample(torch.nn.Module):
    def __init__(self,in_ch:int=3,out_ch:int=3,
                 scale:float=0.5,pixel_shuffle:bool=False,
                 use_norm:bool=False):
        super(Downsample, self).__init__()
        self.norm=LayerNorm(out_ch)
        self.use_norm=use_norm
        if pixel_shuffle:
            self.down = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=in_ch, out_channels=out_ch//4,
                                kernel_size=3, stride=1, padding=1, bias=False),
                torch.nn.PixelUnshuffle(downscale_factor=2))
        
        else:
            self.down = torch.nn.Sequential(torch.nn.Conv2d(in_channels=in_ch,out_channels=out_ch,
                                                            kernel_size=3,stride=1, padding=1,
                                                            bias=False),
                                        torch.nn.UpsamplingBilinear2d(scale_factor=scale),)
    def forward(self, x:torch.Tensor)->torch.Tensor:
        if self.use_norm:
            x = self.norm(self.down(x))
            return x
        else:
            return self.down(x)

class Upsample(torch.nn.Module):
    def __init__(self, in_ch:int,out_ch:int,
                 conv_mode:bool=True,
                 use_norm:bool=False):
        super(Upsample, self).__init__()
        self.use_norm=use_norm
        self.norm=LayerNorm(out_ch)
        if conv_mode:
            self.up = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=in_ch,out_channels=out_ch,
                                kernel_size=3,stride=1, padding=1, bias=False),
                torch.nn.UpsamplingBilinear2d(scale_factor=2))
        else:
            self.up = nn.Sequential(
                        torch.nn.ConvTranspose2d(in_channels=in_ch,out_channels=out_ch,
                                                 kernel_size=3,stride=2,padding=1,
                                                 output_padding=1))
    def forward(self, x:torch.Tensor,y:torch.Tensor)->torch.Tensor:
        x=self.up(x)+y
        if self.use_norm:
            return self.norm(x)
        else:
            return x

def basic_blocks(embed_dims:int, index:int,
                 layers:list,
                 height:int=6,mlp_ratio:float=3.0, bias:bool=False,
                 attn_drop:float=0.0,
                 drop_path_rate:float=0.0,norm_layer:torch.nn.Module=LayerNorm, **kwargs)->torch.nn.Sequential:
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(WTB(norm_dim=embed_dims,
                          dim=embed_dims,
                          height=height,
                          mlp_ratio=mlp_ratio, bias=bias,
                      attn_drop=attn_drop,
                    drop_path=block_dpr, norm_layer=norm_layer))
    blocks = torch.nn.Sequential(*blocks)
    return blocks

class WaveNet(torch.nn.Module):
    """ WaveNet Network """
    def __init__(self, encoder_layers:list,decoder_layers:list,
        encoder_dims:list=None,decoder_dims:list=None,
        transitions:list=None, enc_mlp_ratios:list=None,
        dec_mlp_ratios:list=None,bias:bool=False,
        attn_drop_rate:float=0.0,drop_path_rate:float=0.0,
        height:int=9,use_asff:bool=False,
        norm_layer:torch.nn.Module=LayerNorm):
        super(WaveNet,self).__init__()
        self.use_asff=use_asff
        self.conv_1=torch.nn.Conv2d(in_channels=3, out_channels=encoder_dims[0],
                                    kernel_size=3, stride=1, padding=1, bias=bias)
        encoder = []
        decoder =[]
        for i in range(len(encoder_layers)):
            stage = basic_blocks(embed_dims=encoder_dims[i],
                                 index=i,
                                 layers=encoder_layers,
                                 height=height,
                                 mlp_ratio=enc_mlp_ratios[i],
                                 bias=bias,
                                 attn_drop=attn_drop_rate,
                                 drop_path_rate=drop_path_rate,
                                 norm_layer=norm_layer)
            encoder.append(stage)
            if i >= len(encoder_layers) - 1:
                break
            if transitions[i] or encoder_dims[i] != encoder_dims[i + 1]:
                encoder.append(Downsample(encoder_dims[i], encoder_dims[i + 1], pixel_shuffle=False))
        if len(encoder_layers)>1:
            encoder.append(torch.nn.Conv2d(in_channels=encoder_dims[-1], out_channels=encoder_dims[-1], kernel_size=3, stride=1, padding=1, bias=False))
        self.encoder = torch.nn.ModuleList(encoder)

        for i in range(len(decoder_layers)):
            stage = basic_blocks(decoder_dims[i], i, decoder_layers, height, mlp_ratio=dec_mlp_ratios[i], bias=bias,
                                 attn_drop=attn_drop_rate, drop_path_rate=drop_path_rate,
                                 norm_layer=norm_layer)
            decoder.append(stage)
            if i >= len(encoder_layers) - 1:
                break
            if transitions[i] or decoder_dims[i] != decoder_dims[i + 1]:
                decoder.append(Upsample(decoder_dims[i], decoder_dims[i + 1]))
        self.decoder = torch.nn.ModuleList(decoder)
        if self.use_asff:
            self.asff=ASFF(decoder_dims[-1], height=2)
        self.conv_2=torch.nn.Conv2d(in_channels=decoder_dims[-1], out_channels=3,kernel_size= 3, stride=1, padding=1, bias=bias)
    def forward_tokens(self, x:torch.Tensor)->torch.Tensor:
        enc_maps = []
        for idx, block in enumerate(self.encoder):
            x = block(x)
            #print('down_size',x.size())
            if idx%2==0:
                enc_maps.append(x)
            if idx==len(self.encoder)-1 and len(self.encoder)>1:
                x=enc_maps[-1]+x
                enc_maps[-1]=x
        i=-2
        for idx, block in enumerate(self.decoder):
            #print('up_size',x.size())
            if idx%2==1:
                x= block(x,enc_maps[i])
                i-=1
            else:
                x=block(x)
        return x
    def forward(self, x:torch.Tensor)->torch.Tensor:
        conv_1x=self.conv_1(x)
        embed_x = self.forward_tokens(conv_1x)
        if self.use_asff:
            v=self.asff([conv_1x,embed_x])
            embed_x=conv_1x*v[0]+embed_x*v[1]
        embed_x=self.conv_2(embed_x)
        x=embed_x+x
        return x

@register_model
def WaveNet_T(pretrained:bool=True, **kwargs)->torch.nn.Module:
    transitions = [True]
    encode_layers = [2]
    decode_layers=[2]
    enc_mlp_ratios = [1]
    dec_mlp_ratios=[1]
    enc_dims = [32]
    dec_dims=[32]
    use_asff=False
   
    model = WaveNet(encoder_layers=encode_layers, decoder_layers=decode_layers, encoder_dims=enc_dims,
                    decoder_dims=dec_dims, transitions=transitions, enc_mlp_ratios=enc_mlp_ratios,
                    dec_mlp_ratios=dec_mlp_ratios, height=9, use_asff=use_asff, norm_layer=LayerNorm, **kwargs)
    return model
@register_model
def WaveNet_S(pretrained:bool=False, **kwargs)->torch.nn.Module:
    transitions = [True]
    encode_layers = [2]
    decode_layers=[2]
    enc_mlp_ratios = [4]
    dec_mlp_ratios=[4]
    enc_dims = [128]
    dec_dims=[128]
    use_asff=False
   
    model = WaveNet(encode_layers, decode_layers, enc_dims, dec_dims, transitions=transitions,
                    enc_mlp_ratios=enc_mlp_ratios, dec_mlp_ratios=dec_mlp_ratios, height=9, use_asff=use_asff,
                    norm_layer=LayerNorm, **kwargs)
    return model
@register_model
def WaveNet_B(pretrained:bool=False, **kwargs)->torch.nn.Module:
    transitions = [True,True, True]
    encode_layers = [2,2,4]
    decode_layers=[4,2,2]
    enc_mlp_ratios = [2,2,2]
    dec_mlp_ratios=[2,2,2]
    enc_dims = [128,192,256]
    dec_dims=[256,192,128]
    use_asff=True # For training FvieK: False
   
    model = WaveNet(encode_layers, decode_layers, encoder_dims=enc_dims, decoder_dims=dec_dims, transitions=transitions,
                    enc_mlp_ratios=enc_mlp_ratios, dec_mlp_ratios=dec_mlp_ratios, height=9, use_asff=use_asff,
                    norm_layer=LayerNorm, **kwargs)
    return model
