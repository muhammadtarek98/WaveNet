import torch,torchvision,cv2,torchinfo
import numpy as np
from thop import profile

from basicsr.models.archs.WaveNet_arch import WaveNet,WaveNet_T,WaveNet_B
from waternet.trail import transform_image,transform_array_to_image
from basicsr.models.archs.arch_util import LayerNorm,Mlp
def load_model(chkpt_dir:str,
               device:torch.device):
    #model= WaveNet(,
    state_dict=torch.load(f="/home/muahmmad/projects/Image_enhancement/WaveNet/checkpoints/5k/WaveNet_B_5k.pth",
                                                map_location=device)
    #print(state_dict.keys())
    for key in state_dict.keys():
        print(key," ",state_dict[key].shape)
    #model.load_state_dict(state_dict=state_dict)
    model=model.to(device)
    input=torch.randn(size=(1,3,480,480)).to(device=device)
    torchinfo.summary(model=model,input_data=input)


def run():
    device=torch.device(device="cuda" if torch.cuda.is_available() else "cpu")
    ckpt="/home/muahmmad/projects/Image_enhancement/WaveNet/checkpoints/5k/WaveNet_T_5k.pth"
    model=load_model(ckpt_dir=ckpt,
                     device=device)
    image = cv2.imread(
        filename="/home/muahmmad/projects/Image_enhancement/Enhancement_Dataset/9898_no_fish_f000130.jpg")
    tensors=transform_image(img=image)
    raw_image_tensor=tensors["X"]
    raw_image_tensor=raw_image_tensor.to(device=device)

    model=model.to(device)
    model.eval()
    with torch.no_grad():
        pred=model(raw_image_tensor)
    pred=pred.squeeze_()
    pred=torch.permute(input=pred,dims=(1,2,0))
    output=pred.detach().cpu().numpy()
    #output=transform_array_to_image(arr=output)
    print(output)
    cv2.imshow(winname="pred",mat=output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #print(pred.shape)

if __name__=='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #load_model(chkpt_dir="/home/muahmmad/projects/Image_enhancement/WaveNet/checkpoints/5k/WaveNet_B_5k.pth",
    #           device=device)
    transitions = [True, True, True]
    encoder_layers = [2, 2, 4]
    decoder_layers = [4, 2, 2]
    enc_mlp_ratios = [2, 2, 2]
    dec_mlp_ratios = [2, 2, 2]
    enc_dims = [128, 192, 256]
    dec_dims = [256, 192, 128]
    use_asff = True
    model = WaveNet(encoder_layers=encoder_layers,decoder_layers= decoder_layers, encoder_dims=enc_dims, decoder_dims=dec_dims, transitions=transitions,
                    enc_mlp_ratios=enc_mlp_ratios, dec_mlp_ratios=dec_mlp_ratios, height=9)
    input=torch.randn(size=(1,3,480,480))
    ckpt_dir="/home/muahmmad/projects/Image_enhancement/WaveNet/checkpoints/5k/WaveNet_B_5k.pth"
    state_dict = torch.load(f=ckpt_dir)
    print(state_dict.keys())
    model.load_state_dict(state_dict=state_dict)
    torchinfo.summary(model=model,input_data=input)


    """model=WaveNet_T()
    model = model.to(device)
    if str(device) =='cuda':
        input=torch.randn(1,3,256,256).cuda()
    else:
        input=torch.randn(1,3,256,256)
    print(model)
    flops,params=profile(model,inputs=(input,))
    print('flops:{}G params:{}M'.format(flops/1e9,params/1e6))
"""