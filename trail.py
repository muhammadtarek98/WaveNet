import torch,torchvision,cv2,torchinfo
import numpy as np
from thop import profile

from basicsr.models.archs.WaveNet_arch import WaveNet,WaveNet_T,WaveNet_B
from waternet.trail import transform_image,transform_array_to_image
from basicsr.models.archs.arch_util import LayerNorm,Mlp

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
    model.eval()
    image = cv2.imread(
        filename="/home/muahmmad/projects/Image_enhancement/Enhancement_Dataset/9898_no_fish_f000130.jpg")
    image=cv2.cvtColor(src=image,code=cv2.COLOR_BGR2RGB)
    raw_input_tensor=transform_image(img=image)["X"]
    with torch.no_grad():
        pred=model(raw_input_tensor)
    pred=pred.squeeze_()
    pred=torch.permute(input=pred,dims=(1,2,0))
    pred = pred.detach().cpu().numpy()
    pred=transform_array_to_image(pred)
    cv2.imshow(winname="pred",mat=pred)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

