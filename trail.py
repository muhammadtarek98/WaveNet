import torch,torchvision,cv2,torchinfo
import numpy as np

from basicsr.models.archs.WaveNet_arch import WaveNet,WaveNet_T,WaveNet_B
from waternet.trail import transform_image,transform_array_to_image,load_model

def run():
    device=torch.device(device="cuda" if torch.cuda.is_available() else "cpu")
    ckpt="/home/muahmmad/projects/Image_enhancement/WaveNet/checkpoints/5k/WaveNet_S_5k.pth"
    model=load_model(ckpt_dir=ckpt,
                     model_type=WaveNet_T,
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

run()