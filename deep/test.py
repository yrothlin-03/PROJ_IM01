import torch
from PIL import Image
from torchvision.transforms import functional as F
import torch.nn.functional as Fnn
from EVSSM import EVSSM



img_dir = "./data/" 
img_name = "arbres.png"



model = EVSSM()
ckpt = torch.load("./deep/net_g_GoPro.pth", map_location="cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(ckpt["params"], strict=True)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)



img = Image.open(f"{img_dir}{img_name}").convert("RGB")   
x = F.to_tensor(img).unsqueeze(0).to(device) 
print("x.shape before :", x.shape, x.dtype, x.device)
x = Fnn.interpolate(
    x,
    size=(1384, 1384),  
    mode="bilinear",
    align_corners=False,
)
print("x.shape after :", x.shape, x.dtype, x.device)
print("Inferencing...")
with torch.no_grad():
    y = model(x)  

y = torch.clamp(y, 0, 1)
out = F.to_pil_image(y.squeeze(0).cpu())
out.save("deblur.png")