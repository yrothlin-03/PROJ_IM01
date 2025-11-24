import torch
from PIL import Image
from torchvision.transforms import functional as F
from .EVSSM import EVSSM



img_dir = "./data/" 
img_name = "perso1.png"




model = EVSSM()
ckpt = torch.load("./deep/net_g_GoPro.pth", map_location="cpu")  
model.load_state_dict(ckpt["params"], strict=True)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

img = Image.open(f"{img_dir}{img_name}").convert("RGB")   
x = F.to_tensor(img).unsqueeze(0).to(device) 

with torch.no_grad():
    y = model(x)

y = torch.clamp(y, 0, 1)
out = F.to_pil_image(y.squeeze(0).cpu())
out.save("deblur.png")