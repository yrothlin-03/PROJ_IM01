import os
import torch
from EVSSM.models.EVSSM import EVSSM
from FFTformer.basicsr.models.archs.fftformer_arch import  fftformer as FFTformer
from torchvision.transforms import functional as F
from PIL import Image
from pathlib import Path
import torch.nn.functional as nnF

dir = Path(__file__).resolve().parent

MODEL_PATH_EVSSM = dir / "EVSSM" / "checkpoints" / "net_g_realblur_j.pth"
MODEL_PATH_FFTformer = dir / "FFTformer" / "checkpoints" / "net_g_Realblur_J.pth"

INPUT_PATH = dir.parent / "data" / "arbres.png"


OUTPUT_PATH = dir / "deblurred.png"
OUTPUT_PATH_EVSSM = dir / "deblurred_evssm.png"
OUTPUT_PATH_FFTformer = dir / "deblurred_fftformer.png"



def deblur_single_image_EVSSM():
    model = EVSSM()

    state_dict = torch.load(MODEL_PATH_EVSSM)['params']
    model.load_state_dict(state_dict, strict=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    img = Image.open(INPUT_PATH).convert('RGB')
    input_img = F.to_tensor(img).unsqueeze(0).to(device)  # [1, C, H, W]

    b, c, h, w = input_img.shape
    h_n = (4 - h % 4) % 4
    w_n = (4 - w % 4) % 4

    if h_n != 0 or w_n != 0:
        input_pad = nnF.pad(input_img, (0, w_n, 0, h_n), mode='reflect')
    else:
        input_pad = input_img

    with torch.no_grad():
        pred = model(input_pad)

    pred = pred[:, :, :h, :w]

    pred_clip = torch.clamp(pred, 0, 1)
    pred_clip += 0.5 / 255

    pred_img = F.to_pil_image(pred_clip.squeeze(0).cpu(), 'RGB')

    out_dir = OUTPUT_PATH_EVSSM.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_img.save(OUTPUT_PATH_EVSSM)
    print(f"✅ Image défloutée (EVSSM) sauvegardée dans : {OUTPUT_PATH_EVSSM}")



def deblur_single_image_FFTformer():
    model = FFTformer()  

    state_dict = torch.load(MODEL_PATH_FFTformer)
    model.load_state_dict(state_dict, strict=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    img = Image.open(INPUT_PATH).convert("RGB")
    input_img = F.to_tensor(img).unsqueeze(0).to(device)  # [1, C, H, W]

    b, c, h, w = input_img.shape
    h_n = (32 - h % 32) % 32
    w_n = (32 - w % 32) % 32
    input_pad = nnF.pad(input_img, (0, w_n, 0, h_n), mode="reflect")

    with torch.no_grad():
        pred = model(input_pad)
        if device.type == "cuda":
            torch.cuda.synchronize()

    pred = pred[:, :, :h, :w]

    pred_clip = torch.clamp(pred, 0, 1)
    pred_clip += 0.5 / 255

    pred_img = F.to_pil_image(pred_clip.squeeze(0).cpu(), "RGB")

    out_dir = OUTPUT_PATH_FFTformer.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_img.save(OUTPUT_PATH_FFTformer)
    print(f"✅ Image défloutée (FFTformer) sauvegardée dans : {OUTPUT_PATH_FFTformer}")

if __name__ == "__main__":
    deblur_single_image_EVSSM()
    deblur_single_image_FFTformer()