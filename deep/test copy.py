import os
import torch
from EVSSM import EVSSM
from torchvision.transforms import functional as F
from PIL import Image
from pathlib import Path
import torch.nn.functional as nnF
dir = Path(__file__).resolve().parent

MODEL_PATH = dir / "net_g_GoPro.pth"

INPUT_PATH = dir.parent / "data" / "arbres.png"

OUTPUT_PATH = dir.parent / "mon_image_net.png"



def deblur_single_image():
    # Charger le modèle
    model = EVSSM()

    # Charger les poids
    state_dict = torch.load(MODEL_PATH)['params']
    model.load_state_dict(state_dict, strict=True)

    # Choix du device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # Charger l'image
    img = Image.open(INPUT_PATH).convert('RGB')
    input_img = F.to_tensor(img).unsqueeze(0).to(device)  # [1, C, H, W]

    # --- 🔢 Padding pour avoir une taille compatible avec le réseau ---
    b, c, h, w = input_img.shape
    h_n = (4 - h % 4) % 4
    w_n = (4 - w % 4) % 4

    if h_n != 0 or w_n != 0:
        input_pad = nnF.pad(input_img, (0, w_n, 0, h_n), mode='reflect')
    else:
        input_pad = input_img

    with torch.no_grad():
        pred = model(input_pad)

    # enlever le padding pour revenir à la taille originale
    pred = pred[:, :, :h, :w]

    # Clamp + petit offset comme dans ton code
    pred_clip = torch.clamp(pred, 0, 1)
    pred_clip += 0.5 / 255

    # Conversion en image PIL
    pred_img = F.to_pil_image(pred_clip.squeeze(0).cpu(), 'RGB')

    # Créer le dossier de sortie si besoin
    out_dir = OUTPUT_PATH.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sauvegarde
    pred_img.save(OUTPUT_PATH)
    print(f"✅ Image défloutée sauvegardée dans : {OUTPUT_PATH}")


if __name__ == "__main__":
    deblur_single_image()