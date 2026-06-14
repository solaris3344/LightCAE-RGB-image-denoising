# -*- coding: utf-8 -*-
"""
test_ablation_cbsd68.py — Ablasyon E (d=2) ve F (d=1) modellerini CBSD68 sigma=25
uzerinde AYNI gurultulu girdilerle test eder.

Protokol (makaledeki ile ayni):
  - tek gecis (TTA yok)
  - reflect padding ile 16'nin katina tamamlama
  - cikti uint8'e yuvarlanir, metrikler ondan hesaplanir
  - PSNR: skimage, data_range=1.0
  - SSIM: kanal-ortalama skimage

KULLANIM:
  1) CONFIG'te CBSD68 klasorunu ve checkpoint yollarini ayarla
  2) python test_ablation_cbsd68.py
"""

import os, random
import numpy as np
import torch
import torch.nn as nn
import pywt
from skimage import io
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

# ============================ CONFIG ============================
CBSD68_DIR = r"A:\iyiler_psnr_3282_sigma_15\CBSD68"   # CBSD68 temiz goruntu klasoru

MODELS = {
    # ad : (checkpoint yolu, dilation)
    "E (d=2, reference)": (r"ablation_s25_dil2_scratch.pt", 2),
    "F (d=1, variant)":   (r"ablation_s25_dil1_scratch.pt", 1),
}

SIGMA        = 25 / 255.0
SCALE_FACTOR = 10.0
WAVELET      = "bior4.4"
NOISE_SEED   = 1234          # ayni gurultulu girdiler icin sabit
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ================================================================

# ---- Mimari (egitimle ayni; dilation parametreli) ----
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid())
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class LightResBlock(nn.Module):
    def __init__(self, ch, dilation=2):
        super().__init__()
        pad = dilation  # 3x3 conv icin padding = dilation
        self.body = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, padding=pad, dilation=dilation,
                      padding_mode="reflect", bias=True),
            nn.GELU(),
            nn.Conv2d(ch, ch, 3, 1, padding=1,
                      padding_mode="reflect", bias=True),
            SEBlock(ch, 16))
    def forward(self, x):
        return x + self.body(x)

class LightCAE(nn.Module):
    def __init__(self, in_ch=12, ch=64, n_blocks=20, dilation=2):
        super().__init__()
        self.head = nn.Conv2d(in_ch, ch, 3, 1, 1, padding_mode="reflect", bias=True)
        self.body = nn.Sequential(*[LightResBlock(ch, dilation) for _ in range(n_blocks)])
        self.tail = nn.Conv2d(ch, in_ch, 3, 1, 1, padding_mode="reflect", bias=True)
    def forward(self, x):
        f = self.head(x)
        return self.tail(f + self.body(f))

# ---- DWT yardimcilari (egitimle ayni) ----
def dwt_rgb(img):
    outs = []
    for c in range(3):
        LL, (LH, HL, HH) = pywt.dwt2(img[..., c], WAVELET, mode="periodization")
        outs.extend([LL, LH, HL, HH])
    return np.stack(outs, 0).astype(np.float32)

def idwt_rgb(co):
    out = np.zeros((co.shape[1]*2, co.shape[2]*2, 3), np.float32)
    for c in range(3):
        i = c*4
        out[..., c] = pywt.idwt2((co[i], (co[i+1], co[i+2], co[i+3])),
                                 WAVELET, mode="periodization")
    return np.clip(out, 0, 1)

def load_model(path, dilation):
    m = LightCAE(12, 64, 20, dilation).to(DEVICE)
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    state = ck.get("ema_shadow", ck.get("model", ck))
    m.load_state_dict(state)
    m.eval()
    n = sum(p.numel() for p in m.parameters())
    print(f"  {os.path.basename(path)} | dilation={dilation} | params={n:,} | "
          f"epoch={ck.get('epoch','?')} | ic-val={ck.get('best_psnr','?')}")
    return m

def to_uint8(a):  # protokol: uint8 yuvarlama
    return np.round(np.clip(a, 0, 1) * 255.0) / 255.0

@torch.no_grad()
def denoise(model, noisy):
    h, w, _ = noisy.shape
    ph, pw = (16 - h % 16) % 16, (16 - w % 16) % 16
    pad = np.pad(noisy, ((0, ph), (0, pw), (0, 0)), "reflect")
    nd  = dwt_rgb(pad)
    inp = torch.from_numpy(nd * SCALE_FACTOR)[None].to(DEVICE)
    res = model(inp).float().cpu().numpy()[0]
    out = idwt_rgb(nd - res / SCALE_FACTOR)[:h, :w]
    return out

def main():
    files = sorted(f for f in os.listdir(CBSD68_DIR)
                   if f.lower().endswith((".png", ".jpg", ".bmp", ".tif")))
    assert files, "CBSD68 klasoru bos veya yol yanlis!"
    print(f"CBSD68: {len(files)} goruntu | sigma={SIGMA*255:.0f} | seed={NOISE_SEED}\n")

    print("Modeller yukleniyor:")
    models = {name: load_model(p, d) for name, (p, d) in MODELS.items()}
    res = {name: {"psnr": [], "ssim": []} for name in models}

    rng = np.random.default_rng(NOISE_SEED)
    for k, fname in enumerate(files, 1):
        gt = io.imread(os.path.join(CBSD68_DIR, fname))
        if gt.ndim == 2: gt = np.stack([gt]*3, -1)
        gt = gt[..., :3].astype(np.float32) / 255.0

        noisy = np.clip(gt + rng.standard_normal(gt.shape).astype(np.float32) * SIGMA, 0, 1)

        for name, m in models.items():
            out = to_uint8(denoise(m, noisy))
            p = psnr_fn(gt, out, data_range=1.0)
            s = float(np.mean([ssim_fn(gt[..., c], out[..., c], data_range=1.0)
                               for c in range(3)]))
            res[name]["psnr"].append(p)
            res[name]["ssim"].append(s)

        if k % 10 == 0 or k == len(files):
            print(f"  {k}/{len(files)} tamamlandi")

    print("\n================ SONUCLAR (CBSD68, sigma=25) ================")
    for name in models:
        P = np.mean(res[name]["psnr"]); S = np.mean(res[name]["ssim"])
        print(f"  {name:24s} | PSNR: {P:.2f} dB | SSIM: {S:.4f}")
    if len(models) == 2:
        names = list(models)
        dP = np.mean(res[names[0]]["psnr"]) - np.mean(res[names[1]]["psnr"])
        dS = np.mean(res[names[0]]["ssim"]) - np.mean(res[names[1]]["ssim"])
        print(f"\n  Fark (E - F): {dP:+.2f} dB | {dS:+.4f} SSIM   <- dilation katkisi")

if __name__ == "__main__":
    main()
