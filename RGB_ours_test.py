import os
import time
import numpy as np
import torch
import torch.nn as nn
import pywt
from skimage import io
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
import cv2

# ===================== 1. AYARLAR =====================
TEST_DIR   = r"...path\CBSD68"
MODEL_PATH = "rgb_model_v22_sigma50_best.pt"
OUTPUT_DIR = r"...path\test_results_v22"

SCALE_FACTOR = 10.0
SIGMA_NOISE  = 50 / 255   # 15, 25 or 50 / 255
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WAVELET      = 'bior4.4'
USE_TTA      = False  # 

# ===================== 2. MİMARİ (train ile aynı) =====================

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class LightResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, padding=2, dilation=2,
                      padding_mode='reflect', bias=True),
            nn.GELU(),
            nn.Conv2d(ch, ch, 3, 1, padding=1,
                      padding_mode='reflect', bias=True),
            SEBlock(ch, reduction=16),
        )

    def forward(self, x):
        return x + self.body(x)


class LightCAE(nn.Module):
    def __init__(self, in_ch=12, ch=64, n_blocks=20):
        super().__init__()
        self.head = nn.Conv2d(in_ch, ch, 3, 1, 1,
                              padding_mode='reflect', bias=True)
        self.body = nn.Sequential(*[LightResBlock(ch) for _ in range(n_blocks)])
        self.tail = nn.Conv2d(ch, in_ch, 3, 1, 1,
                              padding_mode='reflect', bias=True)

    def forward(self, x):
        feat = self.head(x)
        return self.tail(feat + self.body(feat))


# ===================== 3. YARDIMCI FONKSİYONLAR =====================

def dwt_rgb(img):
    outs = []
    for c in range(3):
        LL, (LH, HL, HH) = pywt.dwt2(img[..., c], WAVELET, mode='periodization')
        outs.extend([LL, LH, HL, HH])
    return np.stack(outs, axis=0).astype(np.float32)


def idwt_rgb(coeffs):
    out = np.zeros((coeffs.shape[1]*2, coeffs.shape[2]*2, 3), dtype=np.float32)
    for c in range(3):
        idx = c * 4
        out[..., c] = pywt.idwt2(
            (coeffs[idx], (coeffs[idx+1], coeffs[idx+2], coeffs[idx+3])),
            WAVELET, mode='periodization'
        )
    return np.clip(out, 0, 1)


def single_pass(model, noisy_pad):
    """Tek forward pass — TTA yok."""
    x_dwt = dwt_rgb(noisy_pad)
    inp   = torch.from_numpy(x_dwt * SCALE_FACTOR)[None].to(DEVICE)
    with torch.no_grad():
        res = model(inp).cpu().numpy()[0]
    return idwt_rgb(x_dwt - res / SCALE_FACTOR)


def apply_tta(model, noisy_pad):
    """8 yönlü TTA: 4 rotasyon × 2 (flip/no-flip)."""
    preds = []
    for rot in [0, 1, 2, 3]:
        for flip in [False, True]:
            x = np.rot90(noisy_pad, rot).copy()
            if flip:
                x = np.flip(x, axis=1).copy()

            x_dwt = dwt_rgb(x)
            inp   = torch.from_numpy(x_dwt * SCALE_FACTOR)[None].to(DEVICE)
            with torch.no_grad():
                res = model(inp).cpu().numpy()[0]
            out = idwt_rgb(x_dwt - res / SCALE_FACTOR)

            if flip:
                out = np.flip(out, axis=1).copy()
            out = np.rot90(out, -rot).copy()
            preds.append(out)

    return np.mean(preds, axis=0)


# ===================== 4. MODEL YÜKLEME =====================

def load_model(path):
    """
    Checkpoint'ten modeli yükle.
    Hem EMA ağırlıklarını hem de ham model ağırlıklarını destekler.
    """
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)

    # Mimari bilgisi varsa onu kullan, yoksa varsayılan
    arch = checkpoint.get('arch', {'in_ch': 12, 'ch': 64, 'n_blocks': 20})
    model = LightCAE(**arch).to(DEVICE)

    # Önce EMA ağırlıklarını dene (daha iyi performans)
    if 'ema_shadow' in checkpoint:
        model.load_state_dict(checkpoint['ema_shadow'])
        print("EMA ağırlıkları yüklendi.")
    else:
        model.load_state_dict(checkpoint['model'])
        print("Model ağırlıkları yüklendi.")

    model.eval()
    epoch     = checkpoint.get('epoch', '?')
    best_psnr = checkpoint.get('best_psnr', '?')
    print(f"Epoch: {epoch} | Eğitim Val PSNR: {best_psnr}")
    return model


# ===================== 5. ANA DÖNGÜ =====================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 65)
    print(f"  LightCAE v2.2 Test | σ={int(SIGMA_NOISE*255)} | TTA={USE_TTA}")
    print(f"  Test dizini : {TEST_DIR}")
    print(f"  Model       : {MODEL_PATH}")
    print("=" * 65)

    model = load_model(MODEL_PATH)

    # Parametre sayısı
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parametre: {total_params:,} ({total_params/1e6:.2f} M)\n")

    files = sorted([
        f for f in os.listdir(TEST_DIR)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))
    ])

    if not files:
        print(f"HATA: {TEST_DIR} dizininde görüntü bulunamadı.")
        return

    psnr_list, ssim_list, time_list = [], [], []

    for fn in files:
        # ── Görüntü yükle ──
        img = io.imread(os.path.join(TEST_DIR, fn)).astype(np.float32) / 255.0
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)
        img = img[..., :3]

        # ── Gürültü ekle (tekrarlanabilirlik için sabit seed) ──
        np.random.seed(42)
        noisy = np.clip(img + np.random.randn(*img.shape).astype(np.float32) * SIGMA_NOISE, 0, 1)

        # ── Padding (DWT için 16'nın katı olmalı) ──
        h, w, _ = noisy.shape
        ph = (16 - h % 16) % 16
        pw = (16 - w % 16) % 16
        noisy_pad = np.pad(noisy, ((0, ph), (0, pw), (0, 0)), mode='reflect')

        # ── Çıkarım ──
        t0 = time.time()
        if USE_TTA:
            denoised_pad = apply_tta(model, noisy_pad)
        else:
            denoised_pad = single_pass(model, noisy_pad)
        elapsed = time.time() - t0

        denoised = np.clip(denoised_pad[:h, :w], 0, 1)

        # ── Metrikler ──
        p = psnr_metric(img, denoised, data_range=1.0)
        s = ssim_metric(img, denoised, data_range=1.0, channel_axis=2)
        psnr_list.append(p)
        ssim_list.append(s)
        time_list.append(elapsed)

        print(f"{fn:<25} | PSNR: {p:.2f} dB | SSIM: {s:.4f} | {elapsed*1000:.1f} ms")

        # ── Görsel kaydet (noisy | denoised yan yana) ──
        vis = np.concatenate([noisy, denoised], axis=1)
        vis_uint8 = (vis * 255).astype(np.uint8)
        save_path = os.path.join(OUTPUT_DIR, fn)
        cv2.imwrite(save_path, cv2.cvtColor(vis_uint8, cv2.COLOR_RGB2BGR))

    # ── Özet ──
    print("\n" + "=" * 65)
    print(f"  SONUÇLAR — σ={int(SIGMA_NOISE*255)} | TTA={USE_TTA}")
    print("=" * 65)
    print(f"  Görüntü sayısı : {len(psnr_list)}")
    print(f"  Ortalama PSNR  : {np.mean(psnr_list):.4f} dB")
    print(f"  Ortalama SSIM  : {np.mean(ssim_list):.4f}")
    print(f"  Min PSNR       : {np.min(psnr_list):.4f} dB  ({files[np.argmin(psnr_list)]})")
    print(f"  Max PSNR       : {np.max(psnr_list):.4f} dB  ({files[np.argmax(psnr_list)]})")
    print(f"  Ort. Süre      : {np.mean(time_list)*1000:.1f} ms / görüntü")
    print(f"  Sonuçlar       : {OUTPUT_DIR}")
    print("=" * 65)

    # ── Log dosyasına yaz ──
    log_path = os.path.join(OUTPUT_DIR, f"results_sigma{int(SIGMA_NOISE*255)}_tta{USE_TTA}.txt")
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"Sigma: {int(SIGMA_NOISE*255)} | TTA: {USE_TTA}\n")
        f.write(f"{'Dosya':<25} PSNR(dB)  SSIM    Süre(ms)\n")
        f.write("-" * 60 + "\n")
        for i, fn in enumerate(files):
            f.write(f"{fn:<25} {psnr_list[i]:.4f}   {ssim_list[i]:.4f}  {time_list[i]*1000:.1f}\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'ORTALAMA':<25} {np.mean(psnr_list):.4f}   {np.mean(ssim_list):.4f}  {np.mean(time_list)*1000:.1f}\n")
    print(f"  Log kaydedildi : {log_path}")


if __name__ == "__main__":
    main()