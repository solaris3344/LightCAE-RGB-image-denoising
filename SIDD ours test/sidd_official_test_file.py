import os
import numpy as np
import torch
import torch.nn as nn
import pywt
from skimage import io
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
import warnings

warnings.filterwarnings("ignore")

# ===================== 1. AYARLAR =====================
# MATLAB ile çıkardığınız klasörlerin tam yolları
NOISY_DIR = r"...path\SIDD_Val_PNG\Noisy"
GT_DIR    = r"...path\SIDD_Val_PNG\GT"

# YENİ EKLENDİ: Temizlenmiş görüntülerin kaydedileceği klasör
OUTPUT_DIR = r"...path\SIDD_Val_PNG\Denoised" 

# Transfer öğrenme ile eğittiğimiz model
MODEL_PATH = "rgb_model_v22_sidd_finetuned.pt"

# ── Test Yapılandırması ──
USE_EMA  = False   # False: ham (raw) ağırlıklar, True: EMA gölge ağırlıkları
                   # Eğitim logunda RAW her zaman daha iyiydi → False bırakın
USE_TTA  = False   # False: tek geçiş (paper'larla adil karşılaştırma)
                   # True: 8x TTA (D4 grubu); +0.05-0.15 dB ama 8x yavaş
USE_UINT8_ROUND = True   # Resmi SIDD benchmark'ı uint8 round üzerinden ölçer.
                         # Restormer/MIRNet/RIDNet rakamları bu şekildedir.
                         # True bırakın → adil karşılaştırma.

SCALE_FACTOR = 10.0
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WAVELET      = 'bior4.4'

# ===================== 2. MİMARİ =====================
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
            nn.Conv2d(ch, ch, 3, 1, padding=2, dilation=2, padding_mode='reflect'),
            nn.GELU(),
            nn.Conv2d(ch, ch, 3, 1, padding=1, padding_mode='reflect'),
            SEBlock(ch, reduction=16),
        )
    def forward(self, x):
        return x + self.body(x)

class LightCAE(nn.Module):
    def __init__(self, in_ch=12, ch=64, n_blocks=20):
        super().__init__()
        self.head = nn.Conv2d(in_ch, ch, 3, 1, 1, padding_mode='reflect')
        self.body = nn.Sequential(*[LightResBlock(ch) for _ in range(n_blocks)])
        self.tail = nn.Conv2d(ch, in_ch, 3, 1, 1, padding_mode='reflect')
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

def load_model(path, use_ema=False):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    arch = ckpt.get('arch', {'in_ch': 12, 'ch': 64, 'n_blocks': 20})
    model = LightCAE(**arch).to(DEVICE)

    loaded_from = None
    if use_ema:
        if 'ema_shadow' in ckpt:
            model.load_state_dict(ckpt['ema_shadow']);  loaded_from = "ema_shadow"
        else:
            model.load_state_dict(ckpt['model']);       loaded_from = "model (ema yok)"
    else:
        if 'raw_state' in ckpt:
            model.load_state_dict(ckpt['raw_state']);   loaded_from = "raw_state"
        elif 'model' in ckpt:
            model.load_state_dict(ckpt['model']);       loaded_from = "model"
        else:
            model.load_state_dict(ckpt['ema_shadow']);  loaded_from = "ema_shadow (raw yok)"

    model.eval()
    print(f"Model yüklendi: {path}")
    print(f"  Kaynak: '{loaded_from}'")
    print(f"  Eğitim epoch: {ckpt.get('epoch', '?')}")
    print(f"  Eğitim best_psnr (kendi val'i): {ckpt.get('best_psnr', '?')}")
    print(f"  Eğitim best_source: {ckpt.get('best_source', '?')}")
    return model

def denoise_single(model, noisy_img):
    x_dwt = dwt_rgb(noisy_img)
    inp = torch.from_numpy(x_dwt * SCALE_FACTOR)[None].to(DEVICE)
    with torch.no_grad():
        res = model(inp).cpu().numpy()[0]
    return idwt_rgb(x_dwt - res / SCALE_FACTOR)

def denoise_with_tta(model, noisy_img):
    preds = []
    for k in range(4):
        for flip in [False, True]:
            x = np.rot90(noisy_img, k=k, axes=(0, 1))
            if flip:
                x = np.flip(x, axis=1)
            x = np.ascontiguousarray(x)
            y = denoise_single(model, x)
            if flip:
                y = np.flip(y, axis=1)
            y = np.rot90(y, k=-k, axes=(0, 1))
            preds.append(np.ascontiguousarray(y))
    return np.mean(preds, axis=0)

def to_uint8_then_back(img):
    return np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8).astype(np.float32) / 255.0

# ===================== 4. ANA DÖNGÜ =====================
def main():
    print("=" * 65)
    print("  RESMİ SIDD VALIDATION SETİ (1280 BLOK) TESTİ BAŞLIYOR")
    print("=" * 65)
    print(f"  Ağırlık tipi : {'EMA' if USE_EMA else 'RAW'}")
    print(f"  TTA          : {'AÇIK (8x)' if USE_TTA else 'KAPALI'}")
    print(f"  uint8 round  : {'AÇIK (resmi protokol)' if USE_UINT8_ROUND else 'KAPALI'}")
    print("=" * 65)

    model = load_model(MODEL_PATH, use_ema=USE_EMA)

    # YENİ EKLENDİ: Çıktı klasörünü oluştur (eğer yoksa)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"  Çıktı Klasörü: {OUTPUT_DIR}\n")

    noisy_files = sorted([f for f in os.listdir(NOISY_DIR) if f.endswith('.png')])
    gt_files    = sorted([f for f in os.listdir(GT_DIR)    if f.endswith('.png')])

    if len(noisy_files) != 1280 or len(gt_files) != 1280:
        print(f"\nUYARI: 1280 dosya olması gerekirken Noisy: {len(noisy_files)}, GT: {len(gt_files)}\n")
    else:
        print("\n1280 görüntü bulundu. İşlem başlatılıyor...\n")

    psnr_list, ssim_list = [], []

    for idx, f_name in enumerate(noisy_files):
        noisy_path = os.path.join(NOISY_DIR, f_name)
        gt_path    = os.path.join(GT_DIR, f_name)

        noisy_img = io.imread(noisy_path).astype(np.float32) / 255.0
        gt_img    = io.imread(gt_path).astype(np.float32)    / 255.0

        if noisy_img.ndim == 2: noisy_img = np.stack([noisy_img]*3, axis=-1)
        if gt_img.ndim    == 2: gt_img    = np.stack([gt_img]*3,    axis=-1)

        # Çıkarım
        if USE_TTA:
            denoised = denoise_with_tta(model, noisy_img)
        else:
            denoised = denoise_single(model, noisy_img)

        # Resmi SIDD protokolü: uint8'e round et
        if USE_UINT8_ROUND:
            denoised = to_uint8_then_back(denoised)

        # Metrik
        p  = psnr_metric(gt_img, denoised, data_range=1.0)
        ss = ssim_metric(gt_img, denoised, data_range=1.0, channel_axis=2)

        psnr_list.append(p)
        ssim_list.append(ss)

        # YENİ EKLENDİ: Temizlenmiş resmi diske kaydet
        # Float [0, 1] formatındaki görüntüyü standart [0, 255] uint8'e çevirip kaydediyoruz.
        save_img_uint8 = np.clip(np.round(denoised * 255.0), 0, 255).astype(np.uint8)
        save_path = os.path.join(OUTPUT_DIR, f_name)
        io.imsave(save_path, save_img_uint8)

        # Her 32 blok (1 tam sahne) bittiğinde ekrana yazdır
        if (idx + 1) % 32 == 0:
            scene_num = (idx + 1) // 32
            print(f"Sahne {scene_num:02d}/40 | Genel PSNR: {np.mean(psnr_list):.3f} dB | "
                  f"SSIM: {np.mean(ssim_list):.4f}")

    # ── Özet Raporu ──
    final_psnr = np.mean(psnr_list)
    final_ssim = np.mean(ssim_list)
    std_psnr   = np.std(psnr_list)
    std_ssim   = np.std(ssim_list)

    print("\n" + "=" * 65)
    print("  RESMİ SIDD VALIDATION SONUÇLARI (TABLOYA YAZILACAK SKOR)")
    print("=" * 65)
    print(f"  Toplam Blok            : {len(psnr_list)} adet (256x256)")
    print(f"  Yapılandırma           : {'EMA' if USE_EMA else 'RAW'} | "
          f"TTA {'AÇIK' if USE_TTA else 'KAPALI'} | "
          f"uint8 {'AÇIK' if USE_UINT8_ROUND else 'KAPALI'}")
    print(f"  Genel Ortalama PSNR    : {final_psnr:.4f} dB  (std: {std_psnr:.3f})")
    print(f"  Genel Ortalama SSIM    : {final_ssim:.4f}      (std: {std_ssim:.4f})")
    print("=" * 65)

    # Log
    log_path = "official_sidd_validation_results.txt"
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"Veri Seti: Resmi SIDD Validation Set (1280 Blok - PNG Çıktısı)\n")
        f.write(f"Yapılandırma: {'EMA' if USE_EMA else 'RAW'} | "
                f"TTA {'AÇIK' if USE_TTA else 'KAPALI'} | "
                f"uint8 {'AÇIK' if USE_UINT8_ROUND else 'KAPALI'}\n")
        f.write("-" * 50 + "\n")
        f.write(f"Genel Ortalama PSNR : {final_psnr:.4f} dB  (std: {std_psnr:.3f})\n")
        f.write(f"Genel Ortalama SSIM : {final_ssim:.4f}      (std: {std_ssim:.4f})\n")
    print(f"\n  Log dosyası: {os.path.abspath(log_path)}")
    print(f"  Görüntüler başarıyla kaydedildi: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()