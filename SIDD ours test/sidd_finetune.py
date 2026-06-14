import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
import pywt
from skimage import io
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
import warnings

warnings.filterwarnings("ignore")

# ===================== 1. AYARLAR =====================
# SIDD Small Data klasör yolu (İçinde 0001_001_S6_... gibi klasörler olan dizin)
SIDD_DIR = r'...path\SIDD_Small_sRGB_Only\Data'

# Başlangıç ağırlıkları:
# SIDD'deki gerçek gürültü genelde σ=15-30 aralığında olduğu için sigma=25 ile
# eğitilmiş checkpoint sigma=15'ten biraz daha iyi başlangıç olabilir.
# Eğer sigma=25 checkpoint'in varsa onun ismini buraya yaz.
RESUME_CHECKPOINT = "rgb_model_v22_sigma25_best.pt"   # ideal: sigma=25 modeli

# Çıktı model ismi
OUTPUT_MODEL_NAME = "rgb_model_v22_sidd_finetuned50.pt"

SCALE_FACTOR  = 10.0
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WAVELET       = 'bior4.4'

# Eğitim Hiperparametreleri
BATCH_SIZE    = 8          # Patch büyüdüğü için 16'dan düşürüldü (VRAM)
PATCH_SIZE    = 256       # 128'den artırıldı - daha geniş bağlam, +0.3-0.7 dB beklenir
EPOCHS        = 50        # 20'den artırıldı - loss hâlâ düşüyordu, erken kesilmişti
LR_MAX        = 1e-4
LR_MIN        = 1e-6
WARMUP_EPOCHS = 2          # Transfer öğrenmede ilk patlamayı bastırır

# Her epoch'ta her resimden kaç random patch alınsın
PATCHES_PER_IMAGE = 10     # 20'den düşürüldü - 50 epoch ile dengeli

# EMA decay - fine-tune için (önceden 0.9999, çok yavaş sızıyordu)
EMA_DECAY     = 0.999      # ~700 adım yarı ömür

# ===================== 2. MİMARİ & KAYIP FONKSİYONLARI =====================
# (Pre-trained checkpoint ile uyumlu olması için mimari aynı kalmalı: ch=64, n_blocks=20)

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

class FFTLoss(nn.Module):
    def forward(self, pred, target):
        with torch.amp.autocast('cuda', enabled=False):
            pred_fp32 = pred.float()
            target_fp32 = target.float()
            pred_fft = torch.fft.rfft2(pred_fp32)
            target_fft = torch.fft.rfft2(target_fp32)
            return torch.mean(torch.abs(pred_fft - target_fft))

class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_avg = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_avg.clone()
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]

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


# ===================== 3. SIDD İÇİN VERİ SETİ =====================

class SIDDDataset(Dataset):
    def __init__(self, sidd_dir, patch_size=192, is_val=False, patches_per_image=10):
        self.patch_size = patch_size
        self.is_val = is_val
        self.patches_per_image = patches_per_image
        self.pairs = []

        folders = sorted([f for f in os.listdir(sidd_dir) if os.path.isdir(os.path.join(sidd_dir, f))])

        # 160 sahne. Son 10 tanesi validasyon, kalanı eğitim.
        if self.is_val:
            folders = folders[-10:]
        else:
            folders = folders[:-10]

        for folder in folders:
            folder_path = os.path.join(sidd_dir, folder)
            noisy_files = [f for f in os.listdir(folder_path) if "NOISY" in f and f.endswith(".PNG")]
            gt_files = [f for f in os.listdir(folder_path) if "GT" in f and f.endswith(".PNG")]
            if noisy_files and gt_files:
                self.pairs.append((os.path.join(folder_path, noisy_files[0]),
                                   os.path.join(folder_path, gt_files[0])))

    def __len__(self):
        return len(self.pairs) * (1 if self.is_val else self.patches_per_image)

    def __getitem__(self, idx):
        real_idx = idx % len(self.pairs)
        noisy_path, gt_path = self.pairs[real_idx]

        img_noisy = io.imread(noisy_path).astype(np.float32) / 255.0
        img_gt = io.imread(gt_path).astype(np.float32) / 255.0

        if img_noisy.ndim == 2:
            img_noisy = np.stack([img_noisy]*3, axis=-1)
            img_gt = np.stack([img_gt]*3, axis=-1)

        img_noisy = img_noisy[..., :3]
        img_gt = img_gt[..., :3]

        h, w, _ = img_noisy.shape

        if self.is_val:
            th, tw = 512, 512
            i, j = (h - th) // 2, (w - tw) // 2
        else:
            th, tw = self.patch_size, self.patch_size
            i = random.randint(0, h - th)
            j = random.randint(0, w - tw)

        patch_noisy = img_noisy[i:i+th, j:j+tw, :]
        patch_gt = img_gt[i:i+th, j:j+tw, :]

        # Veri Artırma
        if not self.is_val:
            if random.random() < 0.5:
                patch_noisy = np.flip(patch_noisy, axis=0)
                patch_gt = np.flip(patch_gt, axis=0)
            if random.random() < 0.5:
                patch_noisy = np.flip(patch_noisy, axis=1)
                patch_gt = np.flip(patch_gt, axis=1)
            rot = random.choice([0, 1, 2, 3])
            if rot > 0:
                patch_noisy = np.rot90(patch_noisy, k=rot, axes=(0, 1))
                patch_gt = np.rot90(patch_gt, k=rot, axes=(0, 1))

        # DWT Dönüşümü - Model residual öğreniyor: R = Noisy - Clean
        x_dwt = dwt_rgb(patch_noisy.copy())
        y_dwt = dwt_rgb(patch_gt.copy())

        target_res = x_dwt - y_dwt

        inp_tensor = torch.from_numpy(x_dwt * SCALE_FACTOR)
        target_tensor = torch.from_numpy(target_res * SCALE_FACTOR)

        if self.is_val:
            return inp_tensor, target_tensor, torch.from_numpy(patch_noisy.copy()), torch.from_numpy(patch_gt.copy())

        return inp_tensor, target_tensor


# ===================== 4. VALİDASYON FONKSİYONU =====================
def validate_sidd(model, val_loader):
    model.eval()
    psnr_total = 0.0
    with torch.no_grad():
        for inp_tensor, _, patch_noisy, patch_gt in val_loader:
            inp_tensor = inp_tensor.to(DEVICE)
            patch_noisy = patch_noisy.numpy()[0]
            patch_gt = patch_gt.numpy()[0]

            res_pred = model(inp_tensor).cpu().numpy()[0]

            x_dwt = dwt_rgb(patch_noisy)
            out_dwt = x_dwt - (res_pred / SCALE_FACTOR)
            out_spatial = idwt_rgb(out_dwt)

            psnr_val = psnr_metric(patch_gt, out_spatial, data_range=1.0)
            psnr_total += psnr_val

    model.train()
    return psnr_total / len(val_loader)


# ===================== 5. LR SCHEDULER (Warmup + Cosine) =====================
def get_lr(epoch, total_epochs, warmup_epochs, lr_max, lr_min):
    """Warmup + Cosine annealing - transfer öğrenmede ilk gradient patlamasını yumuşatır."""
    if epoch <= warmup_epochs:
        return lr_max * (epoch / max(1, warmup_epochs))
    # Cosine annealing
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    cosine_factor = 0.5 * (1 + np.cos(np.pi * progress))
    return lr_min + (lr_max - lr_min) * cosine_factor


# ===================== 6. EĞİTİM (TRANSFER) DÖNGÜSÜ =====================

def main():
    print(f"Transfer Öğrenme Başlıyor (SIDD Veri Seti)")
    print(f"Cihaz: {DEVICE}")
    print(f"Patch: {PATCH_SIZE} | Batch: {BATCH_SIZE} | Epoch: {EPOCHS} | Warmup: {WARMUP_EPOCHS}")
    print(f"EMA Decay: {EMA_DECAY}")

    # Veri Seti ve Loader
    train_dataset = SIDDDataset(SIDD_DIR, patch_size=PATCH_SIZE, is_val=False,
                                patches_per_image=PATCHES_PER_IMAGE)
    val_dataset = SIDDDataset(SIDD_DIR, is_val=True)

    train_dl = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_dataset, batch_size=1, shuffle=False)

    print(f"Eğitim çiftleri: {len(train_dataset.pairs)} | Validasyon çiftleri: {len(val_dataset.pairs)}")
    print(f"Adım/epoch: {len(train_dl)} | Toplam adım: {len(train_dl) * EPOCHS}")

    model = LightCAE(in_ch=12, ch=64, n_blocks=20).to(DEVICE)

    # Önceden Eğitilmiş AWGN Modelini Yükle
    if os.path.exists(RESUME_CHECKPOINT):
        print(f"AWGN Checkpoint Yükleniyor: {RESUME_CHECKPOINT}")
        ckpt = torch.load(RESUME_CHECKPOINT, map_location=DEVICE, weights_only=False)
        if 'ema_shadow' in ckpt:
            model.load_state_dict(ckpt['ema_shadow'])
            print("  -> EMA shadow ağırlıkları yüklendi.")
        else:
            model.load_state_dict(ckpt['model'])
            print("  -> Model ağırlıkları yüklendi.")
    else:
        print("DİKKAT: Transfer edilecek model bulunamadı, sıfırdan başlıyor!")

    ema = EMA(model, decay=EMA_DECAY)

    criterion_l1 = nn.L1Loss()
    criterion_fft = FFTLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_MAX, weight_decay=1e-4)
    scaler = GradScaler('cuda')

    best_psnr = 0.0

    model.train()
    for epoch in range(1, EPOCHS + 1):
        # Manuel LR güncelleme (warmup + cosine)
        current_lr = get_lr(epoch, EPOCHS, WARMUP_EPOCHS, LR_MAX, LR_MIN)
        for pg in optimizer.param_groups:
            pg['lr'] = current_lr

        loss_sum = 0.0
        for inp, target in train_dl:
            inp, target = inp.to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()
            with autocast('cuda'):
                pred = model(inp)
                loss_l1 = criterion_l1(pred, target)
                loss_f = criterion_fft(pred, target)
                loss = 0.70 * loss_l1 + 0.30 * loss_f

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()
            ema.update()
            loss_sum += loss.item()

        # ÇİFT VALİDASYON: hem ham model hem EMA shadow ile ölç
        val_psnr_raw = validate_sidd(model, val_dl)

        ema.apply_shadow()
        val_psnr_ema = validate_sidd(model, val_dl)
        ema.restore()

        # Hangisi daha iyiyse onu kullan
        val_psnr = max(val_psnr_raw, val_psnr_ema)
        best_source = "EMA" if val_psnr_ema >= val_psnr_raw else "RAW"

        print(f"Epoch {epoch:02d}/{EPOCHS} | Loss: {loss_sum/len(train_dl):.5f} | "
              f"Raw: {val_psnr_raw:.2f} dB | EMA: {val_psnr_ema:.2f} dB | "
              f"LR: {current_lr:.2e}")

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            save_state = ema.shadow if best_source == "EMA" else model.state_dict()

            torch.save({
                'model': save_state,
                'ema_shadow': ema.shadow,
                'raw_state': model.state_dict(),
                'epoch': epoch,
                'best_psnr': best_psnr,
                'best_source': best_source,
                'arch': {'in_ch': 12, 'ch': 64, 'n_blocks': 20}
            }, OUTPUT_MODEL_NAME)
            print(f"  -> Yeni en iyi model kaydedildi! ({best_source} | {best_psnr:.2f} dB)")

if __name__ == '__main__':
    main()