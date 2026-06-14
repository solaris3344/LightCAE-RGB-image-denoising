import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pywt
from skimage import io
from skimage.metrics import peak_signal_noise_ratio as psnr_metric

# ===================== 1. AYARLAR =====================
TRAIN_DIRS = [
    r'A:\iyiler_psnr_3282_sigma_15\DIV2K_train_HR',
    r'A:\iyiler_psnr_3282_sigma_15\pristine_images_color',
    r'A:\iyiler_psnr_3282_sigma_15\Flickr2K',
]

CKPT_NAME     = "ablation_s25_dil2_scratch.pt"   # >>> ABLASYON (E ref): d=2
SCALE_FACTOR  = 10.0
SIGMA_FIXED   = 25 / 255   # >>> ABLASYON: sigma=25 (Tablo 5 protokolu)

DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WAVELET       = 'bior4.4'
NUM_WORKERS   = 4
VAL_COUNT     = 20
REPEAT_FACTOR = 2

# ── Progressive patch & batch ──────────────────────────────
# Aşama 1: epoch  1-60   → patch=64,  batch=32
# Aşama 2: epoch 61-130  → patch=128, batch=16
# Aşama 3: epoch 131-200 → patch=256, batch=8
EPOCHS        = 190
STAGE1_EPOCH  = 60
STAGE2_EPOCH  = 120

PATCH_S1, BATCH_S1 = 64,  32
PATCH_S2, BATCH_S2 = 128, 16
PATCH_S3, BATCH_S3 = 256, 8

LR = 2e-4

# ===================== 2. MİMARİ =====================
# Değişiklikler (v2.1 → v2.2):
#   - BatchNorm kaldırıldı  → PSNR iyileşmesi, genelleşebilirlik
#   - SpatialAttention kaldırıldı → model hafifler, SE zaten yeterli
#   - ch: 96 → 64, n_blocks: 14 → 20  → daha derin ama ince
#   - bias=True tüm conv'larda (BN yoksa bias gerekli)

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
    """
    v2.1'den farklar:
      - BN yok (image restoration'da zararlı, DRUNet makalesi)
      - SpatialAttention yok (SE tek başına yeterli)
      - bias=True (BN olmayınca bias gerekli)
    """
    def __init__(self, ch):
        super().__init__()
        self.body = nn.Sequential(
            # Dilated conv: geniş receptive field
            nn.Conv2d(ch, ch, 3, 1, padding=2, dilation=2,
                      padding_mode='reflect', bias=True),
            nn.GELU(),
            # Standard conv
            nn.Conv2d(ch, ch, 3, 1, padding=1,
                      padding_mode='reflect', bias=True),
            # Channel attention
            SEBlock(ch, reduction=16),
        )

    def forward(self, x):
        return x + self.body(x)


class LightCAE(nn.Module):
    """
    ProfessionalCAE_SOTA v2.2
      ch=64 (96'dan düşürüldü)
      n_blocks=20 (14'ten artırıldı)
      → Tahmini params: ~1.1M  (v2.1: 2.37M)
      → Daha derin ama daha ince
    """
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


# ===================== 3. KAYIP FONKSİYONU =====================
# v2.1: 0.80 L1 + 0.20 FFT
# v2.2: 0.70 L1 + 0.30 FFT  → frekans bilgisi daha ağırlıklı

class FFTLoss(nn.Module):
    def forward(self, pred, target):
        p, t = pred.float(), target.float()
        return torch.mean(torch.abs(
            torch.fft.rfft2(p) - torch.fft.rfft2(t)
        ))


class ImprovedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1  = nn.L1Loss()
        self.fft = FFTLoss()

    def forward(self, pred, target):
        return 0.70 * self.l1(pred, target) + \
               0.30 * self.fft(pred, target)


# ===================== 4. YARDIMCI SINIFLAR =====================

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


def get_files(dirs):
    files = []
    for d in dirs:
        if os.path.exists(d):
            for r, _, f in os.walk(d):
                for n in f:
                    if n.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        files.append(os.path.join(r, n))
    return files


class EMA:
    """
    v2.2: decay 0.9998 → 0.9999 (daha stabil yakınsama)
    """
    def __init__(self, model, decay=0.9999):
        self.model  = model
        self.decay  = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self.backup = None

    @torch.no_grad()
    def update(self):
        msd = self.model.state_dict()
        for k, v in msd.items():
            if v.is_floating_point():
                self.shadow[k].mul_(self.decay).add_(v, alpha=(1 - self.decay))

    def apply_shadow(self):
        self.backup = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        self.model.load_state_dict(self.shadow)

    def restore(self):
        if self.backup:
            self.model.load_state_dict(self.backup)
            self.backup = None


class LazyRGBDataset(Dataset):
    def __init__(self, files, patch_size=PATCH_S1):
        self.files      = files
        self.patch_size = patch_size
        self.virtual_len = len(files) * REPEAT_FACTOR

    def update_patch_size(self, new_size):
        self.patch_size = new_size

    def __len__(self):
        return self.virtual_len

    def __getitem__(self, idx):
        path = self.files[idx % len(self.files)]
        try:
            img = io.imread(path)
            if img.ndim == 2:
                img = np.stack([img]*3, axis=-1)
            img = img[..., :3].astype(np.float32) / 255.0
            h, w, _ = img.shape

            # Pad eğer görüntü küçükse
            if h < self.patch_size or w < self.patch_size:
                img = np.pad(img, (
                    (0, max(0, self.patch_size - h)),
                    (0, max(0, self.patch_size - w)),
                    (0, 0)
                ), mode='reflect')

            # Rastgele kırpma
            y = random.randint(0, img.shape[0] - self.patch_size)
            x = random.randint(0, img.shape[1] - self.patch_size)
            clean = img[y:y+self.patch_size, x:x+self.patch_size, :]

            # Augmentation
            if random.random() < 0.5: clean = np.rot90(clean).copy()
            if random.random() < 0.5: clean = np.fliplr(clean).copy()
            if random.random() < 0.5: clean = np.flipud(clean).copy()  # yeni: dikey flip

            # Gürültülü görüntü
            noisy = np.clip(
                clean + np.random.randn(*clean.shape).astype(np.float32) * SIGMA_FIXED,
                0, 1
            )

            clean_dwt = dwt_rgb(clean)
            noisy_dwt = dwt_rgb(noisy)

            return (
                torch.from_numpy(noisy_dwt * SCALE_FACTOR),
                torch.from_numpy((noisy_dwt - clean_dwt) * SCALE_FACTOR)
            )
        except Exception as e:
            print(f"Hata: {path}: {e}")
            ps = self.patch_size // 2  # DWT sonrası boyut
            return torch.zeros(12, ps, ps), torch.zeros(12, ps, ps)


def validate_internal(model, files, sigma_val):
    model.eval()
    avg_psnr, count = 0, 0
    with torch.no_grad():
        for p in files[:VAL_COUNT]:
            try:
                img = io.imread(p).astype(np.float32) / 255.0
                if img.ndim == 3 and img.shape[2] == 4:
                    img = img[..., :3]
                noisy = np.clip(
                    img + np.random.randn(*img.shape).astype(np.float32) * float(sigma_val),
                    0, 1
                )
                h, w, _ = noisy.shape
                ph = (16 - h % 16) % 16
                pw = (16 - w % 16) % 16
                noisy_pad = np.pad(noisy, ((0, ph), (0, pw), (0, 0)), 'reflect')

                noisy_dwt = dwt_rgb(noisy_pad)
                inp = torch.from_numpy(noisy_dwt * SCALE_FACTOR)[None].to(DEVICE)
                res = model(inp).cpu().numpy()[0]
                denoised = idwt_rgb(noisy_dwt - res / SCALE_FACTOR)[:h, :w]

                avg_psnr += psnr_metric(img, denoised, data_range=1.0)
                count += 1
            except:
                pass
    return avg_psnr / max(count, 1)


# ===================== 5. TRAIN LOOP =====================

def make_loader(ds, batch_size):
    return DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0)
    )


def train():
    # >>> ABLASYON: tekrarlanabilirlik icin sabit tohum (iki kosumda da ayni)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    if DEVICE.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    print("=" * 60)
    print("  ABLASYON E (referans) — sigma=25 SIFIRDAN, dilation=2")
    print("  3 Aşamalı Progressive | EMA=0.9999 | L1+FFT(0.30)")
    print("=" * 60)

    all_files = get_files(TRAIN_DIRS)
    random.shuffle(all_files)
    val_files   = all_files[:VAL_COUNT]
    train_files = all_files[VAL_COUNT:]
    print(f"Eğitim: {len(train_files)} | Validasyon: {len(val_files)}")

    # Model
    model     = LightCAE(in_ch=12, ch=64, n_blocks=20).to(DEVICE)
    ema       = EMA(model, decay=0.9999)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = ImprovedLoss().to(DEVICE)
    scaler    = torch.amp.GradScaler('cuda')

    # Parametre sayısı
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Toplam parametre: {total_params:,} ({total_params/1e6:.2f} M)")

    # Öğrenme hızı planı:
    # Aşama 1 (1-60):   cosine restart
    # Aşama 2 (61-130): cosine restart
    # Aşama 3 (131-200): cosine restart
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=40, T_mult=2, eta_min=1e-6
    )

    # Başlangıç: Aşama 1
    ds = LazyRGBDataset(train_files, patch_size=PATCH_S1)
    dl = make_loader(ds, BATCH_S1)
    current_stage = 1
    print(f"\nAşama 1 başladı: patch={PATCH_S1}, batch={BATCH_S1}")

    best_psnr = 0.0

    for epoch in range(1, EPOCHS + 1):

        # ── Aşama geçişleri ──────────────────────────────────
        if epoch == STAGE1_EPOCH + 1 and current_stage == 1:
            current_stage = 2
            ds.update_patch_size(PATCH_S2)
            dl = make_loader(ds, BATCH_S2)
            print(f"\nAşama 2 başladı: patch={PATCH_S2}, batch={BATCH_S2}")

        if epoch == STAGE2_EPOCH + 1 and current_stage == 2:
            current_stage = 3
            ds.update_patch_size(PATCH_S3)
            dl = make_loader(ds, BATCH_S3)
            print(f"\nAşama 3 başladı: patch={PATCH_S3}, batch={BATCH_S3}")

        # ── Eğitim ──────────────────────────────────────────
        model.train()
        loss_sum = 0

        for i, (inp, tgt) in enumerate(dl):
            inp, tgt = inp.to(DEVICE), tgt.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                pred = model(inp)
                loss = criterion(pred, tgt)

            scaler.scale(loss).backward()

            # Gradient clipping (stabilite için)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
            ema.update()

            scheduler.step(epoch + i / len(dl))
            loss_sum += loss.item()

        # ── Validasyon ──────────────────────────────────────
        ema.apply_shadow()
        val_psnr = validate_internal(model, val_files, SIGMA_FIXED)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Ep {epoch:03d}/{EPOCHS} | "
              f"S{current_stage} | "
              f"Loss: {loss_sum/len(dl):.5f} | "
              f"Val: {val_psnr:.2f} dB | "
              f"LR: {current_lr:.2e}")

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            checkpoint = {
                'model':      model.state_dict(),
                'ema_shadow': ema.shadow,
                'epoch':      epoch,
                'best_psnr':  best_psnr,
                'optimizer':  optimizer.state_dict(),
                'stage':      current_stage,
                # Mimari bilgisi (yeniden yüklemede faydalı)
                'arch': {'in_ch': 12, 'ch': 64, 'n_blocks': 20}
            }
            torch.save(checkpoint, CKPT_NAME)
            print(f"  ✓ Yeni en iyi kaydedildi! ({best_psnr:.2f} dB)")

        ema.restore()

    print(f"\nEğitim tamamlandı. En iyi PSNR: {best_psnr:.2f} dB")


if __name__ == '__main__':
    train()
