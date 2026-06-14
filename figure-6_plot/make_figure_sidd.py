# -*- coding: utf-8 -*-
"""
make_figure_sidd.py — SIDD nitel karsilastirma + hata haritasi (Madde 6 & 11 / Figure X)

Dosya adlandirma beklentisi (senin klasorundeki gibi):
    gt_<SAHNE>.png, noisy_<SAHNE>.png, mirnet_<SAHNE>.png, mprnet_<SAHNE>.png,
    uformer_<SAHNE>.png, nafnet_<SAHNE>.png, ridnet_<SAHNE>.png,
    restormer_<SAHNE>.png, ours_<SAHNE>.png
ornek SAHNE etiketi: "39_02"

KULLANIM:
    1) Asagidaki ROOT yolunu ve SCENES listesini doldur.
    2) Her sahne icin yukaridaki 9 dosya ayni klasorde olmali (en az gt + yontemler).
       noisy_* yoksa SHOW_NOISY = False yap.
    3) pip install numpy pillow matplotlib scikit-image
    4) python make_figure_sidd.py
Cikti: figure_X_sidd_qualitative.png (300 dpi) + .pdf
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

# ============================ CONFIG ============================
ROOT = r"D:\Yeni klasör\benim\editor"   # PNG'lerin bulundugu klasor

# Gostermek istedigin sahneler (alt satirlar). Her biri icin dosyalar ROOT'ta olmali.
# En az 2-3 sahne onerilir: biri BASARISIZLIK (yogun doku), biri basari, biri renk/metin.
SCENES = [
    "39_02",            # senin verdigin basarisizlik ornegi (yogun tekrarli doku)
    # "12_05",          # <-- varsa bir basari ornegi ekle
    # "27_11",          # <-- varsa bir renk-kaymasi/metin ornegi ekle
]

# Yontem adi -> dosya oneki (sira = sutun sirasi). Makaledeki Tablo/Sekil sirasiyla ayni tut.
METHODS = {
    "MIRNet":    "mirnet",
    "MPRNet":    "mprnet",
    "Uformer":   "uformer",
    "NAFNet":    "nafnet",
    "RIDNet":    "ridnet",
    "Restormer": "restormer",
    "Ours":      "ours",
}

SHOW_NOISY = True       # noisy_<SAHNE>.png yoksa False yap
ERR_VMAX   = 0.08       # hata haritalari ortak ust sinir ([0,1]); soluksa 0.08, tasiyorsa 0.18
ERR_CMAP   = "inferno"
DPI        = 300
OUT_BASE   = os.path.join(ROOT, "figure_X_sidd_qualitative")
LF         = 8          # etiket font boyutu
# ================================================================

def load(path):
    return np.asarray(Image.open(path).convert("RGB"), np.float64) / 255.0

def u8(a):
    return np.round(np.clip(a, 0, 1) * 255.0) / 255.0

def metrics(gt, out):
    g, o = u8(gt), u8(out)
    p = psnr_fn(g, o, data_range=1.0)
    s = float(np.mean([ssim_fn(g[..., c], o[..., c], data_range=1.0) for c in range(3)]))
    return p, s

def fpath(prefix, scene):
    return os.path.join(ROOT, f"{prefix}_{scene}.png")

def main():
    method_names = list(METHODS.keys())
    n_img_cols = (1 if SHOW_NOISY else 0) + 1 + len(method_names)   # [Noisy] + GT + yontemler
    rows = 2 * len(SCENES)                                          # her sahne: zoom + hata
    fig, axes = plt.subplots(rows, n_img_cols,
                             figsize=(1.5 * n_img_cols, 1.7 * rows))
    if rows == 1:
        axes = axes[None, :]

    for si, scene in enumerate(SCENES):
        gt = load(fpath("gt", scene))
        r_img, r_err = 2 * si, 2 * si + 1

        col = 0
        # GT
        ax = axes[r_img, col]; ax.imshow(gt); ax.set_title("GT", fontsize=LF)
        axes[r_err, col].axis("off")   # GT'nin hata haritasi yok (referans)
        col += 1

        # Noisy
        if SHOW_NOISY:
            noisy = load(fpath("noisy", scene))
            pe, se = metrics(gt, noisy)
            ax = axes[r_img, col]; ax.imshow(noisy)
            ax.set_title(f"Noisy\n{pe:.2f}/{se:.3f}", fontsize=LF)
            err = np.abs(noisy - gt).mean(2)
            axes[r_err, col].imshow(err, cmap=ERR_CMAP, vmin=0, vmax=ERR_VMAX)
            col += 1

        # Yontemler
        for name in method_names:
            out = load(fpath(METHODS[name], scene))
            p, s = metrics(gt, out)
            ax = axes[r_img, col]; ax.imshow(out)
            title = f"{name}\n{p:.2f}/{s:.3f}"
            # Ours'u vurgula
            ax.set_title(title, fontsize=LF,
                         fontweight=("bold" if name == "Ours" else "normal"))
            err = np.abs(out - gt).mean(2)
            m = axes[r_err, col].imshow(err, cmap=ERR_CMAP, vmin=0, vmax=ERR_VMAX)
            col += 1

        # satir etiketi (sol)
        axes[r_img, 0].set_ylabel(f"Scene {scene}", fontsize=LF, rotation=90, labelpad=2)

    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])

    fig.subplots_adjust(wspace=0.05, hspace=0.30, right=0.90)
    cax = fig.add_axes([0.915, 0.15, 0.013, 0.7])
    cb = fig.colorbar(m, cax=cax); cb.set_label("Mean absolute error", fontsize=LF)
    cb.ax.tick_params(labelsize=LF - 1)

    fig.savefig(OUT_BASE + ".png", dpi=DPI, bbox_inches="tight")
    fig.savefig(OUT_BASE + ".pdf", bbox_inches="tight")
    print("Kaydedildi:", OUT_BASE + ".png / .pdf")

if __name__ == "__main__":
    main()
