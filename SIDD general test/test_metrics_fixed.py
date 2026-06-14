import cv2
import os
import re
import numpy as np
from skimage.metrics import structural_similarity as ssim

# --- DOSYA YOLLARI ---
base_path = r"...path\SIDD_general_test"
paths = {
    'gt':        os.path.join(base_path, "GT"),
    'our':       os.path.join(base_path, "Denoised"),
    'mirnet':    os.path.join(base_path, "mirnet_output"),
    'mprnet':    os.path.join(base_path, "Mprnet"),
    'restormer': os.path.join(base_path, "Restormer_Denoised"),
    'uformer':   os.path.join(base_path, r"uformer_png\png"),
    'nafnet':    os.path.join(base_path, "NAFNet_Results"),
    'ridnet':    os.path.join(base_path, "RIDNet_renamed"),
    'deamnet':   os.path.join(base_path, "Deam_Denoised"),
}
output_file  = os.path.join(base_path, "denoising_full_metrics_fixed.txt")
mismatch_log = os.path.join(base_path, "mismatch.log")

def imread_robust(path):
    if path is None or not os.path.exists(path):
        return None
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)

def extract_numbers(filename):
    return tuple(str(int(n)) for n in re.findall(r'\d+', filename))

def get_metrics(gt, out, fname, model, logf):
    """Protokol: float PSNR + kanal-ortalama SSIM, data_range=255. Resize YOK."""
    if gt is None or out is None:
        return None, None
    if gt.shape != out.shape:
        logf.write(f"BOYUT UYUSMAZLIGI atlandi: {model} / {fname} "
                   f"gt={gt.shape} out={out.shape}\n")
        return None, None
    g = gt.astype(np.float64)
    o = out.astype(np.float64)
    mse = np.mean((g - o) ** 2)
    psnr = 100.0 if mse == 0 else 20.0 * np.log10(255.0 / np.sqrt(mse))
    # kanal-ortalama SSIM
    s = np.mean([ssim(g[..., c], o[..., c], data_range=255) for c in range(g.shape[2])])
    return psnr, float(s)

def build_file_map(folder):
    fm = {}
    if not os.path.exists(folder):
        return fm
    for f in os.listdir(folder):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            nums = extract_numbers(f)
            if nums:
                fm[nums] = os.path.join(folder, f)
    return fm

def find_path(fmap, key):
    p = fmap.get(key)
    if p:
        return p
    # gevsek eslesme: tum sayilar anahtarda geciyorsa
    for k, path in fmap.items():
        if all(num in k for num in key):
            return path
    return None

def main():
    model_list = [m for m in paths if m != 'gt']
    print("Klasorler taraniyor...")
    maps = {m: build_file_map(paths[m]) for m in model_list}

    gt_files = sorted(f for f in os.listdir(paths['gt'])
                      if f.lower().endswith(('.png', '.jpg', '.jpeg')))
    print(f"Toplam {len(gt_files)} GT goruntusu.")

    # her goruntu icin tum modellerin (psnr,ssim) sonucu
    per_image = {}   # fname -> {model: (p,s) veya None}
    coverage  = {m: 0 for m in model_list}

    with open(mismatch_log, "w", encoding="utf-8") as logf:
        for idx, fname in enumerate(gt_files, 1):
            gt = imread_robust(os.path.join(paths['gt'], fname))
            if gt is None:
                continue
            key = extract_numbers(fname)
            row = {}
            for m in model_list:
                mp = find_path(maps[m], key)
                out = imread_robust(mp)
                p, s = get_metrics(gt, out, fname, m, logf)
                row[m] = (p, s) if p is not None else None
                if p is not None:
                    coverage[m] += 1
            per_image[fname] = row
            if idx % 50 == 0:
                print(f"  {idx}/{len(gt_files)}")

    # --- ORTAK KUME: tum modellerin gecerli oldugu goruntuler ---
    common = [f for f, row in per_image.items()
              if all(row[m] is not None for m in model_list)]
    print(f"\nOrtak kume (tum modeller gecerli): {len(common)} / {len(gt_files)} goruntu")

    def avg_over(files):
        out = {}
        for m in model_list:
            ps = [per_image[f][m][0] for f in files if per_image[f][m] is not None]
            ss = [per_image[f][m][1] for f in files if per_image[f][m] is not None]
            out[m] = (np.mean(ps), np.mean(ss), len(ps)) if ps else (None, None, 0)
        return out

    avg_common = avg_over(common)          # adil karsilastirma (rapor edilecek)
    avg_all    = avg_over(list(per_image)) # bilgi amacli (her model kendi kapsamasinda)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("PROTOKOL: float PSNR, kanal-ortalama SSIM (data_range=255), resize YOK, uint8 PNG.\n")
        f.write(f"GT goruntu sayisi: {len(gt_files)} | Ortak kume: {len(common)}\n\n")

        f.write("--- KAPSAMA (her model kac goruntude sonuc uretti) ---\n")
        for m in model_list:
            f.write(f"{m.upper():<10}: {coverage[m]}/{len(gt_files)}\n")

        f.write("\n=== ADIL ORTALAMA (yalnizca ortak kume, N=%d) ===\n" % len(common))
        for m in model_list:
            p, s, n = avg_common[m]
            if p is not None:
                f.write(f"{m.upper():<10} -> PSNR: {p:6.2f} dB | SSIM: {s:.4f}  (N={n})\n")
            else:
                f.write(f"{m.upper():<10} -> sonuc yok\n")

        f.write("\n=== BILGI: TUM KAPSAMA ORTALAMASI (modeller farkli N olabilir) ===\n")
        for m in model_list:
            p, s, n = avg_all[m]
            if p is not None:
                f.write(f"{m.upper():<10} -> PSNR: {p:6.2f} dB | SSIM: {s:.4f}  (N={n})\n")

        # detayli satirlar
        f.write("\n--- DETAY (ortak kume) ---\n")
        header = f"{'Image':<26} | " + " | ".join(f"{m.upper():^13}" for m in model_list)
        f.write(header + "\n" + "-" * len(header) + "\n")
        for fname in sorted(common):
            row = per_image[fname]
            cells = " | ".join(f"{row[m][0]:5.2f}/{row[m][1]:.3f}" for m in model_list)
            f.write(f"{fname:<26} | {cells}\n")

    print(f"\nBitti. Sonuc: {output_file}")
    print(f"Boyut uyusmazliklari (varsa): {mismatch_log}")
    print("\nADIL ORTALAMA (ortak kume):")
    for m in model_list:
        p, s, n = avg_common[m]
        if p is not None:
            print(f"  {m.upper():<10} PSNR {p:.2f} dB | SSIM {s:.4f} (N={n})")

if __name__ == '__main__':
    main()
