import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def get_frame_list(folder):
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))])

def compute_psnr_curve(baseline_dir, experiment_dir):
    baseline_frames = get_frame_list(baseline_dir)
    exp_frames = get_frame_list(experiment_dir)
    assert baseline_frames == exp_frames, f"Frame files do not match for {experiment_dir}!"

    psnr_list = []
    for fname in baseline_frames:
        img1 = cv2.imread(os.path.join(baseline_dir, fname))
        img2 = cv2.imread(os.path.join(experiment_dir, fname))
        if img1 is None or img2 is None or img1.shape != img2.shape:
            psnr_list.append(np.nan)
            continue
        psnr = calculate_psnr(img1, img2)
        psnr_list.append(psnr)
    return psnr_list

def smooth_curve(psnr_list):
    psnr_array = np.array(psnr_list)
    valid = ~np.isnan(psnr_array)
    if np.sum(valid) < 7:
        return psnr_array
    window = (len(psnr_array) // 15) * 2 + 3
    window = min(window, len(psnr_array) if len(psnr_array)%2 == 1 else len(psnr_array) - 1)
    window = max(window, 5)
    try:
        smoothed = savgol_filter(np.nan_to_num(psnr_array, nan=np.nanmean(psnr_array)), window_length=window, polyorder=2)
        smoothed[~valid] = np.nan
        return smoothed
    except Exception as e:
        print(f"Could not smooth curve: {e}")
        return psnr_array
    
def main(baseline_dir, experiments_dirs, labels=None, imgdir='result.png'):
    plt.figure(figsize=(12, 6))
    colors = plt.cm.get_cmap('tab10', len(experiments_dirs))
    for idx, exp_dir in enumerate(experiments_dirs):
        psnr_list = compute_psnr_curve(baseline_dir, exp_dir)
        x = np.arange(len(psnr_list))
        label = labels[idx] if labels else os.path.basename(exp_dir)
        plt.plot(x, psnr_list, label=f'{label} (raw)', color=colors(idx), linewidth=2)

        # plt.plot(x, psnr_list, label=f'{label} (raw)', color=colors(idx), alpha=0.4, linestyle='--')
        # smoothed = smooth_curve(psnr_list)
        # plt.plot(x, smoothed, label=f'{label} (smoothed)', color=colors(idx), linewidth=2)

        avg_psnr = np.nanmean(psnr_list)
        print(f"{label}: Average PSNR = {avg_psnr:.2f} dB")
    plt.xlabel('Frame')
    plt.ylabel('PSNR (dB)')
    plt.title('Frame-wise PSNR Comparison vs Baseline')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(imgdir)
    plt.show()


if __name__ == "__main__":
    baseline_dir = "outputs/snow/baseline"

    # List experiment directories
    experiments_dirs = [
        "outputs/snow/substep_1e-5",
        "outputs/snow/substep_1e-6",
    ]

    # (Optional) Give custom labels for legend
    labels = [
        "substeps = 1e-5",
        "substeps = 1e-6",
    ]
    imgdir = "outputs/snow/substep.png"
    main(baseline_dir, experiments_dirs, labels, imgdir)