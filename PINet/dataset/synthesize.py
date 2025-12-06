import cv2
from pathlib import Path
from tqdm import tqdm
import shutil
import numpy as np

from add_single_degradation import (
    darken,
    add_noise,
    add_jpeg_comp_artifacts,
    add_haze,
    add_motion_blur,
    add_defocus_blur,
    add_rain,
)

DEGRADATION_TYPE = "rain"

def degrade(img, degradation):
    router = {
        "dark": darken,
        "noise": add_noise,
        "jpeg compression artifact": add_jpeg_comp_artifacts,
        "haze": add_haze,
        "motion blur": add_motion_blur,
        "defocus blur": add_defocus_blur,
        "rain": add_rain,
    }
    if degradation == "haze":
        return add_haze(img)
    return router[degradation](img)


def process_folder(input_root: Path, output_root: Path, degradation: str):
    subfolders = sorted([f for f in input_root.iterdir() if f.is_dir()])
    for sub in tqdm(subfolders, desc="Processing folders", unit="folder"):
        out_sub = output_root / sub.name
        out_sub.mkdir(parents=True, exist_ok=True)

        for file in sub.iterdir():
            if file.suffix.lower() == ".tif":
                img = cv2.imread(str(file), cv2.IMREAD_UNCHANGED)

                if img.dtype == np.uint16:
                    img = (img / 257).astype(np.uint8)  # 65535/255 ≈ 257
                elif img.dtype == np.float32 or img.dtype == np.float64:
                    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)

                img = degrade(img, degradation)
                cv2.imwrite(str(out_sub / file.name), img)
            elif file.suffix.lower() in [".txt", ".hdr"]:
                shutil.copy(file, out_sub / file.name)
            else:
                continue

if __name__ == "__main__":
    input_root = Path("D:/gongzhou/Train_data/data/sig/Test")
    output_root = Path("D:/gongzhou/Train_data/data/sig/Test_rain")
    output_root.mkdir(exist_ok=True)

    print(f"Applying degradation: {DEGRADATION_TYPE}")
    process_folder(input_root, output_root, DEGRADATION_TYPE)
    print("All folders processed successfully!")
