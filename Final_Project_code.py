import os
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from skimage.measure import label, regionprops
import pandas as pd

input_folder = "/Users/nisargpatel/Desktop/Python Project /reference-cropped-step3/processedImages"  
output_folder = "defect_outputs"

os.makedirs(output_folder, exist_ok=True)

valid_extensions = ('.tif', '.tiff', '.jpg', '.jpeg', '.png')

image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_extensions)]

report = []

def load_image(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    
    
    if ext in ('.tif', '.tiff'):
        img = tiff.imread(filepath)
        if img.ndim == 3:
            img = img[:, :, 0]  
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            img = img.astype(np.uint8)
        return img
    else:
        
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is not None and img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            img = img.astype(np.uint8)
        return img

for filename in image_files:
    filepath = os.path.join(input_folder, filename)

    try:
        image = load_image(filepath)
        if image is None:
            raise ValueError("Image could not be loaded")
    except Exception as e:
        print(f"Skipping {filename} due to error: {e}")
        continue

    if image.ndim == 2 and image.dtype == np.uint8:
        image = cv2.equalizeHist(image)

    mean_val = np.mean(image)
    std_val = np.std(image)
    threshold_val = mean_val + 2 * std_val
    defect_mask = image > threshold_val

    labeled = label(defect_mask)
    regions = regionprops(labeled)

    num_defects = len(regions)
    total_area = sum([r.area for r in regions])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image, cmap='gray')

    for region in regions:
        y, x = region.centroid
        h = region.bbox[2] - region.bbox[0]
        w = region.bbox[3] - region.bbox[1]
        r = max(h, w) / 2
        circle = patches.Circle((x, y), r, edgecolor='blue', facecolor='none', linewidth=2)
        ax.add_patch(circle)

    ax.set_title(f"{filename}\nDefects: {num_defects}, Area: {total_area}")
    ax.axis('off')

    output_path = os.path.join(output_folder, f"marked_{os.path.splitext(filename)[0]}.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Processed {filename} | Defects Found: {num_defects}, Total Area: {total_area}")

    report.append({
        "filename": filename,
        "number_of_defects": num_defects,
        "total_defect_area": total_area
    })
    
df = pd.DataFrame(report)
csv_path = os.path.join(output_folder, "defect_report.csv")
df.to_csv(csv_path, index=False)

print(f"\n CSV report successfully saved at: {csv_path}")