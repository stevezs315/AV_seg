import os
import numpy as np
from skimage import io, color, measure
from scipy import ndimage, stats
from tqdm import tqdm

def get_fov_mask(image_rgb, threshold=0.01):
    if len(image_rgb.shape) == 2:
        image_rgb = np.stack((image_rgb,)*3, axis=-1)
    if image_rgb.shape[2] > 3:
        image_rgb = image_rgb[:, :, :3]

    image_lab = color.rgb2lab(image_rgb)
    image_lab[:, :, 0] /= 100.0
    
    mask = image_lab[:, :, 0] >= threshold
    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.filters.median_filter(mask, size=(5, 5))

    connected_components = measure.label(mask).astype(float)
    connected_components[connected_components == mask[0][0]] = np.nan

    largest_component_label = stats.mode(connected_components, axis=None, nan_policy='omit')[0]
    mask = connected_components == largest_component_label

    return mask.astype(np.uint8) * 255

def process_dataset(input_dir, output_dir, threshold=0.01):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return

    print(f"Found {len(image_files)} images, starting mask generation...")

    for img_name in tqdm(image_files, desc="Processing"):
        img_path = os.path.join(input_dir, img_name)
        save_path = os.path.join(output_dir, img_name)

        try:
            img = io.imread(img_path)
            mask = get_fov_mask(img, threshold=threshold)
            io.imsave(save_path, mask, check_contrast=False)
            
        except Exception as e:
            print(f"Error processing {img_name}: {e}")

    print("Processing complete!")

if __name__ == "__main__":
    input_folder = r'/mnt/nasv3/zs/RRWNet_zs/large_field_private/images' 
    output_folder = r'/mnt/nasv3/zs/RRWNet_zs/large_field_private/masks'
    process_threshold = 0.01

    process_dataset(input_folder, output_folder, process_threshold)