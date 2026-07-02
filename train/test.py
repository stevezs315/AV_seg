import argparse
import torch
from pathlib import Path
from torchvision import utils as vutils
from skimage import io
import factories
from transformations import to_torch_tensors, pad_images_unet
import numpy as np
import json

# from CAM.attribution import GradCAM, GradCAMPlusPlus, XGradCAM, BagCAM, ScoreCAM, LayerCAM, AblationCAM, FullGrad, EigenCAM, EigenGradCAM, HiResCAM
# from CAM.attribution.utils import normalize_saliency, visualize_single_saliency


parser = argparse.ArgumentParser()
parser.add_argument('-w', '--weights', type=str, required=True)
parser.add_argument('-t', '--test_name', type=str)
parser.add_argument('-i', '--images_path', type=str, required=True)
parser.add_argument('-m', '--masks_path', type=str, required=True)
parser.add_argument('--save_path', type=str, required=True,
        help='Predictions file name')
parser.add_argument('--save_root', type=str, required=True,
        help='Path to save the predictions')
args = parser.parse_args()
save_path = Path(args.save_root, args.test_name)

if args.test_name is None:
    if 'RITE' in args.images_path:
        if 'test' in args.images_path:
            args.test_name = 'RITE-test'
    elif 'LES-AV' in args.images_path:
        args.test_name = 'LES-AV'
    elif 'HRF' in args.images_path:
        if 'test' in args.images_path:
            args.test_name = 'HRF-test'




print('Loading model')
# 指定要使用的GPU ID
gpu_id = 0

# 使用torch.device来创建一个表示目标设备的对象
device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load(Path(args.weights, 'generator_best.pth'), map_location=device)

print('Loading config')
with open(Path(args.weights, 'config.json'), 'r') as f:
    config = json.load(f)

print('Config:', config)
print(json.dumps(config, indent=4))

# Namespace fron config dict
config = argparse.Namespace(**config)

print('Creating model')
model = factories.ModelFactory().create_class(
    config.model,
    config.in_channels,
    config.out_channels,
    config.base_channels,
    config.num_iterations
)

print('Loading weights')
model.load_state_dict(checkpoint)

if torch.cuda.is_available():
    model.to(device)


if args.save_path is not None:
    save_path = save_path / args.save_path
save_path.mkdir(exist_ok=True, parents=True)

for image_fn in Path(args.images_path).iterdir():
    mask_fn = None
    for mask_fn in Path(args.masks_path).iterdir():
        if mask_fn.stem == image_fn.stem:
            break
    if mask_fn is None:
        print(f'Mask not found for {image_fn}')
        exit(1)
    if image_fn.is_file():
        image = io.imread(image_fn)
        if image.max() > 1:
            image = image / 255.0
        mask = io.imread(mask_fn)
        if mask.max() > 1:
            mask = mask / 255.0
        images, paddings = pad_images_unet([image, mask], return_paddings=True)
        # print('Paddings:', paddings)
        img = images[0]
        padding = paddings[0]
        mask = images[1]
        mask = np.stack([mask,] * 3, axis=2)
        mask_padding = paddings[1]
        # padding format: ((top, bottom), (left, right), (0, 0))
        # print(img.shape, padding)
        tensors = to_torch_tensors([img, mask])
        tensor = tensors[0]
        mask_tensor = tensors[1]
        if torch.cuda.is_available():
            tensor = tensor.to(device)
        else:
            tensor = tensor.cpu()
        tensor = tensor.unsqueeze(0)
        mask_tensor = mask_tensor.unsqueeze(0)
        with torch.no_grad():
            output = model(tensor)
            preds = output[1]
            imgs = output[0]
            if isinstance(preds, list):
                pred = preds[-1]
            else:
                pred = preds
            # print(pred.shape, imgs.shape)
            pred = torch.sigmoid(pred)
            # prob = post_processor(imgs.detach().cpu().numpy(), pred.detach().cpu().numpy())
            # prob = crf.crf_inference_inf(imgs.detach().cpu().numpy(), pred.detach().cpu().numpy())
            pred[mask_tensor < 0.5] = 0
            pred = pred[:, :, padding[0][0]:-padding[0][1], padding[1][0]:-padding[1][1]]
            # print(pred.shape)
            target_fn = save_path / Path(image_fn).name
            vutils.save_image(pred, target_fn)
