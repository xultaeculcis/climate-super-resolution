import os.path as osp
import glob
import cv2
import numpy as np
import torch
import os

from PIL import Image
from torchvision.transforms import transforms
from tqdm import tqdm

from models import Generator

model_path = '../model_weights/RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu
# device = torch.device('cpu')

test_img_folder = '../datasets/lr/*'

model = Generator(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.half()
model = model.to(device)

os.makedirs("../results", exist_ok=True)

print('Model path {:s}. \nTesting...'.format(model_path))

idx = 0
for path in tqdm(sorted(glob.glob(test_img_folder))[:32]):
    idx += 1
    base = osp.splitext(osp.basename(path))[0]
    # read images
    img = Image.open(path).convert("RGB")
    common_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    img_LR = common_transforms(img).unsqueeze(0).half().to(device)

    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1)

    img = transforms.ToPILImage(mode="RGB")(output)
    img.save('../results/{:s}_rlt.png'.format(base), "png")
