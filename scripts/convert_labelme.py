from glob import glob
from pathlib import Path
from PIL import Image, ImageDraw
import json
import numpy as np
import shutil


root = "/home/torchlab/下载/huahen"

# shutil.rmtree("output/huahen")
target_imgs = Path("output/huahen/train/images")
target_imgs.mkdir(exist_ok=True, parents=True)
target_mask = Path("output/huahen/train/masks")
target_mask.mkdir(exist_ok=True, parents=True)

color_map = [0, 0, 0, 255, 0, 0, 0, 0, 255]


for i, im_file in enumerate(glob(f"{root}/*.jpg")):
    if i == 100:
        break
    # if i < 101:
    #     continue

    # if i == 121:
    #     break
    label_file = Path(im_file).with_suffix(".json")

    with open(label_file) as f:
        label = json.load(f)

    names = []
    im = Image.open(im_file)
    mask = Image.new(mode="L", size=(im.width, im.height))
    # mask.putpalette(color_map)
    mask_draw = ImageDraw.Draw(mask)
    for shape in label["shapes"]:
        label_name = shape["label"]
        points = shape["points"]
        points = np.array(points, dtype=np.float32)
        mask_draw.polygon(points, fill=1)
    mask.save(target_mask.joinpath(label_file.name).with_suffix(".png"))
    shutil.copy(im_file, target_imgs.joinpath(Path(im_file).name))
