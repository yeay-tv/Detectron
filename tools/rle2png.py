import argparse
import numpy as np
import pycocotools.mask as mask_util
from PIL import Image
import os
import json

parser = argparse.ArgumentParser(description="JSON with RLE masks to PNG")
parser.add_argument("--output", "-o", type=str, default=None)
parser.add_argument("input", type=str)
args = parser.parse_args()

JSONPATH = args.input
OUTPUT_BASENAME = args.output if args.output else JSONPATH.rsplit(".", 1)[0]

data = json.load(open(JSONPATH))

for fnum, rlemasks in enumerate(data["coco_rle_masks"]):
    print("frame: {}".format(fnum))
    for mnum, rlem in enumerate(rlemasks):
        npmask = mask_util.decode(rlem)
        for i, npi in enumerate(npmask.transpose(2, 0, 1)):
            if i == 0:
                im = np.zeros(npi.shape)
            im[npi > 0] = 1
        im = np.clip(im, 0, 255)
        im = im.astype(np.uint8)
        im = Image.fromarray(im, 'L')
        im.save("{}{}{}.png".format(OUTPUT_BASENAME, fnum, mnum))
        im.close()
