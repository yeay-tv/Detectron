#!/usr/bin/env python2

# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time
import json

from caffe2.python import workspace

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging
from detectron.utils.timer import Timer
import detectron.core.test_engine as infer_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils
import detectron.utils.vis as vis_utils

c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default="configs/12_2017_baselines/e2e_mask_rcnn_R-50-FPN_2x.yaml",
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default="https://s3-us-west-2.amazonaws.com/detectron/35859007/12_2017_baselines/e2e_mask_rcnn_R-50-FPN_2x.yaml.01_49_07.By8nQcCH/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl",
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: /tmp/infer_simple)',
        default='/tmp/infer_video_segm',
        type=str
    )
    parser.add_argument(
        '--s3-bucket-path',
        help='path to the s3 bucket',
        type=str,
        default="/datasets/yeay/s3/yeay-user-data-eu"
    )
    parser.add_argument(
        '--cloudfront-url',
        help='cloudfront url to replaced by s3_bucket_path',
        type=str,
        default="https://d9w0wfiu0u1pg.cloudfront.net"
    )
    parser.add_argument(
        '--video-id',
        help='manually specify video id',
        type=str,
        default=None
    )
    parser.add_argument(
        '--class-filter',
        help='only save masks for certain filters',
        type=int,
        nargs='+',
        default=[1]  # 1 = person, see https://github.com/facebookresearch/Detectron/blob/master/detectron/datasets/dummy_datasets.py
    )
    parser.add_argument(
        '--clobber',
        help='overwrite previous predictions if they are found',
        action="store_true"
    )
    parser.add_argument(
        'im_or_folder', help='image or folder of images', default=None
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main(args):
    logger = logging.getLogger(__name__)
    merge_cfg_from_file(args.cfg)
    cfg.NUM_GPUS = 1
    args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)
    model = infer_engine.initialize_model_from_cfg(args.weights)
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()

    im_or_folder = os.path.expanduser(args.im_or_folder)
    if os.path.isfile(im_or_folder):
        if os.path.splitext(im_or_folder)[1] == ".json":
            manifest = json.load(open(im_or_folder, 'r'))
            manifest = [f.replace(args.cloudfront_url, args.s3_bucket_path) for f in manifest]
        else:  # just a video file
            manifest = [im_or_folder]
    elif os.path.isdir(im_or_folder):
        manifest = [os.path.join(im_or_folder, f) for f in os.listdir(im_or_folder)]
    else:
        print("Invalid Input:\n{}".format(im_or_folder))

    output_dir = os.path.expanduser(args.output_dir)
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except:
            pass

    if args.class_filter[0] == -1:
        class_filter = None
    else:
        class_filter = set(args.class_filter)

    for i, video_fn in enumerate(manifest):
        # check if the input_src is an int or a string
        input_src = video_fn
        input_bn = os.path.basename(video_fn)
        input_ext = os.path.splitext(input_bn)[1]
        input_bn_noext = input_bn.replace(input_ext, "")
        input_dirs = os.path.dirname(input_src).split("/")
        if len(input_dirs) > 0:  # video ID is in the dir structure
            video_id = input_dirs[-1]
        elif args.video_id:
            video_id = args.video_id
        else:  # otherwise, just say no idea
            video_id = "noid"
        json_filename = os.path.join(output_dir, video_id + '.json')
        video_filename = os.path.join(output_dir, input_bn_noext + "_ann.avi")

        logger.info('Processing {} -> {}'.format(input_bn, json_filename))

        # Check if json file already exists
        if os.path.exists(json_filename) and not args.clobber:
            output_json = json.load(open(json_filename))
        else:
            output_json = {}

        output_json["coco_rle_masks"] = []

        # create timers
        timers = defaultdict(Timer)
        t = time.time()

        # create video capture device and get some info
        cap_dev = cv2.VideoCapture(input_src)
        input_w, input_h = int(cap_dev.get(3)), int(cap_dev.get(4))
        input_fps, input_fcnt = cap_dev.get(5), int(cap_dev.get(7))
        # start getting frames
        frame_num = 0
        while(cap_dev.isOpened()):
            ret, frame = cap_dev.read()
            if ret:
                # debugging info
                if frame_num % int(input_fps) == 0:
                    logger.info("frame {} ({:.2f} seconds), ({})".format(frame_num+1, float(frame_num+1)/float(input_fps), frame.shape))

                with c2_utils.NamedCudaScope(0):
                    cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                        model, frame, None, timers=timers
                    )

                # filter out extraneous classes
                if class_filter:
                    cls_segms = [segms for i, segms in enumerate(cls_segms) if i in class_filter]
                output_json["coco_rle_masks"].append(cls_segms)

                # simple frame count
                frame_num += 1
            # stop grabbing frames
            else:
                break
        logger.info('Inference time: {:.3f}s'.format(time.time() - t))
        json.dump(output_json, open(json_filename, 'w'))

if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    setup_logging(__name__)
    args = parse_args()
    main(args)
