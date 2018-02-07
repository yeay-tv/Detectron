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

from caffe2.python import workspace

from core.config import assert_and_infer_cfg
from core.config import cfg
from core.config import merge_cfg_from_file
from utils.timer import Timer
import core.test_engine as infer_engine
#import datasets.dummy_datasets as dummy_datasets
from datasets.json_dataset import JsonDataset
import utils.c2 as c2_utils
import utils.logging
import utils.vis as vis_utils

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
        default=None,
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: /tmp/infer_simple)',
        default='/tmp/infer_simple',
        type=str
    )
    parser.add_argument(
        '--input-src',
        dest='input_src',
        help='either a video file path or a device number',
        default='0',
        type=str
    )
    parser.add_argument(
        '--dataset',
        dest='dataset',
        help='dataset',
        default='coco_2014_minival',
        type=str
    )
    parser.add_argument(
        '--thresh',
        dest='thresh',
        help='detection prob threshold',
        default=0.7,
        type=float
    )
    parser.add_argument(
        '--kp-thresh',
        help='keypoint threshold',
        default=2,
        type=int
    )
    parser.add_argument(
        '--output-video',
        help='output example videos',
        action="store_true"
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main(args):
    logger = logging.getLogger(__name__)
    logger.info(cv2.__version__)
    merge_cfg_from_file(args.cfg)
    cfg.TEST.WEIGHTS = args.weights
    cfg.NUM_GPUS = 1
    assert_and_infer_cfg()
    model = infer_engine.initialize_model_from_cfg()
    ds_name = args.dataset
    ds = JsonDataset(ds_name)

    # check if the input_src is an int or a string
    try:
        input_src = int(args.input_src)
    except ValueError:
        input_src = args.input_src

    # opencv3 video file capture adopted from https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html

    cap_dev = cv2.VideoCapture(input_src)
    input_w, input_h, input_fps = int(cap_dev.get(3)), int(cap_dev.get(4)), cap_dev.get(5)
    print(input_w, input_h, input_fps)
    assert isinstance(input_w, int) and isinstance(input_h, int)

    if args.output_video:
        fourcc = cv2.VideoWriter_fourcc(*[str(c) for c in 'XVID'])
        vid_writer = cv2.VideoWriter(os.path.join(args.output_dir, 'output.avi'), fourcc, input_fps, (input_h,input_w))
    i = 0
    while(cap_dev.isOpened()):
        ret, frame = cap_dev.read()
        if i % int(input_fps) == 0:
            logger.info("processing frame {} ({} seconds)".format(i+1, i / input_fps))
        if ret:
            if i % int(input_fps) == 0:
                print("processing video at frame {} ({} seconds)".format(i+1, i/int(input_fps)))
            frame = cv2.transpose(frame)
            with c2_utils.NamedCudaScope(0):
                cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                    model, frame, None
                )

            if args.output_video:
                frame_annotated = vis_utils.vis_one_image_opencv(frame,
                                                                 cls_boxes,
                                                                 dataset=ds,
                                                                 show_box=True,
                                                                 show_class=True,
                                                                 thresh=args.thresh,
                                                                 kp_thresh=args.kp_thresh)
                vid_writer.write(frame_annotated)

            #below does not work with linux
            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break
        else:
            break
        i += 1
    cap_dev.release()
    if args.output_video:
        vid_writer.release()
    print(type(cls_boxes), cls_boxes)
    #cv2.destroyAllWindows()

if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    utils.logging.setup_logging(__name__)
    args = parse_args()
    main(args)
