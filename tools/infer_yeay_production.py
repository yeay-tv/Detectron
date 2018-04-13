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
import logging
import os
import sys
import time
import json
import errno
import random

from caffe2.python import workspace

from core.config import assert_and_infer_cfg
from core.config import cfg
from core.config import merge_cfg_from_file
from utils.io import cache_url
import core.test_engine as infer_engine
from datasets.json_dataset import JsonDataset
import utils.c2 as c2_utils
import utils.logging
import utils.vis as vis_utils
import utils.yeay as yeay_utils

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
        default='/models/visgen-c37-production/train/yeay/generalized_rcnn/visgen-c37-production.yaml',
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default='/models/visgen-c37-production/train/yeay/generalized_rcnn/model_final.pkl',
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: /tmp/infer_simple)',
        default='/datasets/yeay/output',
        type=str
    )
    parser.add_argument(
        '--input-src',
        dest='input_src',
        help='either a video file path, directory containing video, or json manifest',
        default='/datasets/yeay/video_url_list.json',
        type=str
    )
    parser.add_argument(
        '--dataset',
        dest='dataset',
        help='dataset',
        default='yeay',
        type=str
    )
    parser.add_argument(
        '--thresh',
        dest='thresh',
        help='detection prob threshold',
        default=0.4,
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
    parser.add_argument(
        '--create-vis',
        help='output visuals for development',
        action="store_true"
    )
    parser.add_argument(
        '--rotate',
        dest='rotate',
        help='rotate image',
        action="store_true"
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
        '--shuffle',
        help='shuffle manifest before doing classification',
        action="store_true"
    )
    parser.add_argument(
        '--id-in-filename',
        help='get id from the filename rather than the folder structure',
        action="store_true"
    )
    parser.add_argument(
        '--clobber',
        help='overwrite previous predictions if they are found',
        action="store_true"
    )
    return parser.parse_args()



def main(args):
    logger = logging.getLogger(__name__)
    logger.info(cv2.__version__)
    merge_cfg_from_file(args.cfg)
    cfg.NUM_GPUS = 1
    assert_and_infer_cfg(cache_urls=False)
    args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
    model = infer_engine.initialize_model_from_cfg(args.weights)
    ds_name = args.dataset
    ds = JsonDataset(ds_name)
    try:
        os.makedirs(args.output_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    if os.path.isfile(args.input_src):
        if os.path.splitext(args.input_src)[1] == ".json":
            manifest = json.load(open(args.input_src, 'r'))
            manifest = [f.replace(args.cloudfront_url, args.s3_bucket_path) for f in manifest]
        else:
            manifest = [args.input_src]
    else:
        manifest = [os.path.join(args.input_src, f) for f in os.listdir(args.input_src)]

    if args.shuffle:
        random.shuffle(manifest)

    for video_fn in manifest:
        # check if the input_src is an int or a string
        input_src = video_fn
        input_bn = os.path.basename(video_fn)
        input_ext = os.path.splitext(input_bn)[1]
        input_bn_noext = input_bn.replace(input_ext, "")
        if not args.id_in_filename:
            input_dirs = os.path.dirname(input_src).split("/")
            video_id = input_dirs[-1]
        elif "_" in input_src:
            video_id = input_bn.split("_", 1)[0]
        else:
            video_id = "noid"
        json_filename = os.path.join(args.output_dir, video_id + '.json')
        video_filename = os.path.join(args.output_dir, input_bn_noext + "_ann.avi")
        print(json_filename)

        if not os.path.exists(json_filename) or args.clobber:
            # opencv3 video file capture adopted from https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html
            cap_dev = cv2.VideoCapture(input_src)
            input_w, input_h = int(cap_dev.get(3)), int(cap_dev.get(4))
            input_fps, input_fcnt = cap_dev.get(5), int(cap_dev.get(7))
            print(input_w, input_h, input_fps, input_fcnt)
            assert isinstance(input_w, int) and isinstance(input_h, int)

            if args.output_video:
                fourcc = cv2.VideoWriter_fourcc(*[str(c) for c in 'XVID'])
                vid_writer = cv2.VideoWriter(video_filename, fourcc, input_fps, (input_w,input_h))

            output_json = {
                "video_id": video_id,
                "video_loc": input_src,
                "video_dim": (input_w, input_h),
                "video_fps": input_fps,
                "video_fcnt": input_fcnt,
                "classifiedAt": None,
                "classesInDS": ds.classes,
                #"classifierConfig": args.cfg,
                #"classifierWeights": args.weights,
                "items":[],
            }
            i = 0
            while(cap_dev.isOpened()):
                ret, frame = cap_dev.read()
                if ret:
                    # debugging info
                    if i % int(input_fps) == 0:
                        print("processing video at frame {} ({} seconds), ({})".format(i+1, i/input_fps, frame.shape))
                    # rotate image if desired
                    if args.rotate:
                        frame = cv2.transpose(frame)
                    # do actual inference with network
                    with c2_utils.NamedCudaScope(0):
                        cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                            model, frame, None
                        )
                    # add box info to json dict to be saved later
                    frame_items = yeay_utils.clsboxes2list(i, cls_boxes)
                    if len(frame_items) > 0:
                        output_json["items"].extend(frame_items)
                    # output to video if desired
                    if args.output_video:
                        frame_annotated = vis_utils.vis_one_image_opencv(frame,
                                                                         cls_boxes,
                                                                         dataset=ds,
                                                                         show_box=True,
                                                                         show_class=True,
                                                                         thresh=args.thresh,
                                                                         kp_thresh=args.kp_thresh)
                        vid_writer.write(frame_annotated)
                    # simple frame count
                    i += 1
                # stop grabbing frames
                else:
                    break
            # video classification complete
            output_json["classifiedAt"] = time.strftime("%Y-%m-%jT%H:%M:%S%Z")
            #print(type(cls_boxes), len(cls_boxes), output_json["items"][0], frame_items)
            output_json["top_N_frames"] = yeay_utils.find_best_frames(output_json)
            output_json["top_frames_each_class"] = yeay_utils.find_best_frames_each_class(output_json)
            # release opencv devices
            cap_dev.release()
            if args.output_video:
                vid_writer.release()
            # save json
            if len(output_json["items"]) > 0:
                json.dump(output_json, open(json_filename, 'w'))
            # visual output
            if args.create_vis:
                frames_lists = [output_json["top_N_frames"], output_json["top_frames_each_class"][0]]
                prefixes = ["top_N", "top_each_class"]
                kfd = {}
                for a_i, b_i in zip(frames_lists, prefixes):
                    for a_i_j in a_i:
                        if a_i_j == -1:
                            continue
                        if a_i_j in kfd:
                            kfd[a_i_j].add(b_i)
                        else:
                            kfd[a_i_j] = set([b_i])
                vis_utils.vis_capture_frames_lists(output_json, kfd, crop_bboxes=True, output_dir=args.output_dir)

            #cls_boxes_reconstructed = yeay_utils.list2clsboxes(frame_items, ds.classes)
            #print(len(cls_boxes), cls_boxes[:5])
            #print(len(cls_boxes_reconstructed), cls_boxes_reconstructed[:5])
            #print(cls_boxes_reconstructed == cls_boxes)

if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    utils.logging.setup_logging(__name__)
    args = parse_args()
    main(args)
