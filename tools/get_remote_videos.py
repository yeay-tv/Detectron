#!/usr/bin/env python2

# Copyright (c) 2018-present, Yeay, GmbH.
#
##############################################################################

"""Download videos from the Yeay video feed for processing by the classifier.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import json
import argparse
import socket
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x): return x
try:
    # For Python 3.0 and later
    from urllib.request import urlopen, URLError
except ImportError:
    # Fall back to Python 2's urllib2
    from urllib2 import urlopen, URLError

parser = argparse.ArgumentParser()
parser.add_argument('--feed-loc', type=str, default="https://yeay-api.herokuapp.com/v1/catalog/videos")
parser.add_argument('--save-feed-path', type=str, default=None) #  "/tmp/yeay_video_feed.json"
parser.add_argument('--tmp-path', type=str, default="/datasets/yeay")
parser.add_argument('--page-size', type=int, default=100)
parser.add_argument('--page-number', type=int, default=1)
parser.add_argument('--clobber', action="store_true")
parser.add_argument('--download', action="store_true")
args = parser.parse_args()

# get json with video locations
try:
    feed_uri = "{}?pageSize={}&pageNumber={}".format(args.feed_loc, args.page_size, args.page_number)
    vf = urlopen(feed_uri)
except ValueError:  # invalid URL
    vf = open(args.feed_loc)
json_data = json.loads(vf.read())
vf.close()

# optionally save json feed locally for caching
if args.save_feed_path is not None:
    tmp_dir = os.path.dirname(args.save_feed_path)
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    with open(args.save_feed_path, "w") as fp:
        json.dump(json_data, fp)

# download videos for processing
# first create destination folder if it's not there.
if not os.path.exists(args.tmp_path):
    os.makedirs(args.tmp_path)
if not args.download:
    dst_path = os.path.join(args.tmp_path, "video_url_list.json")
    url_list = [(item["sourceMediaUrl"]) for item in json_data]
    with open(dst_path, 'w') as jsonfile:
        json.dump(url_list, jsonfile, indent=4)
else:
    for item in tqdm(json_data):
        src_uri = item["sourceMediaUrl"]
        src_video_id = item["id"]
        src_name = os.path.basename(src_uri)
        dst_path = os.path.join(args.tmp_path, "videos", src_video_id + "_" + src_name)
        if not os.path.exists(dst_path) or args.clobber:
            try:
                vid_fp = urlopen(src_uri, timeout=30)
                vid_data = vid_fp.read()
                vid_fp.close()
            except URLError:
                print("URLError downloading {}".format(src_uri))
                continue
            except socket.timeout:
                print("socket.timeout error downloading {}".format(src_uri))
                continue
            with open(dst_path, "wb") as dst_fp:
                dst_fp.write(vid_data)
