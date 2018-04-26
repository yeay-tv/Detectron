"""Yeay-Specific Utilities"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import os
import json


def clsboxes2list(frame_num, cls_boxes):
    l = []
    for cls_id, cls_array in enumerate(cls_boxes):
        for box in cls_array:
            prob = float(box[4])
            bbox = box[:4].tolist()
            l.append((frame_num, cls_id, prob, bbox))
    return l

def list2clsboxes(l, classes):
    num_classes = len(classes)
    # the cls_boxes seems to always begin with an empty list
    cls_boxes = [[]] + [np.zeros(shape=(0, 5), dtype=np.float32) for _ in range(1, num_classes)]
    for item in l:
        frame_num, cls_id, prob, bbox = item
        cls_item = bbox + [prob]
        cls_item = np.array(cls_item, dtype=np.float32).reshape(1, -1)
        cls_boxes[cls_id] = np.r_[cls_boxes[cls_id], cls_item]
    return cls_boxes

def find_best_frames(jsondata, N = 5, thresh=0.0, criteria="prob", smooth=False):
    """Find the N best frames in the videos.

        First we will get the top 30 items found by probability and then select the
        N frames with the highest top 30 summed probability.

        Arguments:
            N: (int) number of best frames to find
            criteria: (str) type of criteria to use
            smooth: (bool) smooth amounts across frames
    """
    K = 30
    num_frames = jsondata["video_fcnt"]
    num_classes = len(jsondata["classesInDS"])
    items = jsondata["items"]
    top_K = np.zeros(shape=(K, num_frames, 6), dtype=float) # (K, num_frames, (score, cls_id, bbox))

    for item in items:
        if not item:
            continue
        frame_index = item[0] - 1
        cls_id = item[1]
        prob = item[2]
        bbox = item[3]

        # skip if probability is below the threshold
        if thresh > prob:
            continue

        """Top K Calculation by Probability

        This first checks to see how many items are in our top K.  If the number is less than
        K, then we put our item into that spot.  If it is more than K, then we check to see if
        the new probability is greater than the lowest existing probabilty.  If it is greater,
        then we replace the lowest item with the new item.
        """

        items_in_frame = np.count_nonzero(top_K[:, frame_index, 0])
        min_prob_pos = top_K[:, frame_index, 0].argmin() if items_in_frame == K else items_in_frame
        min_prob = top_K[min_prob_pos, frame_index, 0]
        if prob > min_prob:
            new_item = [prob] + [cls_id] + bbox
            top_K[min_prob_pos, frame_index, :] = new_item

    # Sort
    # Note to self: figure out how to do this without transposing
    top_K_sort_idx = np.argsort(-top_K[:,:,0].T, axis=1)  # (K, frame_index)
    top_K = top_K.transpose(1, 0, 2)[np.arange(top_K.shape[1])[:,np.newaxis], top_K_sort_idx].transpose(1, 0, 2)

    if smooth:
        top_K_probs = smoother(top_K[:,:,0])
        top_K[:,:,0] = top_K_probs

    n_top_idx = top_K[:,:,0].sum(axis=0).argsort()[-N:].tolist()

    return n_top_idx

def find_best_frames_each_class(jsondata, thresh=0.0, criteria="prob*sqrt(area)", smooth=False):
    """Find frame which has the best example of each in the videos.

        We subjectively define "best" as the highest probability of the class times
        the square root of the area of the

        Arguments:
            N: (int) number of best frames to find
            criteria: (str) type of criteria to use
            smooth: (bool) smooth amounts across frames
    """
    num_frames = jsondata["video_fcnt"]
    num_classes = len(jsondata["classesInDS"])
    vid_area = np.array(jsondata["video_dim"]).prod()
    items = jsondata["items"]
    top_classes = np.zeros(shape=(num_classes, 2), dtype=float) # (classes, (frame_index, score = prob * sqrt of area))
    top_classes[:, 0] = -1.  # so we know which classes weren't found

    top_bboxes = [None] * num_classes

    blur_scores = np.array(jsondata["blur_scores"]).flatten()

    for item in items:
        if not item:
            continue
        frame_index = item[0] - 1
        cls_id = item[1]
        prob = item[2]
        bbox = item[3]
        # skip if probability is below the threshold
        if thresh > prob:
            continue

        area = np.float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) / vid_area
        # calculate a score based on probability, area, and blurriness
        score_curr = np.float(prob) * np.sqrt(area) * blur_scores[frame_index]
        score_top = top_classes[cls_id, 1]
        if score_curr > score_top:
            top_classes[cls_id, 0] = frame_index
            top_classes[cls_id, 1] = score_curr
            top_bboxes[cls_id] = bbox


    return top_classes[:, 0].astype(np.int).tolist(), top_bboxes


def smoother(arr, win_len=9):
    win = np.hanning(win_len+2)[1:-1]  # numpy windows have zeros on end
    if arr.ndim > 1:
        arr_smooth = np.array([np.convolve(cct, win/win.sum(), 'same') for cct in arr])
    else:
        arr_smooth = np.convolve(arr, win/win.sum(), 'same')
    return arr_smooth


def normalize_blur_scores(blur_scores, f=np.sqrt):
    """ take a list of lists and output a normalized list of lists """
    blur_scores = np.array(blur_scores)
    blur_scores = f(blur_scores / blur_scores.max())
    return blur_scores.tolist()
