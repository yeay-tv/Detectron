# Copyright (c) 2018-present, Yeay, GmbH.
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

"""Implements a LaPlacian Blur Detection.

See: https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import (
    brew,
    core,
    model_helper,
    workspace,
)

import numpy as np

laplacian_kernel = np.zeros((1, 1, 3, 3), dtype=np.float32)
laplacian_kernel[:,:, [0, 2], [1]] = 1.
laplacian_kernel[:,:, [1], [0, 2]] = 1.
laplacian_kernel[:,:, 1, 1] = -4.

def get_lpm_weights(layer_name):
    if "mean_conv_w" in layer_name:
        return np.ones((1, 3, 1, 1), dtype=np.float32) / 3.
    elif "lp_conv_w" in layer_name:
        return laplacian_kernel
    else:
        return None

def build_laplacian(model):
    blobs = []
    mean_conv = brew.conv(model, "data", "mean_conv", 3, 1, 1,
                          lr_mult=0., no_bias=True,
                          weight_init=('ConstantFill', {"value": 1. / 3.}))
    blobs.append(mean_conv)
    lp_conv = brew.conv(model, "mean_conv", "lp_conv", 1, 1, 3,
                        lr_mult=0., no_bias=True,
                        weight_init=('GivenTensorFill', {"values": laplacian_kernel}))
    blobs.append(lp_conv)
    return blobs
