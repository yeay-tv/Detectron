# Use Caffe2 image as parent image
FROM caffe2/caffe2:snapshot-py2-cuda9.0-cudnn7-ubuntu16.04

RUN mv /usr/local/caffe2 /usr/local/caffe2_build
ENV Caffe2_DIR /usr/local/caffe2_build

ENV PYTHONPATH /usr/local/caffe2_build:${PYTHONPATH}
ENV LD_LIBRARY_PATH /usr/local/caffe2_build/lib:${LD_LIBRARY_PATH}

# Clone the Detectron repository and link datasets
RUN git clone --branch yeay_newstruc https://github.com/yeay-tv/detectron /detectron
RUN ln -s /datasets/yeay /detectron/detectron/datasets/data/yeay

# Install Python dependencies
RUN pip install -r /detectron/requirements.txt
RUN pip install tqdm

# Install the COCO API
RUN git clone https://github.com/cocodataset/cocoapi.git /cocoapi
WORKDIR /cocoapi/PythonAPI
RUN make install

# Set up Detectron-specific python modules
WORKDIR /detectron
RUN make
RUN make ops

# extract datasets and models into docker image
#ADD yeay.tar.gz /datasets #  going to mount this as a volume
#RUN curl -o /datasets/yeay/videos/Pks2RQhXu9_c3fab014-4b22-488c-b0a1-46453fdc579f.mp4 https://d9w0wfiu0u1pg.cloudfront.net/lQ6iasLj5Y/videos/Pks2RQhXu9/c3fab014-4b22-488c-b0a1-46453fdc579f.mp4
ADD visgen-c37-production.tar.gz /models

# add the ubuntu user
#RUN groupadd --gid 1000 ubuntu && \
#    useradd --uid 1000 --gid 1000 -d /home/ubuntu -s /bin/sh ubuntu
#RUN chown 1000:1000 -R /cocoapi /detectron /models
#USER ubuntu

#RUN python tools/infer_yeay_production.py --cfg configs/yeay/visgen_cat37_faster_rcnn_R-50-FPN.yaml --output-dir /datasets/yeay/output --input-src /datasets/yeay/videos/Pks2RQhXu9_c3fab014-4b22-488c-b0a1-46453fdc579f.mp4 --dataset yeay --wts /models/visgen-c37-production/train/yeay/generalized_rcnn/model_final.pkl --thresh 0.4
