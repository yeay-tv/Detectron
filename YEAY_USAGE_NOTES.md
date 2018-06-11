# Usage of Detectron (Yeaytectron?)
for object detection of clothing classes and semantic segmentation of people at Yeay GmbH (written 07.06.2018)

## Machine Setup

One has to setup an EC2 instance with a GPU.  For inference and testing, I have chosen a p2.xlarge instance with one K80 NVIDIA GPU.  For training a new network, I use a p3.8xlarge with four V100 GPUs (this should run approximately 12 to 16 times faster than the p2.xlarge instance).  I used ubuntu 16.04 as the operating system and did a relatively clean install.  One could also try the [Amazon Deep Learning AMI](https://aws.amazon.com/blogs/machine-learning/get-started-with-deep-learning-using-the-aws-deep-learning-ami/), but I haven't tested it.  You could basically skip to step 4 below (I would use one of the Conda-based AMIs).

### Steps

1) Install CUDA and cudnn from NVIDIA.
    - Get free NVIDIA developer program account [here](https://developer.nvidia.com/developer-program)
    - Download CUDA [here](https://developer.nvidia.com/cuda-downloads) (latest is 9.2, but if that doesn't work, use 8.0)
        - there are a bunch of options, choose: Linux -> x86_64 -> Ubuntu -> 16.04 -> deb (network)
    - Download cudnn [here](https://developer.nvidia.com/cudnn) (make sure it corresponds to your current CUDA version)
        - install both the library and the dev library
    - everything works by running `nvcc --version` or `nvidia-smi`
        - you may need to add the cuda stuff to your PATH environmental variable.  CUDA lives in /usr/local/cuda, which is a symlink to the particular version of CUDA that was installed.  With multiple version of CUDA installed, you will have multiple CUDA folders in /usr/local (i.e. /usr/local/cuda-9.2) and the cuda folder will symlink to one of those.
        - `echo "export PATH=$PATH:/usr/local/cuda/bin >> ~/.bashrc"` to add the cuda path or add this line manually to your bash profile.

2) Install docker and nvidia-docker
    - install docker-ce, follow [these instructions](https://docs.docker.com/install/linux/docker-ce/ubuntu/#set-up-the-repository)
        - add the ubuntu user to the docker group `sudo usermod -a -G docker ubuntu`
    - install nvidia-docker, follow [these instructions](https://github.com/NVIDIA/nvidia-docker)
3) Install miniconda (not required but makes things easier)
    - download miniconda [here](https://conda.io/miniconda.html)
    - get into base environment `source ~/miniconda3/bin/activate` or `conda activate base`
    - create a new python2 environment `conda create -n mlpy2 python=2.7`
    - activate this environment `conda activate mlpy2`
    - download and install [s3fuse](https://github.com/s3fs-fuse/s3fs-fuse), you'll need yeay's aws credentials to have access to the s3 buckets
        - there is a sample credentials file on the Google drive.  I tarred it and encrypted it with gpg.  The password to the gpg will be passed on.  `tar xvf passwd-s3.tar && gpg .passwd-s3.gpg`
        - Note, these credentials will likely no longer be valid and should be replaced with your own credentials
    - install ffmpeg, opencv, and numpy if you want to use this outside of the docker image
4) Download visualgenome dataset (warning this is pretty huge)
    - `mkdir -p ~/datasets/visualgenome && cd ~/datasets/visualgenome`
    - all the necessary files are [here](https://visualgenome.org/api/v0/api_home.html) for duplicates use the newer version
        - I didn't download the region graphs or scene graphs
        - move VG_100K and VG_100K_2 into a single folder called Images `mkdir Images && mv VG_100K/*.* Images && mv VG_100K_2/*.* Images`
    - uncompress these into a single folder (I choose ~/datasets/visualgenome)
    - download the `vg_yeay_utils.tar.gz` from the Google drive or whereever you've stored it.
    - run `python create_manifest.py`, then `python vg2coco.new.py -o acf.20180502.json`.  You need to run these with python3 or do some minor editing to get them to work in python2.  I would just go to the base conda environment to run these.
    - this should create a file called `acf.20180502.json`, this is the file that will be used by Detectron to setup the dataset and classes for our custom dataset of clothing items.  Also make a copy of this file to acf.new.json `cp acf.20180502.json acf.new.json`
5) clone our version of Detectron
    - `mkdir ~/repos`
    - `cd ~/repos && git clone --branch yeay https://github.com/yeay-tv/Detectron.git && cd Detectron`
    - create a yeay dataset in the "~/datasets" folder with the following subfolders "output", "s3/yeay-user-data-eu" `mkdir -p ~/datasets/yeay/output ~/datasets/yeay/s3/yeay-user-data-eu`
    - create symlinks from the detectron/datasets/data folder to the visual genome and yeay folders `cd detectron/datasets/data && ln -s /home/ubuntu/datasets/visualgenome visualgenome && ln -s /home/ubuntu/datasets/yeay yeay`
    - copy the acf.20180502.json file to the yeay folder `cp ~/datasets/visualgenome/acf.20180502.json ~/datasets/yeay/acf.20180502.json`
    - upload the visgen-c37-production.tar.gz file to ~/repos/Detectron/docker/yeay, this file can be found on the Google drive.
    - build our Dockerfile image `cd ~/repos/Detectron/docker/yeay && docker build --no-cache -t yeay-tv-fb-detectron .`
    - to use this docker image, read the README.md file in ~/repos/Detectron/docker/yeay
        - Note: the docker image will create files that are owned by root.  This is a feature of docker and not something that I could easily figure out how to change.
    - if you want to use this outside of the docker, then follow the normal Detectron install instructions

## Run training and inference

### Inference

1) follow the docker/yeay/README.md instructions for inference
    - this should produce json files in ~/datasets/yeay/output
    - if one uses the option `--create-vis` then a folder with the same name as the video ID will be created with the best frames for each class, the cropped photo of each class, and the frames with the most best classes
2) if you want to do one off, inference, you can test the script with something like, just make sure the input file "/home/ubuntu/datasets/dummy/video/trim_test_vid.mp4" is a valid file:

```sh
python tools/infer_yeay_production.py --cfg /home/ubuntu/tmp/visgen-c38-production/train/yeay/generalized_rcnn/visgen-c38-test.yaml --wts /home/ubuntu/tmp/visgen-c38-production/train/yeay/generalized_rcnn/model_final.pkl --output-dir /tmp/test --input-src /home/ubuntu/datasets/dummy/video/trim_test_vid.mp4 --create-vis

python tools/infer_video_segm.py --cfg configs/12_2017_baselines/e2e_mask_rcnn_R-50-FPN_2x.yaml --wts https://s3-us-west-2.amazonaws.com/detectron/35859007/12_2017_baselines/e2e_mask_rcnn_R-50-FPN_2x.yaml.01_49_07.By8nQcCH/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl --output-dir /tmp/test /home/ubuntu/datasets/dummy/video/trim_test_vid.mp4
```


### Training

1) use the Detectron instructions as a template.  I tend to use the `--skip-test` option.  This assumes 38 classes in the file created by vg2coco.new.py file.  You can change yaml file to have more classes, but if you have less then Detectron will crash during training.  The 38th category was a test category for Nike clothing, it didn't really work, but it didn't hurt the other classes and the vg2coco file creates this class.

```sh
python2 tools/train_net.py \
    --multi-gpu-testing \
    --cfg configs/yeay/visgen-c38-test.yaml \
    OUTPUT_DIR /home/ubuntu/tmp/visgen-c38-production
```
