# Install

```sh
docker build --no-cache -t yeay-tv-fb-detectron .
```

# Run
0) get detectron trained models
1) mount s3 bucket
2) get urls/paths of videos
3) run classifier

The trained models are on Google drive (https://drive.google.com/drive/u/0/folders/1rsin7N9R081yHMn5c9pu2sL7iwewBO1r)

```sh
# 1)

# edit /etc/fuse.conf if necessary to allow allow_other
s3fs -o allow_other -o umask=0002 yeay-user-data-eu /home/ubuntu/datasets/yeay/s3/yeay-user-data-eu

# 2)
nvidia-docker run -v ~/datasets/yeay:/datasets/yeay -u `id -u ubuntu`:`id -g ubuntu` --rm -it yeay-tv-fb-detectron python tools/get_remote_videos.py

# 3)
nvidia-docker run -v ~/datasets/yeay:/datasets/yeay -v ~/datasets/visualgenome:/datasets/visualgenome --rm -it yeay-tv-fb-detectron python tools/infer_yeay_production.py

# 4)
nvidia-docker run -v ~/datasets/yeay:/datasets/yeay -v ~/datasets/visualgenome:/datasets/visualgenome --rm -it yeay-tv-fb-detectron python tools/infer_video_segm.py --cfg configs/12_2017_baselines/e2e_mask_rcnn_R-50-FPN_2x.yaml --wts https://s3-us-west-2.amazonaws.com/detectron/35859007/12_2017_baselines/e2e_mask_rcnn_R-50-FPN_2x.yaml.01_49_07.By8nQcCH/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl --output-dir /datasets/yeay/output /datasets/yeay/video_url_list.json
```

# Cleanup

To remove old images after building (or rebuilding), run the following command

```sh
docker rmi $(docker images --filter "dangling=true" -q --no-trunc)
```
