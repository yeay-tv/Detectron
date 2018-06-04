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

s3fs -o allow_other -o umask=0002 yeay-user-data-eu /home/ubuntu/datasets/yeay/s3/yeay-user-data-eu

# 2)
nvidia-docker run -v ~/datasets/yeay:/datasets/yeay -u `id -u ubuntu`:`id -g ubuntu` --rm -it yeay-tv-fb-detectron python tools/get_remote_videos.py

# 3)
nvidia-docker run -v ~/datasets/yeay:/datasets/yeay --rm -it yeay-tv-fb-detectron python tools/infer_yeay_production.py
```
