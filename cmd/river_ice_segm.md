<!-- MarkdownTOC -->

- [videoToImgSeq](#videotoimgseq_)
   - [1920x1080       @ videoToImgSeq](#1920x1080___videotoimgse_q_)
   - [4k       @ videoToImgSeq](#4k___videotoimgse_q_)
      - [YUN00001_3600       @ 4k/videoToImgSeq](#yun00001_3600___4k_videotoimgseq_)
      - [YUN00001_3600__win       @ 4k/videoToImgSeq](#yun00001_3600_win___4k_videotoimgseq_)
      - [20160121_YUN00002_2000       @ 4k/videoToImgSeq](#20160121_yun00002_2000___4k_videotoimgseq_)
      - [20161201_YUN00002_1800       @ 4k/videoToImgSeq](#20161201_yun00002_1800___4k_videotoimgseq_)
      - [20160122_YUN00002_700_2500       @ 4k/videoToImgSeq](#20160122_yun00002_700_2500___4k_videotoimgseq_)
      - [20160122_YUN00020_2000_3800       @ 4k/videoToImgSeq](#20160122_yun00020_2000_3800___4k_videotoimgseq_)
      - [20160122_YUN00020_2000_3800__win_pc       @ 4k/videoToImgSeq](#20160122_yun00020_2000_3800_win_pc___4k_videotoimgseq_)
      - [20161203_Deployment_1_YUN00001_900_2700       @ 4k/videoToImgSeq](#20161203_deployment_1_yun00001_900_2700___4k_videotoimgseq_)
      - [20161203_Deployment_1_YUN00001_900_1200_win_pc       @ 4k/videoToImgSeq](#20161203_deployment_1_yun00001_900_1200_win_pc___4k_videotoimgseq_)
      - [20161203_Deployment_1_YUN00001_2000_2300__win_pc       @ 4k/videoToImgSeq](#20161203_deployment_1_yun00001_2000_2300_win_pc___4k_videotoimgseq_)
      - [20161203_Deployment_1_YUN00002_1800       @ 4k/videoToImgSeq](#20161203_deployment_1_yun00002_1800___4k_videotoimgseq_)
      - [20170114_YUN00005_1800       @ 4k/videoToImgSeq](#20170114_yun00005_1800___4k_videotoimgseq_)
- [stitch multiple results](#stitch_multiple_result_s_)

<!-- /MarkdownTOC -->

<a id="videotoimgseq_"></a>
# videoToImgSeq

<a id="1920x1080___videotoimgse_q_"></a>
## 1920x1080       @ videoToImgSeq-->river_ice_segm

python3 videoToImgSeq.py db_root_dir=/data/617/videos actor=20160121 seq_name=YUN00001 vid_fmt=mp4 n_frames=1800 resize_factor=0.50 dst_dir=/data/617/images/YUN00001_1920x1080/images

python3 videoToImgSeq.py db_root_dir=/data/617/videos actor=20160121 seq_name=YUN00002 vid_fmt=mp4 n_frames=1800 resize_fa ctor=0.50 dst_dir=/data/617/images/YUN00002_1920x1080/images

<a id="4k___videotoimgse_q_"></a>
## 4k       @ videoToImgSeq-->river_ice_segm

python3 videoToImgSeq.py db_root_dir=/data/617/videos actor=20160121 seq_name=YUN00001 vid_fmt=mp4 n_frames=-1 resize_factor=1 dst_dir=/data/617/images/YUN00001/images

python3 videoToImgSeq.py db_root_dir=/data/617/videos actor=20160121 seq_name=YUN00001 vid_fmt=mp4 n_frames=1800 resize_factor=1 dst_dir=/data/617/images/YUN00001_1800/images

<a id="yun00001_3600___4k_videotoimgseq_"></a>
### YUN00001_3600       @ 4k/videoToImgSeq-->river_ice_segm

python3 videoToImgSeq.py db_root_dir=/data/617/videos actor=20160121 seq_name=YUN00001 vid_fmt=mp4 n_frames=3600 resize_factor=1 dst_dir=/data/617/images/YUN00001_3600/images

<a id="yun00001_3600_win___4k_videotoimgseq_"></a>
### YUN00001_3600__win       @ 4k/videoToImgSeq-->river_ice_segm

python3 videoToImgSeq.py db_root_dir=E:\Datasets\617\videos actor=20160121 seq_name=YUN00001 vid_fmt=mp4 n_frames=3600 resize_factor=1 dst_dir=E:\Datasets\617\images\YUN00001_3600\images

<a id="20160121_yun00002_2000___4k_videotoimgseq_"></a>
### 20160121_YUN00002_2000       @ 4k/videoToImgSeq-->river_ice_segm

python3 videoToImgSeq.py db_root_dir=/data/617/videos actor=20160121 seq_name=YUN00002 vid_fmt=mp4 n_frames=1800 resize_factor=1.0 start_id=2000 dst_dir=/data/617/images/20160121_YUN00002_2000/images 

<a id="20161201_yun00002_1800___4k_videotoimgseq_"></a>
### 20161201_YUN00002_1800       @ 4k/videoToImgSeq-->river_ice_segm

python3 videoToImgSeq.py db_root_dir=/data/617/videos actor=20161201 seq_name=YUN00002 vid_fmt=mp4 n_frames=1800 resize_factor=1.0 start_id=0 dst_dir=/data/617/images/20161201_YUN00002_1800/images

<a id="20160122_yun00002_700_2500___4k_videotoimgseq_"></a>
### 20160122_YUN00002_700_2500       @ 4k/videoToImgSeq-->river_ice_segm

python3 videoToImgSeq.py db_root_dir=/data/617/videos actor=20160122 seq_name=YUN00002 vid_fmt=mp4 n_frames=1800 resize_factor=1.0 start_id=700 dst_dir=/data/617/images/20160122_YUN00002_700_2500/images 

<a id="20160122_yun00020_2000_3800___4k_videotoimgseq_"></a>
### 20160122_YUN00020_2000_3800       @ 4k/videoToImgSeq-->river_ice_segm

python3 videoToImgSeq.py db_root_dir=/data/617/videos actor=20160122 seq_name=YUN00020 vid_fmt=mp4 n_frames=1800 resize_factor=1.0 start_id=2000 dst_dir=/data/617/images/20160122_YUN00020_2000_3800/images

<a id="20160122_yun00020_2000_3800_win_pc___4k_videotoimgseq_"></a>
### 20160122_YUN00020_2000_3800__win_pc       @ 4k/videoToImgSeq-->river_ice_segm

python3 videoToImgSeq.py db_root_dir=E:\Datasets\617\videos actor=20160122 seq_name=YUN00020 vid_fmt=mp4 n_frames=300 resize_factor=1 start_id=2000 dst_dir=P:\Datasets\617\images\20160122_YUN00020_2000_300\images

<a id="20161203_deployment_1_yun00001_900_2700___4k_videotoimgseq_"></a>
### 20161203_Deployment_1_YUN00001_900_2700       @ 4k/videoToImgSeq-->river_ice_segm

python3 videoToImgSeq.py db_root_dir=/data/617/videos actor=20161203 seq_name=Deployment_1_YUN00001 vid_fmt=mp4 n_frames=1800 resize_factor=1.0 start_id=900 dst_dir=/data/617/images/20161203_Deployment_1_YUN00001_900_2700/images

<a id="20161203_deployment_1_yun00001_900_1200_win_pc___4k_videotoimgseq_"></a>
### 20161203_Deployment_1_YUN00001_900_1200_win_pc       @ 4k/videoToImgSeq-->river_ice_segm

__corrected__
python3 videoToImgSeq.py db_root_dir=E:\Datasets\617\videos actor=20161203 seq_name=Deployment_1_YUN00001 vid_fmt=mp4 n_frames=300 resize_factor=1 start_id=900 dst_dir=P:\Datasets\617\images\20161203_Deployment_1_YUN00001_900_1200\images

<a id="20161203_deployment_1_yun00001_2000_2300_win_pc___4k_videotoimgseq_"></a>
### 20161203_Deployment_1_YUN00001_2000_2300__win_pc       @ 4k/videoToImgSeq-->river_ice_segm

python3 videoToImgSeq.py db_root_dir=E:\Datasets\617\videos actor=20161203 seq_name=Deployment_1_YUN00001 vid_fmt=mp4 n_frames=300 resize_factor=1 start_id=2000 dst_dir=E:\Datasets\617\images\20161203_Deployment_1_YUN00001_2000_2300\images

<a id="20161203_deployment_1_yun00002_1800___4k_videotoimgseq_"></a>
### 20161203_Deployment_1_YUN00002_1800       @ 4k/videoToImgSeq-->river_ice_segm

python3 videoToImgSeq.py db_root_dir=/data/617/videos actor=20161203 seq_name=Deployment_1_YUN00002 vid_fmt=mp4 n_frames=1800 resize_factor=1.0 start_id=0 dst_dir=/data/617/images/20161203_Deployment_1_YUN00002_1800/images


<a id="20170114_yun00005_1800___4k_videotoimgseq_"></a>
### 20170114_YUN00005_1800       @ 4k/videoToImgSeq-->river_ice_segm

python3 videoToImgSeq.py db_root_dir=/data/617/videos actor=20170114 seq_name=YUN00005 vid_fmt=mp4 n_frames=1800 resize_factor=1.0 start_id=0 dst_dir=/data/617/images/20170114_YUN00005_1800/images 


<a id="stitch_multiple_result_s_"></a>
# stitch multiple results

python3 stitchMultipleResults.py --seg_root_dir=/home/abhineet/H/UofA/617/Project/presentation --images_path=/data/617/images/validation/images --save_path=/home/abhineet/H/UofA/617/Project/presentation/stitched --show_img=1 --resize_factor=0.25










