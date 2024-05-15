set -x

## 1000

CUDA_VISIBLE_DEVICES=1 python3 il_predict.py --images_path=/data/617/images/20161201_YUN00002_0_1799_1000_1000_1000_1000/images --height=1000 --width=1000 --n_classes=3 --start_id=0 --end_id=2500 --weights_path=log/50_10000_10000_1000_random_801_4/weights/ --save_path=log/50_10000_10000_1000_random_801_4/predict/20161201_YUN00002_0_1799_1000_1000_1000_1000 --save_stitched=1 --gpu_id=2 --loss_type=4 --save_raw=0 --out_ext=jpg

python3 ~/617_w18/Project/code/stitchSubPatchDataset.py src_path=/data/617/images/20161201_YUN00002/images img_ext=jpg  patch_seq_path=log/50_10000_10000_1000_random_801_4/predict/20161201_YUN00002_0_1799_1000_1000_1000_1000 stitched_seq_path=log/50_10000_10000_1000_random_801_4/predict/20161201_YUN00002_0_1799_1000_1000_1000_1000_stitched out_ext=mkv patch_height=1000 patch_width=1000 start_id=0 end_id=-1  show_img=0 stacked=1 method=1 normalize_patches=0 resize_factor=0.5 patch_ext=jpg del_patch_seq=1

# python3 ~/PTF/imgSeqToVideo.py src_path=log/50_10000_10000_1000_random_801_4/predict/20161201_YUN00002_0_1799_1000_1000_1000_1000_stitched save_path=log/50_10000_10000_1000_random_801_4/predict/20161201_YUN00002_0_1799_1000_1000_1000_1000_stitched out_ext=mkv width=1920 height=1080 show_img=0

# 20170114_YUN00005

CUDA_VISIBLE_DEVICES=1 python3 il_predict.py --images_path=/data/617/images/20170114_YUN00005_0_1799_1000_1000_1000_1000/images --height=1000 --width=1000 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/50_10000_10000_1000_random_801_4/weights/ --save_path=log/50_10000_10000_1000_random_801_4/predict/20170114_YUN00005_0_1799_1000_1000_1000_1000 --save_stitched=1 --gpu_id=2 --loss_type=4 --save_raw=0 --out_ext=jpg

python3 ~/617_w18/Project/code/stitchSubPatchDataset.py src_path=/data/617/images/20170114_YUN00005/images img_ext=jpg  patch_seq_path=log/50_10000_10000_1000_random_801_4/predict/20170114_YUN00005_0_1799_1000_1000_1000_1000 stitched_seq_path=log/50_10000_10000_1000_random_801_4/predict/20170114_YUN00005_0_1799_1000_1000_1000_1000_stitched out_ext=mkv patch_height=1000 patch_width=1000 start_id=0 end_id=-1  show_img=0 stacked=1 method=1 normalize_patches=0 resize_factor=0.5 patch_ext=jpg del_patch_seq=1

# python3 ~/PTF/imgSeqToVideo.py src_path=log/50_10000_10000_1000_random_801_4/predict/20170114_YUN00005_0_1799_1000_1000_1000_1000_stitched save_path=log/50_10000_10000_1000_random_801_4/predict/20170114_YUN00005_0_1799_1000_1000_1000_1000_stitched out_ext=mkv width=1920 height=1080 show_img=0


# YUN00002_2000

CUDA_VISIBLE_DEVICES=1 python3 il_predict.py --images_path=/data/617/images/YUN00002_2000_0_1799_1000_1000_1000_1000/images --height=1000 --width=1000 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/50_10000_10000_1000_random_801_4/weights/ --save_path=log/50_10000_10000_1000_random_801_4/predict/YUN00002_2000_0_1799_1000_1000_1000_1000 --save_stitched=1 --gpu_id=2 --loss_type=4 --save_raw=0 --out_ext=jpg

python3 ~/617_w18/Project/code/stitchSubPatchDataset.py src_path=/data/617/images/YUN00002_2000/images img_ext=jpg  patch_seq_path=log/50_10000_10000_1000_random_801_4/predict/YUN00002_2000_0_1799_1000_1000_1000_1000 stitched_seq_path=log/50_10000_10000_1000_random_801_4/predict/YUN00002_2000_0_1799_1000_1000_1000_1000_stitched out_ext=mkv patch_height=1000 patch_width=1000 start_id=0 end_id=-1  show_img=0 stacked=1 method=1 normalize_patches=0 resize_factor=0.5 patch_ext=jpg del_patch_seq=1

# python3 ~/PTF/imgSeqToVideo.py src_path=log/50_10000_10000_1000_random_801_4/predict/YUN00002_2000_0_1799_1000_1000_1000_1000_stitched save_path=log/50_10000_10000_1000_random_801_4/predict/YUN00002_2000_0_1799_1000_1000_1000_1000_stitched out_ext=mkv width=1920 height=1080 show_img=0


# 20160122_YUN00002_700

CUDA_VISIBLE_DEVICES=1 python3 il_predict.py --images_path=/data/617/images/20160122_YUN00002_700_0_1799_1000_1000_1000_1000/images --height=1000 --width=1000 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/50_10000_10000_1000_random_801_4/weights/ --save_path=log/50_10000_10000_1000_random_801_4/predict/20160122_YUN00002_700_0_1799_1000_1000_1000_1000 --save_stitched=1 --gpu_id=2 --loss_type=4 --save_raw=0 --out_ext=jpg

python3 ~/617_w18/Project/code/stitchSubPatchDataset.py src_path=/data/617/images/20160122_YUN00002_700/images img_ext=jpg  patch_seq_path=log/50_10000_10000_1000_random_801_4/predict/20160122_YUN00002_700_0_1799_1000_1000_1000_1000 stitched_seq_path=log/50_10000_10000_1000_random_801_4/predict/20160122_YUN00002_700_0_1799_1000_1000_1000_1000_stitched out_ext=mkv patch_height=1000 patch_width=1000 start_id=0 end_id=-1  show_img=0 stacked=1 method=1 normalize_patches=0 resize_factor=0.5 patch_ext=jpg del_patch_seq=1

# python3 ~/PTF/imgSeqToVideo.py src_path=log/50_10000_10000_1000_random_801_4/predict/20160122_YUN00002_700_0_1799_1000_1000_1000_1000_stitched save_path=log/50_10000_10000_1000_random_801_4/predict/20160122_YUN00002_700_0_1799_1000_1000_1000_1000_stitched out_ext=mkv width=1920 height=1080 show_img=0

