set -x


# 2

## YUN00001_3600

CUDA_VISIBLE_DEVICES=1 python3 il_predict.py --images_path=/data/617/images/YUN00001_3600_0_3599_800_800_800_800/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/rt2_training_0_3_640_640_640_640_0_2_2_640_0_24_4_elu/weights/ --save_path=log/rt2_training_0_3_640_640_640_640_0_2_2_640_0_24_4_elu/predict/YUN00001_3600_0_3599_800_800_800_800 --save_stitched=0 --save_seg=1 --gpu_id=2 --loss_type=4 --psi_act_type=1

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/YUN00001_3600/images patch_seq_path=log/rt2_training_0_3_640_640_640_640_0_2_2_640_0_24_4_elu/predict/YUN00001_3600_0_3599_800_800_800_800/raw stitched_seq_path=log/rt2_training_0_3_640_640_640_640_0_2_2_640_0_24_4_elu/predict/YUN00001_3600 patch_height=800 patch_width=800 start_id=0 end_id=-1 show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=jpg del_patch_seq=0 out_ext=mkv width=1920 height=1080

## 20160122_YUN00020_2000_3800

CUDA_VISIBLE_DEVICES=1 python3 il_predict.py --images_path=/data/617/images/20160122_YUN00020_2000_3800_0_1799_800_800_800_800/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/rt2_training_0_3_640_640_640_640_0_2_2_640_0_24_4_elu/weights/ --save_path=log/rt2_training_0_3_640_640_640_640_0_2_2_640_0_24_4_elu/predict/20160122_YUN00020_2000_3800_0_1799_800_800_800_800 --save_stitched=0 --save_seg=1 --gpu_id=2 --loss_type=4 --psi_act_type=1

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/20160122_YUN00020_2000_3800/images patch_seq_path=log/rt2_training_0_3_640_640_640_640_0_2_2_640_0_24_4_elu/predict/20160122_YUN00020_2000_3800_0_1799_800_800_800_800/raw stitched_seq_path=log/rt2_training_0_3_640_640_640_640_0_2_2_640_0_24_4_elu/predict/20160122_YUN00020_2000_3800 patch_height=800 patch_width=800 start_id=0 end_id=-1 show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=jpg del_patch_seq=0 out_ext=mkv width=1920 height=1080


# 100

## YUN00001_3600

CUDA_VISIBLE_DEVICES=1 python3 il_predict.py --images_path=/data/617/images/YUN00001_3600_0_3599_800_800_800_800/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/rt2_training_0_3_640_640_640_640_0_100_100_640_0_24_4_elu/weights/ --save_path=log/rt2_training_0_3_640_640_640_640_0_100_100_640_0_24_4_elu/predict/YUN00001_3600_0_3599_800_800_800_800 --save_stitched=0 --save_seg=1 --gpu_id=2 --loss_type=4 --psi_act_type=1

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/YUN00001_3600/images patch_seq_path=log/rt2_training_0_3_640_640_640_640_0_100_100_640_0_24_4_elu/predict/YUN00001_3600_0_3599_800_800_800_800/raw stitched_seq_path=log/rt2_training_0_3_640_640_640_640_0_100_100_640_0_24_4_elu/predict/YUN00001_3600 patch_height=800 patch_width=800 start_id=0 end_id=-1 show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=jpg del_patch_seq=0 out_ext=mkv width=1920 height=1080

## 20160122_YUN00020_2000_3800

CUDA_VISIBLE_DEVICES=1 python3 il_predict.py --images_path=/data/617/images/20160122_YUN00020_2000_3800_0_1799_800_800_800_800/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/rt2_training_0_3_640_640_640_640_0_100_100_640_0_24_4_elu/weights/ --save_path=log/rt2_training_0_3_640_640_640_640_0_100_100_640_0_24_4_elu/predict/20160122_YUN00020_2000_3800_0_1799_800_800_800_800 --save_stitched=0 --save_seg=1 --gpu_id=2 --loss_type=4 --psi_act_type=1

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/20160122_YUN00020_2000_3800/images patch_seq_path=log/rt2_training_0_3_640_640_640_640_0_100_100_640_0_24_4_elu/predict/20160122_YUN00020_2000_3800_0_1799_800_800_800_800/raw stitched_seq_path=log/rt2_training_0_3_640_640_640_640_0_100_100_640_0_24_4_elu/predict/20160122_YUN00020_2000_3800 patch_height=800 patch_width=800 start_id=0 end_id=-1 show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=jpg del_patch_seq=0 out_ext=mkv width=1920 height=1080




# 1000

## YUN00001_3600

CUDA_VISIBLE_DEVICES=1 python3 il_predict.py --images_path=/data/617/images/YUN00001_3600_0_3599_800_800_800_800/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/rt2_training_0_3_640_640_640_640_0_1000_1000_640_0_24_4_elu/weights/ --save_path=log/rt2_training_0_3_640_640_640_640_0_1000_1000_640_0_24_4_elu/predict/YUN00001_3600_0_3599_800_800_800_800 --save_stitched=0 --save_seg=1 --gpu_id=2 --loss_type=4 --psi_act_type=1

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/YUN00001_3600/images patch_seq_path=log/rt2_training_0_3_640_640_640_640_0_1000_1000_640_0_24_4_elu/predict/YUN00001_3600_0_3599_800_800_800_800/raw stitched_seq_path=log/rt2_training_0_3_640_640_640_640_0_1000_1000_640_0_24_4_elu/predict/YUN00001_3600 patch_height=800 patch_width=800 start_id=0 end_id=-1 show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=jpg del_patch_seq=0 out_ext=mkv width=1920 height=1080

## 20160122_YUN00020_2000_3800

CUDA_VISIBLE_DEVICES=1 python3 il_predict.py --images_path=/data/617/images/20160122_YUN00020_2000_3800_0_1799_800_800_800_800/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/rt2_training_0_3_640_640_640_640_0_1000_1000_640_0_24_4_elu/weights/ --save_path=log/rt2_training_0_3_640_640_640_640_0_1000_1000_640_0_24_4_elu/predict/20160122_YUN00020_2000_3800_0_1799_800_800_800_800 --save_stitched=0 --save_seg=1 --gpu_id=2 --loss_type=4 --psi_act_type=1

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/20160122_YUN00020_2000_3800/images patch_seq_path=log/rt2_training_0_3_640_640_640_640_0_1000_1000_640_0_24_4_elu/predict/20160122_YUN00020_2000_3800_0_1799_800_800_800_800/raw stitched_seq_path=log/rt2_training_0_3_640_640_640_640_0_1000_1000_640_0_24_4_elu/predict/20160122_YUN00020_2000_3800 patch_height=800 patch_width=800 start_id=0 end_id=-1 show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=jpg del_patch_seq=0 out_ext=mkv width=1920 height=1080



