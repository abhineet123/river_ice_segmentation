set -x


# predict_acc

CUDA_VISIBLE_DEVICES=1 python3 il_predict.py --images_path=/data/617/images/YUN00001_3600_0_3599_800_800_800_800/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu/weights_acc/ --save_path=log/rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu/predict_acc/YUN00001_3600_0_3599_800_800_800_800 --save_stitched=0 --save_seg=1 --gpu_id=2 --loss_type=4 --psi_act_type=1

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/YUN00001_3600/images patch_seq_path=log/rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu/predict_acc/YUN00001_3600_0_3599_800_800_800_800/raw stitched_seq_path=log/rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu/predict_acc/YUN00001_3600 patch_height=800 patch_width=800 start_id=0 end_id=-1 show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=jpg del_patch_seq=0 out_ext=mkv width=1920 height=1080


# predict_loss

CUDA_VISIBLE_DEVICES=1 python3 il_predict.py --images_path=/data/617/images/YUN00001_3600_0_3599_800_800_800_800/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu/weights_loss/ --save_path=log/rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu/predict_loss/YUN00001_3600_0_3599_800_800_800_800 --save_stitched=0 --save_seg=1 --gpu_id=2 --loss_type=4 --psi_act_type=1

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/YUN00001_3600/images patch_seq_path=log/rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu/predict_loss/YUN00001_3600_0_3599_800_800_800_800/raw stitched_seq_path=log/rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu/predict_loss/YUN00001_3600 patch_height=800 patch_width=800 start_id=0 end_id=-1 show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=jpg del_patch_seq=0 out_ext=mkv width=1920 height=1080

# predict

CUDA_VISIBLE_DEVICES=1 python3 il_predict.py --images_path=/data/617/images/YUN00001_3600_0_3599_800_800_800_800/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu/weights/ --save_path=log/rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu/predict/YUN00001_3600_0_3599_800_800_800_800 --save_stitched=0 --save_seg=1 --gpu_id=2 --loss_type=4 --psi_act_type=1

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/YUN00001_3600/images patch_seq_path=log/rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu/predict/YUN00001_3600_0_3599_800_800_800_800/raw stitched_seq_path=log/rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu/predict/YUN00001_3600 patch_height=800 patch_width=800 start_id=0 end_id=-1 show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=jpg del_patch_seq=0 out_ext=mkv width=1920 height=1080