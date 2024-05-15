# 2

## acc


### vis_no_aug

CUDA_VISIBLE_DEVICES=1 python3 il_predict.py --images_path=/data/617/images/training_4_49_800_800_800_800/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/rt2_training_0_3_800_800_800_800_0_2_2_800_0_17_4_elu/weights_acc/ --save_path=log/rt2_training_0_3_800_800_800_800_0_2_2_800_0_17_4_elu/predict_acc/training_4_49_800_800_800_800 --save_stitched=1 --gpu_id=2 --loss_type=4 --labels_path=/data/617/images/training_4_49_800_800_800_800/labels --psi_act_type=1

### vis_stitched

# python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_4_49/images labels_path=/data/617/images/training_4_49/labels patch_seq_path=log/rt2_training_0_3_800_800_800_800_0_2_2_800_0_17_4_elu/predict_acc/training_4_49_800_800_800_800/raw stitched_seq_path=log/rt2_training_0_3_800_800_800_800_0_2_2_800_0_17_4_elu/predict_acc/training_4_49/raw patch_height=800 patch_width=800 start_id=0 end_id=-1 show_img=0 stacked=1 method=1 normalize_patches=0 img_ext=png del_patch_seq=0

## loss

### vis_no_aug

CUDA_VISIBLE_DEVICES=1 python3 il_predict.py --images_path=/data/617/images/training_4_49_800_800_800_800/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/rt2_training_0_3_800_800_800_800_0_2_2_800_0_17_4_elu/weights_loss/ --save_path=log/rt2_training_0_3_800_800_800_800_0_2_2_800_0_17_4_elu/predict_loss/training_4_49_800_800_800_800 --save_stitched=1 --gpu_id=2 --loss_type=4 --labels_path=/data/617/images/training_4_49_800_800_800_800/labels --psi_act_type=1

### vis_stitched

# python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_4_49/images labels_path=/data/617/images/training_4_49/labels patch_seq_path=log/rt2_training_0_3_800_800_800_800_0_2_2_800_0_17_4_elu/predict_loss/training_4_49_800_800_800_800/raw stitched_seq_path=log/rt2_training_0_3_800_800_800_800_0_2_2_800_0_17_4_elu/predict_loss/training_4_49/raw patch_height=800 patch_width=800 start_id=0 end_id=-1 show_img=0 stacked=1 method=1 normalize_patches=0 img_ext=png del_patch_seq=0

## latest

### vis_no_aug

CUDA_VISIBLE_DEVICES=1 python3 il_predict.py --images_path=/data/617/images/training_4_49_800_800_800_800/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/rt2_training_0_3_800_800_800_800_0_2_2_800_0_17_4_elu/weights/ --save_path=log/rt2_training_0_3_800_800_800_800_0_2_2_800_0_17_4_elu/predict/training_4_49_800_800_800_800 --save_stitched=1 --gpu_id=2 --loss_type=4 --labels_path=/data/617/images/training_4_49_800_800_800_800/labels --psi_act_type=1

### vis_stitched

# python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_4_49/images labels_path=/data/617/images/training_4_49/labels patch_seq_path=log/rt2_training_0_3_800_800_800_800_0_2_2_800_0_17_4_elu/predict/training_4_49_800_800_800_800/raw stitched_seq_path=log/rt2_training_0_3_800_800_800_800_0_2_2_800_0_17_4_elu/predict/training_4_49/raw patch_height=800 patch_width=800 start_id=0 end_id=-1 show_img=0 stacked=1 method=1 normalize_patches=0 img_ext=png del_patch_seq=0


# 10

## acc

### vis_no_aug

CUDA_VISIBLE_DEVICES=1 python3 il_predict.py --images_path=/data/617/images/training_4_49_800_800_800_800/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/rt2_training_0_3_800_800_800_800_0_10_10_800_0_17_4_elu/weights_acc/ --save_path=log/rt2_training_0_3_800_800_800_800_0_10_10_800_0_17_4_elu/predict_acc/training_4_49_800_800_800_800 --save_stitched=1 --gpu_id=2 --loss_type=4 --labels_path=/data/617/images/training_4_49_800_800_800_800/labels --psi_act_type=1

### vis_stitched

# python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_4_49/images labels_path=/data/617/images/training_4_49/labels patch_seq_path=log/rt2_training_0_3_800_800_800_800_0_10_10_800_0_17_4_elu/predict_acc/training_4_49_800_800_800_800/raw stitched_seq_path=log/rt2_training_0_3_800_800_800_800_0_10_10_800_0_17_4_elu/predict_acc/training_4_49/raw patch_height=800 patch_width=800 start_id=0 end_id=-1 show_img=0 stacked=1 method=1 normalize_patches=0 img_ext=png del_patch_seq=0

## loss

### vis_no_aug

CUDA_VISIBLE_DEVICES=1 python3 il_predict.py --images_path=/data/617/images/training_4_49_800_800_800_800/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/rt2_training_0_3_800_800_800_800_0_10_10_800_0_17_4_elu/weights_loss/ --save_path=log/rt2_training_0_3_800_800_800_800_0_10_10_800_0_17_4_elu/predict_loss/training_4_49_800_800_800_800 --save_stitched=1 --gpu_id=2 --loss_type=4 --labels_path=/data/617/images/training_4_49_800_800_800_800/labels --psi_act_type=1

### vis_stitched

# python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_4_49/images labels_path=/data/617/images/training_4_49/labels patch_seq_path=log/rt2_training_0_3_800_800_800_800_0_10_10_800_0_17_4_elu/predict_loss/training_4_49_800_800_800_800/raw stitched_seq_path=log/rt2_training_0_3_800_800_800_800_0_10_10_800_0_17_4_elu/predict_loss/training_4_49/raw patch_height=800 patch_width=800 start_id=0 end_id=-1 show_img=0 stacked=1 method=1 normalize_patches=0 img_ext=png del_patch_seq=0

## latest

### vis_no_aug

CUDA_VISIBLE_DEVICES=1 python3 il_predict.py --images_path=/data/617/images/training_4_49_800_800_800_800/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/rt2_training_0_3_800_800_800_800_0_10_10_800_0_17_4_elu/weights/ --save_path=log/rt2_training_0_3_800_800_800_800_0_10_10_800_0_17_4_elu/predict/training_4_49_800_800_800_800 --save_stitched=1 --gpu_id=2 --loss_type=4 --labels_path=/data/617/images/training_4_49_800_800_800_800/labels --psi_act_type=1

### vis_stitched

# python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_4_49/images labels_path=/data/617/images/training_4_49/labels patch_seq_path=log/rt2_training_0_3_800_800_800_800_0_10_10_800_0_17_4_elu/predict/training_4_49_800_800_800_800/raw stitched_seq_path=log/rt2_training_0_3_800_800_800_800_0_10_10_800_0_17_4_elu/predict/training_4_49/raw patch_height=800 patch_width=800 start_id=0 end_id=-1 show_img=0 stacked=1 method=1 normalize_patches=0 img_ext=png del_patch_seq=0


# 100

## acc


### vis_no_aug

CUDA_VISIBLE_DEVICES=1 python3 il_predict.py --images_path=/data/617/images/training_4_49_800_800_800_800/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/rt2_training_0_3_800_800_800_800_0_100_100_800_0_17_4_elu/weights_acc/ --save_path=log/rt2_training_0_3_800_800_800_800_0_100_100_800_0_17_4_elu/predict_acc/training_4_49_800_800_800_800 --save_stitched=1 --gpu_id=2 --loss_type=4 --labels_path=/data/617/images/training_4_49_800_800_800_800/labels --psi_act_type=1

### vis_stitched

# python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_4_49/images labels_path=/data/617/images/training_4_49/labels patch_seq_path=log/rt2_training_0_3_800_800_800_800_0_100_100_800_0_17_4_elu/predict_acc/training_4_49_800_800_800_800/raw stitched_seq_path=log/rt2_training_0_3_800_800_800_800_0_100_100_800_0_17_4_elu/predict_acc/training_4_49/raw patch_height=800 patch_width=800 start_id=0 end_id=-1 show_img=0 stacked=1 method=1 normalize_patches=0 img_ext=png del_patch_seq=0

## loss

### vis_no_aug

CUDA_VISIBLE_DEVICES=1 python3 il_predict.py --images_path=/data/617/images/training_4_49_800_800_800_800/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/rt2_training_0_3_800_800_800_800_0_100_100_800_0_17_4_elu/weights_loss/ --save_path=log/rt2_training_0_3_800_800_800_800_0_100_100_800_0_17_4_elu/predict_loss/training_4_49_800_800_800_800 --save_stitched=1 --gpu_id=2 --loss_type=4 --labels_path=/data/617/images/training_4_49_800_800_800_800/labels --psi_act_type=1

### vis_stitched

# python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_4_49/images labels_path=/data/617/images/training_4_49/labels patch_seq_path=log/rt2_training_0_3_800_800_800_800_0_100_100_800_0_17_4_elu/predict_loss/training_4_49_800_800_800_800/raw stitched_seq_path=log/rt2_training_0_3_800_800_800_800_0_100_100_800_0_17_4_elu/predict_loss/training_4_49/raw patch_height=800 patch_width=800 start_id=0 end_id=-1 show_img=0 stacked=1 method=1 normalize_patches=0 img_ext=png del_patch_seq=0

## latest

### vis_no_aug

CUDA_VISIBLE_DEVICES=1 python3 il_predict.py --images_path=/data/617/images/training_4_49_800_800_800_800/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/rt2_training_0_3_800_800_800_800_0_100_100_800_0_17_4_elu/weights/ --save_path=log/rt2_training_0_3_800_800_800_800_0_100_100_800_0_17_4_elu/predict/training_4_49_800_800_800_800 --save_stitched=1 --gpu_id=2 --loss_type=4 --labels_path=/data/617/images/training_4_49_800_800_800_800/labels --psi_act_type=1

### vis_stitched

# python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_4_49/images labels_path=/data/617/images/training_4_49/labels patch_seq_path=log/rt2_training_0_3_800_800_800_800_0_100_100_800_0_17_4_elu/predict/training_4_49_800_800_800_800/raw stitched_seq_path=log/rt2_training_0_3_800_800_800_800_0_100_100_800_0_17_4_elu/predict/training_4_49/raw patch_height=800 patch_width=800 start_id=0 end_id=-1 show_img=0 stacked=1 method=1 normalize_patches=0 img_ext=png del_patch_seq=0

# 1000

## acc

### vis_no_aug

CUDA_VISIBLE_DEVICES=1 python3 il_predict.py --images_path=/data/617/images/training_4_49_800_800_800_800/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/rt2_training_0_3_800_800_800_800_0_1000_1000_800_0_17_4_elu/weights_acc/ --save_path=log/rt2_training_0_3_800_800_800_800_0_1000_1000_800_0_17_4_elu/predict_acc/training_4_49_800_800_800_800 --save_stitched=1 --gpu_id=2 --loss_type=4 --labels_path=/data/617/images/training_4_49_800_800_800_800/labels --psi_act_type=1

### vis_stitched

# python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_4_49/images labels_path=/data/617/images/training_4_49/labels patch_seq_path=log/rt2_training_0_3_800_800_800_800_0_1000_1000_800_0_17_4_elu/predict_acc/training_4_49_800_800_800_800/raw stitched_seq_path=log/rt2_training_0_3_800_800_800_800_0_1000_1000_800_0_17_4_elu/predict_acc/training_4_49/raw patch_height=800 patch_width=800 start_id=0 end_id=-1 show_img=0 stacked=1 method=1 normalize_patches=0 img_ext=png del_patch_seq=0

## loss

### vis_no_aug

CUDA_VISIBLE_DEVICES=1 python3 il_predict.py --images_path=/data/617/images/training_4_49_800_800_800_800/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/rt2_training_0_3_800_800_800_800_0_1000_1000_800_0_17_4_elu/weights_loss/ --save_path=log/rt2_training_0_3_800_800_800_800_0_1000_1000_800_0_17_4_elu/predict_loss/training_4_49_800_800_800_800 --save_stitched=1 --gpu_id=2 --loss_type=4 --labels_path=/data/617/images/training_4_49_800_800_800_800/labels --psi_act_type=1

### vis_stitched

# python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_4_49/images labels_path=/data/617/images/training_4_49/labels patch_seq_path=log/rt2_training_0_3_800_800_800_800_0_1000_1000_800_0_17_4_elu/predict_loss/training_4_49_800_800_800_800/raw stitched_seq_path=log/rt2_training_0_3_800_800_800_800_0_1000_1000_800_0_17_4_elu/predict_loss/training_4_49/raw patch_height=800 patch_width=800 start_id=0 end_id=-1 show_img=0 stacked=1 method=1 normalize_patches=0 img_ext=png del_patch_seq=0

## latest

### vis_no_aug

CUDA_VISIBLE_DEVICES=1 python3 il_predict.py --images_path=/data/617/images/training_4_49_800_800_800_800/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/rt2_training_0_3_800_800_800_800_0_1000_1000_800_0_17_4_elu/weights/ --save_path=log/rt2_training_0_3_800_800_800_800_0_1000_1000_800_0_17_4_elu/predict/training_4_49_800_800_800_800 --save_stitched=1 --gpu_id=2 --loss_type=4 --labels_path=/data/617/images/training_4_49_800_800_800_800/labels --psi_act_type=1

### vis_stitched

# python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_4_49/images labels_path=/data/617/images/training_4_49/labels patch_seq_path=log/rt2_training_0_3_800_800_800_800_0_1000_1000_800_0_17_4_elu/predict/training_4_49_800_800_800_800/raw stitched_seq_path=log/rt2_training_0_3_800_800_800_800_0_1000_1000_800_0_17_4_elu/predict/training_4_49/raw patch_height=800 patch_width=800 start_id=0 end_id=-1 show_img=0 stacked=1 method=1 normalize_patches=0 img_ext=png del_patch_seq=0
