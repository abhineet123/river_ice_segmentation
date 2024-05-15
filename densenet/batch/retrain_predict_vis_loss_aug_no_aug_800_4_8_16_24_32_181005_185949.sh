set -x


## 8


### vis

CUDA_VISIBLE_DEVICES=2 python3 il_predict.py --images_path=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/training_0_7_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_581_4/weights_loss/ --save_path=log/training_0_7_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_581_4/predict_loss/training_32_49_800_800_80_320_rot_15_345_4_flip --save_stitched=1 --gpu_id=2 --loss_type=4

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/labels --seg_path=log/training_0_7_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_581_4/predict_loss/training_32_49_800_800_80_320_rot_15_345_4_flip/raw --save_path=log/vis/training_32_49_800_800_80_320_rot_15_345_4_flip/training_0_7_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_581_4 --n_classes=3 --start_id=0 --end_id=-1

#### no_aug

CUDA_VISIBLE_DEVICES=2 python3 il_predict.py --images_path=/data/617/images/training_32_49_800_800_800_800/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/training_0_7_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_581_4/weights_loss/ --save_path=log/training_0_7_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_581_4/predict_loss/training_32_49_800_800_800_800 --save_stitched=1 --gpu_id=2 --loss_type=4

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_800_800_800_800/images --labels_path=/data/617/images/training_32_49_800_800_800_800/labels --seg_path=log/training_0_7_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_581_4/predict_loss/training_32_49_800_800_800_800/raw --save_path=log/vis/training_32_49_800_800_800_800/training_0_7_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_581_4 --n_classes=3 --start_id=0 --end_id=-1

#### stitched

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/training_0_7_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_581_4/predict_loss/training_32_49_800_800_800_800 stitched_seq_path=log/training_0_7_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_581_4/predict_loss/training_32_49/raw patch_height=800 patch_width=800 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python3 ../visDataset.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_path=log/training_0_7_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_581_4/predict_loss/training_32_49/raw --save_path=log/training_0_7_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_581_4/predict_loss/training_32_49_800_800_800_800 --save_path=log/training_0_7_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_581_4/predict_loss/training_32_49/vis --n_classes=3 --start_id=0 --end_id=-1


### 16


### vis

CUDA_VISIBLE_DEVICES=2 python3 il_predict.py --images_path=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/training_0_15_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_887_4/weights_loss/ --save_path=log/training_0_15_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_887_4/predict_loss/training_32_49_800_800_80_320_rot_15_345_4_flip --save_stitched=1 --gpu_id=2 --loss_type=4

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/labels --seg_path=log/training_0_15_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_887_4/predict_loss/training_32_49_800_800_80_320_rot_15_345_4_flip/raw --save_path=log/vis/training_32_49_800_800_80_320_rot_15_345_4_flip/training_0_15_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_887_4 --n_classes=3 --start_id=0 --end_id=-1

#### no_aug

CUDA_VISIBLE_DEVICES=2 python3 il_predict.py --images_path=/data/617/images/training_32_49_800_800_800_800/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/training_0_15_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_887_4/weights_loss/ --save_path=log/training_0_15_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_887_4/predict_loss/training_32_49_800_800_800_800 --save_stitched=1 --gpu_id=2 --loss_type=4

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_800_800_800_800/images --labels_path=/data/617/images/training_32_49_800_800_800_800/labels --seg_path=log/training_0_15_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_887_4/predict_loss/training_32_49_800_800_800_800/raw --save_path=log/vis/training_32_49_800_800_800_800/training_0_15_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_887_4 --n_classes=3 --start_id=0 --end_id=-1

#### stitched

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/training_0_15_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_887_4/predict_loss/training_32_49_800_800_800_800 stitched_seq_path=log/training_0_15_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_887_4/predict_loss/training_32_49/raw patch_height=800 patch_width=800 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python3 ../visDataset.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_path=log/training_0_15_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_887_4/predict_loss/training_32_49/raw --save_path=log/training_0_15_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_887_4/predict_loss/training_32_49_800_800_800_800 --save_path=log/training_0_15_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_887_4/predict_loss/training_32_49/vis --n_classes=3 --start_id=0 --end_id=-1


### 24

### vis

CUDA_VISIBLE_DEVICES=2 python3 il_predict.py --images_path=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/training_0_23_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1274_4/weights_loss/ --save_path=log/training_0_23_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1274_4/predict_loss/training_32_49_800_800_80_320_rot_15_345_4_flip --save_stitched=1 --gpu_id=2 --loss_type=4

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/labels --seg_path=log/training_0_23_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1274_4/predict_loss/training_32_49_800_800_80_320_rot_15_345_4_flip/raw --save_path=log/vis/training_32_49_800_800_80_320_rot_15_345_4_flip/training_0_23_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1274_4 --n_classes=3 --start_id=0 --end_id=-1

#### no_aug

CUDA_VISIBLE_DEVICES=2 python3 il_predict.py --images_path=/data/617/images/training_32_49_800_800_800_800/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/training_0_23_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1274_4/weights_loss/ --save_path=log/training_0_23_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1274_4/predict_loss/training_32_49_800_800_800_800 --save_stitched=1 --gpu_id=2 --loss_type=4

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_800_800_800_800/images --labels_path=/data/617/images/training_32_49_800_800_800_800/labels --seg_path=log/training_0_23_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1274_4/predict_loss/training_32_49_800_800_800_800/raw --save_path=log/vis/training_32_49_800_800_800_800/training_0_23_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1274_4 --n_classes=3 --start_id=0 --end_id=-1

#### stitched

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/training_0_23_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1274_4/predict_loss/training_32_49_800_800_800_800 stitched_seq_path=log/training_0_23_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1274_4/predict_loss/training_32_49/raw patch_height=800 patch_width=800 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python3 ../visDataset.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_path=log/training_0_23_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1274_4/predict_loss/training_32_49/raw --save_path=log/training_0_23_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1274_4/predict_loss/training_32_49_800_800_800_800 --save_path=log/training_0_23_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1274_4/predict_loss/training_32_49/vis --n_classes=3 --start_id=0 --end_id=-1

### 32

### vis

CUDA_VISIBLE_DEVICES=2 python3 il_predict.py --images_path=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4/weights_loss/ --save_path=log/training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4/predict_loss/training_32_49_800_800_80_320_rot_15_345_4_flip --save_stitched=1 --gpu_id=2 --loss_type=4

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/labels --seg_path=log/training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4/predict_loss/training_32_49_800_800_80_320_rot_15_345_4_flip/raw --save_path=log/vis/training_32_49_800_800_80_320_rot_15_345_4_flip/vis_50_8000_8000_800_random_401_4 --n_classes=3 --start_id=0 --end_id=-1

#### no_aug

CUDA_VISIBLE_DEVICES=2 python3 il_predict.py --images_path=/data/617/images/training_32_49_800_800_800_800/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4/weights_loss/ --save_path=log/training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4/predict_loss/training_32_49_800_800_800_800 --save_stitched=1 --gpu_id=2 --loss_type=4

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_800_800_800_800/images --labels_path=/data/617/images/training_32_49_800_800_800_800/labels --seg_path=log/training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4/predict_loss/training_32_49_800_800_800_800/raw --save_path=log/vis/training_32_49_800_800_800_800/training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4 --n_classes=3 --start_id=0 --end_id=-1

#### stitched

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4/predict_loss/training_32_49_800_800_800_800 stitched_seq_path=log/training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4/predict_loss/training_32_49/raw patch_height=800 patch_width=800 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python3 ../visDataset.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_path=log/training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4/predict_loss/training_32_49/raw --save_path=log/training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4/predict_loss/training_32_49_800_800_800_800 --save_path=log/training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4/predict_loss/training_32_49/vis --n_classes=3 --start_id=0 --end_id=-1
