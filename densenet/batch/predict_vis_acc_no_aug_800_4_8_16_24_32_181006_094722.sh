set -x

### 4

#### no_aug

CUDA_VISIBLE_DEVICES=2 python3 il_predict.py --images_path=/data/617/images/training_32_49_800_800_800_800/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/50_8000_8000_800_0_320_4/weights/ --save_path=log/50_8000_8000_800_0_320_4/predict/training_32_49_800_800_800_800 --save_stitched=1 --gpu_id=2 --loss_type=4

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_800_800_800_800/images --labels_path=/data/617/images/training_32_49_800_800_800_800/labels --seg_path=log/50_8000_8000_800_0_320_4/predict/training_32_49_800_800_800_800/raw --n_classes=3 --start_id=0 --end_id=-1

#### stitched

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/50_8000_8000_800_0_320_4/predict/training_32_49_800_800_800_800 stitched_seq_path=log/50_8000_8000_800_0_320_4/predict/training_32_49/raw patch_height=800 patch_width=800 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png del_patch_seq=0

python3 ../visDataset.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_path=log/50_8000_8000_800_0_320_4/predict/training_32_49/raw --n_classes=3 --start_id=0 --end_id=-1

### 8

#### no_aug

CUDA_VISIBLE_DEVICES=2 python3 il_predict.py --images_path=/data/617/images/training_32_49_800_800_800_800/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/50_8000_8000_800_0_431_4/weights/ --save_path=log/50_8000_8000_800_0_431_4/predict/training_32_49_800_800_800_800 --save_stitched=1 --gpu_id=2 --loss_type=4

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_800_800_800_800/images --labels_path=/data/617/images/training_32_49_800_800_800_800/labels --seg_path=log/50_8000_8000_800_0_431_4/predict/training_32_49_800_800_800_800/raw --n_classes=3 --start_id=0 --end_id=-1

#### stitched

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/50_8000_8000_800_0_431_4/predict/training_32_49_800_800_800_800 stitched_seq_path=log/50_8000_8000_800_0_431_4/predict/training_32_49/raw patch_height=800 patch_width=800 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png del_patch_seq=0

python3 ../visDataset.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_path=log/50_8000_8000_800_0_431_4/predict/training_32_49/raw --n_classes=3 --start_id=0 --end_id=-1

### 16

#### no_aug

CUDA_VISIBLE_DEVICES=2 python3 il_predict.py --images_path=/data/617/images/training_32_49_800_800_800_800/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/50_8000_8000_800_0_920_4/weights/ --save_path=log/50_8000_8000_800_0_920_4/predict/training_32_49_800_800_800_800 --save_stitched=1 --gpu_id=2 --loss_type=4

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_800_800_800_800/images --labels_path=/data/617/images/training_32_49_800_800_800_800/labels --seg_path=log/50_8000_8000_800_0_920_4/predict/training_32_49_800_800_800_800/raw --n_classes=3 --start_id=0 --end_id=-1

#### stitched

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/50_8000_8000_800_0_920_4/predict/training_32_49_800_800_800_800 stitched_seq_path=log/50_8000_8000_800_0_920_4/predict/training_32_49/raw patch_height=800 patch_width=800 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png del_patch_seq=0

python3 ../visDataset.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_path=log/50_8000_8000_800_0_920_4/predict/training_32_49/raw --n_classes=3 --start_id=0 --end_id=-1

### 24

#### no_aug

CUDA_VISIBLE_DEVICES=2 python3 il_predict.py --images_path=/data/617/images/training_32_49_800_800_800_800/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/50_8000_8000_800_0_1505_4/weights/ --save_path=log/50_8000_8000_800_0_1505_4/predict/training_32_49_800_800_800_800 --save_stitched=1 --gpu_id=2 --loss_type=4

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_800_800_800_800/images --labels_path=/data/617/images/training_32_49_800_800_800_800/labels --seg_path=log/50_8000_8000_800_0_1505_4/predict/training_32_49_800_800_800_800/raw --n_classes=3 --start_id=0 --end_id=-1

#### stitched

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/50_8000_8000_800_0_1505_4/predict/training_32_49_800_800_800_800 stitched_seq_path=log/50_8000_8000_800_0_1505_4/predict/training_32_49/raw patch_height=800 patch_width=800 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png del_patch_seq=0

python3 ../visDataset.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_path=log/50_8000_8000_800_0_1505_4/predict/training_32_49/raw --n_classes=3 --start_id=0 --end_id=-1

### 32

#### no_aug

CUDA_VISIBLE_DEVICES=2 python3 il_predict.py --images_path=/data/617/images/training_32_49_800_800_800_800/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/50_8000_8000_800_0_1586_4/weights/ --save_path=log/50_8000_8000_800_0_1586_4/predict/training_32_49_800_800_800_800 --save_stitched=1 --gpu_id=2 --loss_type=4

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_800_800_800_800/images --labels_path=/data/617/images/training_32_49_800_800_800_800/labels --seg_path=log/50_8000_8000_800_0_1586_4/predict/training_32_49_800_800_800_800/raw --n_classes=3 --start_id=0 --end_id=-1

#### stitched

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/50_8000_8000_800_0_1586_4/predict/training_32_49_800_800_800_800 stitched_seq_path=log/50_8000_8000_800_0_1586_4/predict/training_32_49/raw patch_height=800 patch_width=800 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png del_patch_seq=0

python3 ../visDataset.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_path=log/50_8000_8000_800_0_1586_4/predict/training_32_49/raw --n_classes=3 --start_id=0 --end_id=-1