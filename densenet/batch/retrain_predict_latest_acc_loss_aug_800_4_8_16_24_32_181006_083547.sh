set -x

# acc

## 8


CUDA_VISIBLE_DEVICES=2 python3 il_predict.py --images_path=/data/617/images/training_32_49_800_800_800_800/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/training_0_7_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_581_4/weights_acc/ --save_path=log/training_0_7_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_581_4/predict_acc/training_32_49_800_800_800_800 --save_stitched=1 --gpu_id=2 --loss_type=4

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_800_800_800_800/images --labels_path=/data/617/images/training_32_49_800_800_800_800/labels --seg_path=log/training_0_7_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_581_4/predict_acc/training_32_49_800_800_800_800/raw --n_classes=3 --start_id=0 --end_id=-1

### 16


CUDA_VISIBLE_DEVICES=2 python3 il_predict.py --images_path=/data/617/images/training_32_49_800_800_800_800/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/training_0_15_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_887_4/weights_acc/ --save_path=log/training_0_15_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_887_4/predict_acc/training_32_49_800_800_800_800 --save_stitched=1 --gpu_id=2 --loss_type=4

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_800_800_800_800/images --labels_path=/data/617/images/training_32_49_800_800_800_800/labels --seg_path=log/training_0_15_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_887_4/predict_acc/training_32_49_800_800_800_800/raw --n_classes=3 --start_id=0 --end_id=-1


### 24

CUDA_VISIBLE_DEVICES=2 python3 il_predict.py --images_path=/data/617/images/training_32_49_800_800_800_800/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/training_0_23_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1274_4/weights_acc/ --save_path=log/training_0_23_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1274_4/predict_acc/training_32_49_800_800_800_800 --save_stitched=1 --gpu_id=2 --loss_type=4

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_800_800_800_800/images --labels_path=/data/617/images/training_32_49_800_800_800_800/labels --seg_path=log/training_0_23_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1274_4/predict_acc/training_32_49_800_800_800_800/raw --n_classes=3 --start_id=0 --end_id=-1


### 32

CUDA_VISIBLE_DEVICES=2 python3 il_predict.py --images_path=/data/617/images/training_32_49_800_800_800_800/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4/weights_acc/ --save_path=log/training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4/predict_acc/training_32_49_800_800_800_800 --save_stitched=1 --gpu_id=2 --loss_type=4

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_800_800_800_800/images --labels_path=/data/617/images/training_32_49_800_800_800_800/labels --seg_path=log/training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4/predict_acc/training_32_49_800_800_800_800/raw --n_classes=3 --start_id=0 --end_id=-1

# latest


## 8


CUDA_VISIBLE_DEVICES=2 python3 il_predict.py --images_path=/data/617/images/training_32_49_800_800_800_800/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/training_0_7_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_581_4/weights/ --save_path=log/training_0_7_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_581_4/predict/training_32_49_800_800_800_800 --save_stitched=1 --gpu_id=2 --loss_type=4

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_800_800_800_800/images --labels_path=/data/617/images/training_32_49_800_800_800_800/labels --seg_path=log/training_0_7_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_581_4/predict/training_32_49_800_800_800_800/raw --n_classes=3 --start_id=0 --end_id=-1

### 16


CUDA_VISIBLE_DEVICES=2 python3 il_predict.py --images_path=/data/617/images/training_32_49_800_800_800_800/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/training_0_15_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_887_4/weights/ --save_path=log/training_0_15_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_887_4/predict/training_32_49_800_800_800_800 --save_stitched=1 --gpu_id=2 --loss_type=4

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_800_800_800_800/images --labels_path=/data/617/images/training_32_49_800_800_800_800/labels --seg_path=log/training_0_15_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_887_4/predict/training_32_49_800_800_800_800/raw --n_classes=3 --start_id=0 --end_id=-1


### 24

CUDA_VISIBLE_DEVICES=2 python3 il_predict.py --images_path=/data/617/images/training_32_49_800_800_800_800/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/training_0_23_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1274_4/weights/ --save_path=log/training_0_23_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1274_4/predict/training_32_49_800_800_800_800 --save_stitched=1 --gpu_id=2 --loss_type=4

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_800_800_800_800/images --labels_path=/data/617/images/training_32_49_800_800_800_800/labels --seg_path=log/training_0_23_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1274_4/predict/training_32_49_800_800_800_800/raw --n_classes=3 --start_id=0 --end_id=-1


### 32

CUDA_VISIBLE_DEVICES=2 python3 il_predict.py --images_path=/data/617/images/training_32_49_800_800_800_800/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4/weights/ --save_path=log/training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4/predict/training_32_49_800_800_800_800 --save_stitched=1 --gpu_id=2 --loss_type=4

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_800_800_800_800/images --labels_path=/data/617/images/training_32_49_800_800_800_800/labels --seg_path=log/training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4/predict/training_32_49_800_800_800_800/raw --n_classes=3 --start_id=0 --end_id=-1




# loss


## 8


CUDA_VISIBLE_DEVICES=2 python3 il_predict.py --images_path=/data/617/images/training_32_49_800_800_800_800/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/training_0_7_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_581_4/weights_loss/ --save_path=log/training_0_7_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_581_4/predict_loss/training_32_49_800_800_800_800 --save_stitched=1 --gpu_id=2 --loss_type=4

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_800_800_800_800/images --labels_path=/data/617/images/training_32_49_800_800_800_800/labels --seg_path=log/training_0_7_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_581_4/predict_loss/training_32_49_800_800_800_800/raw --n_classes=3 --start_id=0 --end_id=-1

### 16


CUDA_VISIBLE_DEVICES=2 python3 il_predict.py --images_path=/data/617/images/training_32_49_800_800_800_800/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/training_0_15_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_887_4/weights_loss/ --save_path=log/training_0_15_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_887_4/predict_loss/training_32_49_800_800_800_800 --save_stitched=1 --gpu_id=2 --loss_type=4

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_800_800_800_800/images --labels_path=/data/617/images/training_32_49_800_800_800_800/labels --seg_path=log/training_0_15_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_887_4/predict_loss/training_32_49_800_800_800_800/raw --n_classes=3 --start_id=0 --end_id=-1


### 24

CUDA_VISIBLE_DEVICES=2 python3 il_predict.py --images_path=/data/617/images/training_32_49_800_800_800_800/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/training_0_23_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1274_4/weights_loss/ --save_path=log/training_0_23_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1274_4/predict_loss/training_32_49_800_800_800_800 --save_stitched=1 --gpu_id=2 --loss_type=4

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_800_800_800_800/images --labels_path=/data/617/images/training_32_49_800_800_800_800/labels --seg_path=log/training_0_23_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1274_4/predict_loss/training_32_49_800_800_800_800/raw --n_classes=3 --start_id=0 --end_id=-1


### 32

CUDA_VISIBLE_DEVICES=2 python3 il_predict.py --images_path=/data/617/images/training_32_49_800_800_800_800/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4/weights_loss/ --save_path=log/training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4/predict_loss/training_32_49_800_800_800_800 --save_stitched=1 --gpu_id=2 --loss_type=4

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_800_800_800_800/images --labels_path=/data/617/images/training_32_49_800_800_800_800/labels --seg_path=log/training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4/predict_loss/training_32_49_800_800_800_800/raw --n_classes=3 --start_id=0 --end_id=-1





