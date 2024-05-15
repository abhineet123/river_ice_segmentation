set -x

CUDA_VISIBLE_DEVICES=1 python3 il_predict.py --images_path=/data/617/images/training_32_49_384_384_25_100_rot_15_345_4_flip/images --height=384 --width=384 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/50_10000_10000_384_random_601_4/weights/ --save_path=log/50_10000_10000_384_random_601_4/predict/training_32_49_384_384_25_100_rot_15_345_4_flip --save_stitched=1 --gpu_id=2 --loss_type=4

CUDA_VISIBLE_DEVICES=1 python3 il_predict.py --images_path=/data/617/images/validation_0_20_384_384_384_384/images --height=384 --width=384 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/50_10000_10000_384_random_601_4/weights/ --save_path=log/50_10000_10000_384_random_601_4/predict/validation_0_20_384_384_384_384 --save_stitched=1 --gpu_id=2 --loss_type=4