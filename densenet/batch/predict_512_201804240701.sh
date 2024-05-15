set -x

CUDA_VISIBLE_DEVICES=0 python3 il_predict.py --images_path=/data/617/images/training_32_49_512_512_25_100_rot_15_345_4_flip/images --height=512 --width=512 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/50_10000_10000_512_random_501_4/weights/ --save_path=log/50_10000_10000_512_random_501_4/predict/training_32_49_512_512_25_100_rot_15_345_4_flip --save_stitched=1 --gpu_id=2 --loss_type=4

CUDA_VISIBLE_DEVICES=0 python3 il_predict.py --images_path=/data/617/images/validation_0_20_512_512_512_512/images --height=512 --width=512 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/50_10000_10000_512_random_501_4/weights/ --save_path=log/50_10000_10000_512_random_501_4/predict/validation_0_20_512_512_512_512 --save_stitched=1 --gpu_id=2 --loss_type=4