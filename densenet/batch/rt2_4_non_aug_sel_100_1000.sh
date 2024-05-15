set -x

### sel 100

CUDA_VISIBLE_DEVICES=1 python3 il_train.py --train_images=/data/617/images/training_0_3_640_640_640_640/images --train_labels=/data/617/images/training_0_3_640_640_640_640/labels --test_images=/data/617/images/training_32_49_640_640_640_640/images --test_labels=/data/617/images/training_32_49_640_640_640_640/labels  --height=640 --width=640 --index_percent=0 --n_classes=3 --start_id=0 --end_id=-1 --n_epochs=1000 --gpu_id=2 --max_indices=100 --min_indices=100 --test_start_id=0 --test_end_id=-1 --save_stitched=1 --loss_type=4 --eval_every=1 --log_dir=log/rt2_training_0_3_640_640_640_640 --lr_dec_rate=0.95 --lr_dec_epochs=10 --psi_act_type=1 --load_weights=1 --preload_images=0

### sel 1000

CUDA_VISIBLE_DEVICES=1 python3 il_train.py --train_images=/data/617/images/training_0_3_640_640_640_640/images --train_labels=/data/617/images/training_0_3_640_640_640_640/labels --test_images=/data/617/images/training_32_49_640_640_640_640/images --test_labels=/data/617/images/training_32_49_640_640_640_640/labels  --height=640 --width=640 --index_percent=0 --n_classes=3 --start_id=0 --end_id=-1 --n_epochs=1000 --gpu_id=2 --max_indices=1000 --min_indices=1000 --test_start_id=0 --test_end_id=-1 --save_stitched=1 --loss_type=4 --eval_every=1 --log_dir=log/rt2_training_0_3_640_640_640_640 --lr_dec_rate=0.95 --lr_dec_epochs=10 --psi_act_type=1 --load_weights=1 --preload_images=0