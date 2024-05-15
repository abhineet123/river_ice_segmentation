set -x 
python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=training_0_49_800_800_80_320_rot_15_345_4_flip --image_format=png --label_format=png --output_dir=training_0_31_49_800_800_80_320_rot_15_345_4_flip

python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=training_0_31_800_800_80_320_rot_15_345_4_flip --image_format=png --label_format=png --output_dir=training_0_31_49_800_800_80_320_rot_15_345_4_flip

python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=training_32_49_800_800_80_320_rot_15_345_4_flip --image_format=png --label_format=png --output_dir=training_0_31_49_800_800_80_320_rot_15_345_4_flip