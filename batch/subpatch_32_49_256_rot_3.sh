python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=256 patch_width=256 min_stride=25 max_stride=100 enable_rot=0 enable_flip=1 start_id=32 end_id=49
python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=256 patch_width=256 min_stride=25 max_stride=100 enable_rot=1 min_rot=15 max_rot=125 enable_flip=1 start_id=32 end_id=49
python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=256 patch_width=256 min_stride=25 max_stride=100 enable_rot=1 min_rot=126 max_rot=235 enable_flip=1 start_id=32 end_id=49
python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=256 patch_width=256 min_stride=25 max_stride=100 enable_rot=1 min_rot=236 max_rot=345 enable_flip=1 start_id=32 end_id=49

python3 mergeDatasets.py training_32_49_256_256_25_100_flip training_32_49_256_256_25_100_rot_15_125_235_345_flip start_id=32 end_id=49
python3 mergeDatasets.py training_32_49_256_256_25_100_rot_126_235_flip training_32_49_256_256_25_100_rot_15_125_235_345_flip start_id=32 end_id=49
python3 mergeDatasets.py training_32_49_256_256_25_100_rot_15_125_flip training_32_49_256_256_25_100_rot_15_125_235_345_flip start_id=32 end_id=49
python3 mergeDatasets.py training_32_49_256_256_25_100_rot_236_345_flip training_32_49_256_256_25_100_rot_15_125_235_345_flip start_id=32 end_id=49