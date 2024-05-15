set -x

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_height=256 patch_width=256 min_stride=256 max_stride=256 enable_rot=0 enable_flip=0 start_id=0 end_id=20 show_img=0

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_height=384 patch_width=384 min_stride=384 max_stride=384 enable_rot=0 enable_flip=0 start_id=0 end_id=20

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_height=512 patch_width=512 min_stride=512 max_stride=512 enable_rot=0 enable_flip=0 start_id=0 end_id=20

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_height=640 patch_width=640 min_stride=640 max_stride=640 enable_rot=0 enable_flip=0 start_id=0 end_id=20

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_height=800 patch_width=800 min_stride=800 max_stride=800 enable_rot=0 enable_flip=0 start_id=0 end_id=20 show_img=0
