<!-- MarkdownTOC -->

- [sub_patch](#sub_patch_)
    - [640       @ sub_patch](#640___sub_patc_h_)

<!-- /MarkdownTOC -->

<a id="sub_patch_"></a>
# sub_patch
<a id="640___sub_patc_h_"></a>
## 640       @ sub_patch-->cell_segm

python36 subPatchBatch.py cfg=_subpatch_:ctc:fluo-r:size-640:smin-64:smax-256:rmin-15:rmax-345:rnum-4:flip:log:seq-1


 python3 subPatchDataset.py db_root_dir=/data/CTC seq_name=Fluo-C2DL-Huh7_01 img_ext=jpg labels_ext=png out_ext=png patch_height=640 patch_width=640 min_stride=64 max_stride=256 enable_flip=1 start_id=0 end_id=29 n_frames=30 show_img=0 out_seq_name=Fluo-C2DL-Huh7_01_0_29_640_640_64_256_rot_15_345_4_flip src_path=/data/CTC/Images/Fluo-C2DL-Huh7_01 labels_path=/data/CTC/Labels_PNG/Fluo-C2DL-Huh7_01 enable_rot=0
