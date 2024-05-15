<!-- MarkdownTOC -->

- [build_data](#build_dat_a_)
    - [bf_r       @ build_data](#bf_r___build_data_)
    - [huh_r       @ build_data](#huh_r___build_data_)
    - [fluo_r       @ build_data](#fluo_r___build_data_)
    - [phc_r       @ build_data](#phc_r___build_data_)
- [hnasnet](#hnasnet_)
    - [atrous:6_12_18       @ hnasnet](#atrous_6_12_18___hnasne_t_)
        - [huh       @ atrous:6_12_18/hnasnet](#huh___atrous_6_12_18_hnasnet_)
            - [on_train       @ huh/atrous:6_12_18/hnasnet](#on_train___huh_atrous_6_12_18_hnasnet_)
        - [bf       @ atrous:6_12_18/hnasnet](#bf___atrous_6_12_18_hnasnet_)
        - [fluo       @ atrous:6_12_18/hnasnet](#fluo___atrous_6_12_18_hnasnet_)
        - [phc       @ atrous:6_12_18/hnasnet](#phc___atrous_6_12_18_hnasnet_)

<!-- /MarkdownTOC -->

<a id="build_dat_a_"></a>
# build_data

<a id="bf_r___build_data_"></a>
## bf_r       @ build_data-->new_deeplab_ctc
python36 datasets/build_ctc_data.py db_split=bf_r

<a id="huh_r___build_data_"></a>
## huh_r       @ build_data-->new_deeplab_ctc
python36 datasets/build_ctc_data.py db_split=huh_r
python36 datasets/build_ctc_data.py db_split=huh_e disable_seg=1

<a id="fluo_r___build_data_"></a>
## fluo_r       @ build_data-->new_deeplab_ctc
python36 datasets/build_ctc_data.py db_split=fluo_r

<a id="phc_r___build_data_"></a>
## phc_r       @ build_data-->new_deeplab_ctc
python36 datasets/build_ctc_data.py db_split=phc_r

<a id="hnasnet_"></a>
# hnasnet

<a id="atrous_6_12_18___hnasne_t_"></a>
## atrous:6_12_18       @ hnasnet-->new_deeplab_ctc

<a id="huh___atrous_6_12_18_hnasnet_"></a>
### huh       @ atrous:6_12_18/hnasnet-->new_deeplab_ctc
python36 new_deeplab_run.py cfg=gpu:0,_hnas_:atrous-6_12_18,_ctc_:train:huh-r:+++vis:huh-e,_train_:b2 start=2

<a id="on_train___huh_atrous_6_12_18_hnasnet_"></a>
#### on_train       @ huh/atrous:6_12_18/hnasnet-->new_deeplab_ctc
python36 new_deeplab_run.py cfg=gpu:0,_hnas_:atrous-6_12_18,_ctc_:train:huh-r:+++vis:huh-r,_train_:b2 start=1

python36 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_31/training_32_49_640_640_640_640/raw stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_31/training_32_49/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python36 ../visDataset.py images_path=/data/617/images/training_32_49/images labels_path=/data/617/images/training_32_49/labels seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_31/training_32_49/raw save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_31/training_32_49/vis n_classes=3 start_id=0 end_id=-1 normalize_labels=1

<a id="bf___atrous_6_12_18_hnasnet_"></a>
### bf       @ atrous:6_12_18/hnasnet-->new_deeplab_ctc
python36 new_deeplab_run.py cfg=gpu:1,_hnas_:atrous-6_12_18,_ctc_:train:bf-r:+++vis:bf-e,_train_:b2,_vis_:640

<a id="fluo___atrous_6_12_18_hnasnet_"></a>
### fluo       @ atrous:6_12_18/hnasnet-->new_deeplab_ctc
python36 new_deeplab_run.py cfg=gpu:2,_hnas_:atrous-6_12_18,_ctc_:train:fluo-r:+++vis:fluo-e,_train_:b2,_vis_:640

<a id="phc___atrous_6_12_18_hnasnet_"></a>
### phc       @ atrous:6_12_18/hnasnet-->new_deeplab_ctc
python36 new_deeplab_run.py cfg=gpu:0,_hnas_:atrous-6_12_18,_ctc_:train:phc-r:+++vis:phc-e,_train_:b2,_vis_:640 start=0
