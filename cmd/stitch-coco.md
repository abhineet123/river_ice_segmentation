<!-- MarkdownTOC -->

- [val-rfm](#val_rfm_)
    - [r-640-lac-sub-8-fbb       @ val-rfm](#r_640_lac_sub_8_fbb___val_rf_m_)
        - [on-val       @ r-640-lac-sub-8-fbb/val-rfm](#on_val___r_640_lac_sub_8_fbb_val_rf_m_)
- [val](#val_)
    - [r-640-p-640-lac-sub-8-fbb       @ val](#r_640_p_640_lac_sub_8_fbb___va_l_)
        - [on-val       @ r-640-p-640-lac-sub-8-fbb/val](#on_val___r_640_p_640_lac_sub_8_fbb_va_l_)
- [train](#train_)
    - [r-640-p-640-lac-sub-8-fbb       @ train](#r_640_p_640_lac_sub_8_fbb___trai_n_)
        - [on-train-end-2000       @ r-640-p-640-lac-sub-8-fbb/train](#on_train_end_2000___r_640_p_640_lac_sub_8_fbb_trai_n_)
            - [batch-16       @ on-train-end-2000/r-640-p-640-lac-sub-8-fbb/train](#batch_16___on_train_end_2000_r_640_p_640_lac_sub_8_fbb_trai_n_)
        - [on-val       @ r-640-p-640-lac-sub-8-fbb/train](#on_val___r_640_p_640_lac_sub_8_fbb_trai_n_)
            - [batch-16       @ on-val/r-640-p-640-lac-sub-8-fbb/train](#batch_16___on_val_r_640_p_640_lac_sub_8_fbb_train_)
    - [r-1280-p-640-mc-sub-4-fbb       @ train](#r_1280_p_640_mc_sub_4_fbb___trai_n_)
        - [on-train-end-600       @ r-1280-p-640-mc-sub-4-fbb/train](#on_train_end_600___r_1280_p_640_mc_sub_4_fbb_trai_n_)
        - [on-val-end-500       @ r-1280-p-640-mc-sub-4-fbb/train](#on_val_end_500___r_1280_p_640_mc_sub_4_fbb_trai_n_)
        - [on-val       @ r-1280-p-640-mc-sub-4-fbb/train](#on_val___r_1280_p_640_mc_sub_4_fbb_trai_n_)

<!-- /MarkdownTOC -->

<a id="val_rfm_"></a>
# val-rfm
<a id="r_640_lac_sub_8_fbb___val_rf_m_"></a>
## r-640-lac-sub-8-fbb       @ val-rfm-->stitch-coco
<a id="on_val___r_640_lac_sub_8_fbb_val_rf_m_"></a>
### on-val       @ r-640-lac-sub-8-fbb/val-rfm-->stitch-coco
python stitch.py cfg=coco:val:r-640:batch-8:sub-8:rfm:lac:voc18:seq2k:vis-0:_in_-resnet_640_semantic_val2017-coco_semantic_val2017-rfm-batch_4-res_640-rot-crop-flip-seq2k-fbb-voc18/ckpt-__var__:_out_-coco-rfm-val-r-640-lac-sub-8-fbb

<a id="val_"></a>
# val
<a id="r_640_p_640_lac_sub_8_fbb___va_l_"></a>
## r-640-p-640-lac-sub-8-fbb       @ val-->stitch-coco
<a id="on_val___r_640_p_640_lac_sub_8_fbb_va_l_"></a>
### on-val       @ r-640-p-640-lac-sub-8-fbb/val-->stitch-coco
python stitch.py cfg=coco:val:p-640:r-640:batch-8:sub-8:lac:voc18:seq2k:vis-0:_in_-resnet_640_resize_640-0_4999-640_640-640_640-sub_8-lac-coco_semantic_val2017-batch_64-seq2k-fbb-voc18-self2-1/ckpt-__var__:_out_-coco-val-r-640-p-640-lac-sub-8-fbb

<a id="train_"></a>
# train
<a id="r_640_p_640_lac_sub_8_fbb___trai_n_"></a>
## r-640-p-640-lac-sub-8-fbb       @ train-->stitch-coco
<a id="on_train_end_2000___r_640_p_640_lac_sub_8_fbb_trai_n_"></a>
### on-train-end-2000       @ r-640-p-640-lac-sub-8-fbb/train-->stitch-coco
python stitch.py cfg=coco:train:p-640:r-640:batch-12:sub-8:end-2000:lac:voc18:seq2k:vis-0:_in_-resnet_640_resize_640-0_118286-640_640-640_640-sub_8-lac-coco_semantic_train2017-batch_64-seq2k-fbb-voc18-self2-0/ckpt-__var__:_out_-coco-train-end-2k-r-640-p-640-lac-sub-8-fbb-fixed:save-0
`camiou`
python stitch.py cfg=coco:train:p-640:r-640:batch-12:sub-8:end-2000:lac:voc18:seq2k:vis-0:_in_-resnet_640_resize_640-0_118286-640_640-640_640-sub_8-lac-coco_semantic_train2017-batch_64-seq2k-fbb-voc18-self2-0/ckpt-__var__:_out_-coco-train-end-2k-r-640-p-640-lac-sub-8-fbb-camiou:save-0
<a id="batch_16___on_train_end_2000_r_640_p_640_lac_sub_8_fbb_trai_n_"></a>
#### batch-16       @ on-train-end-2000/r-640-p-640-lac-sub-8-fbb/train-->stitch-coco
python stitch.py cfg=coco:train:p-640:r-640:batch-16:sub-8:end-2000:lac:voc18:seq2k:vis-0:_in_-resnet_640_resize_640-0_118286-640_640-640_640-sub_8-lac-coco_semantic_train2017-batch_64-seq2k-fbb-voc18-self2-0/ckpt-__var__:_out_-coco-train-end-2k-r-640-p-640-lac-sub-8-fbb-fixed:save-0
`camiou`
python stitch.py cfg=coco:train:p-640:r-640:batch-16:sub-8:end-2000:lac:voc18:seq2k:vis-0:_in_-resnet_640_resize_640-0_118286-640_640-640_640-sub_8-lac-coco_semantic_train2017-batch_64-seq2k-fbb-voc18-self2-0/ckpt-__var__:_out_-coco-train-end-2k-r-640-p-640-lac-sub-8-fbb-camiou:save-0
<a id="on_val___r_640_p_640_lac_sub_8_fbb_trai_n_"></a>
### on-val       @ r-640-p-640-lac-sub-8-fbb/train-->stitch-coco
python stitch.py cfg=coco:val:p-640:r-640:batch-32:sub-8:lac:voc18:seq2k:vis-0:_in_-resnet_640_resize_640-0_118286-640_640-640_640-sub_8-lac-coco_semantic_train2017-batch_64-seq2k-fbb-voc18-self2-0/ckpt-__var__:_out_-coco-val-r-640-p-640-lac-sub-8-fbb-fixed:save-0
`camiou`
python stitch.py cfg=coco:val:p-640:r-640:batch-32:sub-8:lac:voc18:seq2k:vis-0:_in_-resnet_640_resize_640-0_118286-640_640-640_640-sub_8-lac-coco_semantic_train2017-batch_64-seq2k-fbb-voc18-self2-0/ckpt-__var__:_out_-coco-val-r-640-p-640-lac-sub-8-fbb-camiou:save-0
<a id="batch_16___on_val_r_640_p_640_lac_sub_8_fbb_train_"></a>
#### batch-16       @ on-val/r-640-p-640-lac-sub-8-fbb/train-->stitch-coco
python stitch.py cfg=coco:val:p-640:r-640:batch-16:sub-8:lac:voc18:seq2k:vis-0:_in_-resnet_640_resize_640-0_118286-640_640-640_640-sub_8-lac-coco_semantic_train2017-batch_64-seq2k-fbb-voc18-self2-0/ckpt-__var__:_out_-coco-val-r-640-p-640-lac-sub-8-fbb-fixed:save-0
`camiou`
python stitch.py cfg=coco:val:p-640:r-640:batch-16:sub-8:lac:voc18:seq2k:vis-0:_in_-resnet_640_resize_640-0_118286-640_640-640_640-sub_8-lac-coco_semantic_train2017-batch_64-seq2k-fbb-voc18-self2-0/ckpt-__var__:_out_-coco-val-r-640-p-640-lac-sub-8-fbb-camiou:save-0

<a id="r_1280_p_640_mc_sub_4_fbb___trai_n_"></a>
## r-1280-p-640-mc-sub-4-fbb       @ train-->stitch-coco
<a id="on_train_end_600___r_1280_p_640_mc_sub_4_fbb_trai_n_"></a>
### on-train-end-600       @ r-1280-p-640-mc-sub-4-fbb/train-->stitch-coco
python stitch.py cfg=coco:train:p-640:r-1280:batch-8:sub-4:end-600:voc28:seq4k:vis-0:_in_-resnet_640_resize_1280-0_118286-640_640-640_640-sub_4-mc-coco_semantic_train2017-batch_24-seq4k-voc28-fbb-zedg/ckpt-__var__:_out_-coco-train-end-600-r-1280-p-640-mc-sub-4-seq4k-voc28-fbb:save-0
`camiou`
python stitch.py cfg=coco:train:p-640:r-1280:batch-8:sub-4:end-600:voc28:seq4k:vis-0:_in_-resnet_640_resize_1280-0_118286-640_640-640_640-sub_4-mc-coco_semantic_train2017-batch_24-seq4k-voc28-fbb-zedg/ckpt-__var__:_out_-coco-train-end-600-r-1280-p-640-mc-sub-4-seq4k-voc28-fbb-camiou:save-0
<a id="on_val_end_500___r_1280_p_640_mc_sub_4_fbb_trai_n_"></a>
### on-val-end-500       @ r-1280-p-640-mc-sub-4-fbb/train-->stitch-coco
python stitch.py cfg=coco:val:p-640:r-1280:batch-8:sub-4:end-500:voc28:seq4k:vis-0:_in_-resnet_640_resize_1280-0_118286-640_640-640_640-sub_4-mc-coco_semantic_train2017-batch_24-seq4k-voc28-fbb-zedg/ckpt-__var__:_out_-coco-val-end-500-r-1280-p-640-mc-sub-4-seq4k-voc28-fbb:save-0
`camiou`
python stitch.py cfg=coco:val:p-640:r-1280:batch-8:sub-4:end-500:voc28:seq4k:vis-0:_in_-resnet_640_resize_1280-0_118286-640_640-640_640-sub_4-mc-coco_semantic_train2017-batch_24-seq4k-voc28-fbb-zedg/ckpt-__var__:_out_-coco-val-end-500-r-1280-p-640-mc-sub-4-seq4k-voc28-fbb-camiou:save-0
<a id="on_val___r_1280_p_640_mc_sub_4_fbb_trai_n_"></a>
### on-val       @ r-1280-p-640-mc-sub-4-fbb/train-->stitch-coco
`camiou`
python stitch.py cfg=coco:val:p-640:r-1280:batch-16:sub-4:voc28:seq4k:vis-0:_in_-resnet_640_resize_1280-0_118286-640_640-640_640-sub_4-mc-coco_semantic_train2017-batch_24-seq4k-voc28-fbb-zedg/ckpt-__var__:_out_-coco-val-r-1280-p-640-mc-sub-4-seq4k-voc28-fbb-camiou-vis:save-1

