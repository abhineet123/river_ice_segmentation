<!-- MarkdownTOC -->

- [train](#train_)
    - [r-640-sub-8-dm2-fbb       @ train](#r_640_sub_8_dm2_fbb___trai_n_)
        - [on-train       @ r-640-sub-8-dm2-fbb/train](#on_train___r_640_sub_8_dm2_fbb_trai_n_)
        - [on-val       @ r-640-sub-8-dm2-fbb/train](#on_val___r_640_sub_8_dm2_fbb_trai_n_)
    - [r-640-sub-8-lac-fbb       @ train](#r_640_sub_8_lac_fbb___trai_n_)
        - [on-train       @ r-640-sub-8-lac-fbb/train](#on_train___r_640_sub_8_lac_fbb_trai_n_)
        - [on-val       @ r-640-sub-8-lac-fbb/train](#on_val___r_640_sub_8_lac_fbb_trai_n_)
            - [gt_80       @ on-val/r-640-sub-8-lac-fbb/train](#gt_80___on_val_r_640_sub_8_lac_fbb_train_)
    - [r-1280_640-p-640-sub-8-lac-fbb       @ train](#r_1280_640_p_640_sub_8_lac_fbb___trai_n_)
        - [on-val       @ r-1280_640-p-640-sub-8-lac-fbb/train](#on_val___r_1280_640_p_640_sub_8_lac_fbb_train_)
            - [gt_80       @ on-val/r-1280_640-p-640-sub-8-lac-fbb/train](#gt_80___on_val_r_1280_640_p_640_sub_8_lac_fbb_trai_n_)
- [train-rfm](#train_rfm_)
    - [p-1024-sub-8-lac-fbb       @ train-rfm](#p_1024_sub_8_lac_fbb___train_rf_m_)
        - [on-train       @ p-1024-sub-8-lac-fbb/train-rfm](#on_train___p_1024_sub_8_lac_fbb_train_rfm_)
        - [on-val       @ p-1024-sub-8-lac-fbb/train-rfm](#on_val___p_1024_sub_8_lac_fbb_train_rfm_)
            - [ssgt       @ on-val/p-1024-sub-8-lac-fbb/train-rfm](#ssgt___on_val_p_1024_sub_8_lac_fbb_train_rf_m_)
    - [r-1280_640-p-640-sub-8-mhd-fbb       @ train-rfm](#r_1280_640_p_640_sub_8_mhd_fbb___train_rf_m_)
        - [on-val       @ r-1280_640-p-640-sub-8-mhd-fbb/train-rfm](#on_val___r_1280_640_p_640_sub_8_mhd_fbb_train_rfm_)
    - [r-1280_640-p-640-sub-8-mhd_1241-fbb       @ train-rfm](#r_1280_640_p_640_sub_8_mhd_1241_fbb___train_rf_m_)
        - [on-val       @ r-1280_640-p-640-sub-8-mhd_1241-fbb/train-rfm](#on_val___r_1280_640_p_640_sub_8_mhd_1241_fbb_train_rf_m_)
    - [r-1280_640-p-640-sub-8-mhd_1241-fbb-b54-zeg       @ train-rfm](#r_1280_640_p_640_sub_8_mhd_1241_fbb_b54_zeg___train_rf_m_)
        - [on-val       @ r-1280_640-p-640-sub-8-mhd_1241-fbb-b54-zeg/train-rfm](#on_val___r_1280_640_p_640_sub_8_mhd_1241_fbb_b54_zeg_train_rf_m_)

<!-- /MarkdownTOC -->


<a id="train_"></a>
# train
<a id="r_640_sub_8_dm2_fbb___trai_n_"></a>
## r-640-sub-8-dm2-fbb       @ train-->stitch-ctscp
<a id="on_train___r_640_sub_8_dm2_fbb_trai_n_"></a>
### on-train       @ r-640-sub-8-dm2-fbb/train-->stitch-ctscp
python stitch.py cfg=ctscp:train:r-640:sub-8:dm2:batch-48:logits:vis-0:img:_in_-resnet_640_ctscp-train-resize_640-sub_8-dm2-mc-batch_48-seq3k-voc7k-fbb-gdez/ckpt-__var__:_out_-ctscp-r-640-sub-8-dm2-fbb-train
```
resnet_640_ctscp-train-resize_640-sub_8-dm2-mc-batch_48-seq3k-voc7k-fbb-gdez/ckpt-*-ctscp-train-resize_640-sub_8-dm2-mc/masks-batch_48

```
<a id="on_val___r_640_sub_8_dm2_fbb_trai_n_"></a>
### on-val       @ r-640-sub-8-dm2-fbb/train-->stitch-ctscp
python stitch.py cfg=ctscp:val:r-640:sub-8:dm2:batch-48:logits:vis-0:img:_in_-resnet_640_ctscp-train-resize_640-sub_8-dm2-mc-batch_48-seq3k-voc7k-fbb-gdez/ckpt-__var__:_out_-ctscp-r-640-sub-8-dm2-fbb
```
resnet_640_ctscp-train-resize_640-sub_8-dm2-mc-batch_48-seq3k-voc7k-fbb-gdez/ckpt-*-ctscp-val-resize_640-sub_8-dm2-mc/masks-batch_48

```
<a id="r_640_sub_8_lac_fbb___trai_n_"></a>
## r-640-sub-8-lac-fbb       @ train-->stitch-ctscp
<a id="on_train___r_640_sub_8_lac_fbb_trai_n_"></a>
### on-train       @ r-640-sub-8-lac-fbb/train-->stitch-ctscp
python stitch.py cfg=ctscp:train:r-640:sub-8:lac:batch-32:logits:vis-0:img:_in_-resnet_640_ctscp-train-resize_640-sub_8-lac-batch_40-seq3k-voc20-fbb-gdez/ckpt-__var__:_out_-ctscp-r-640-sub-8-lac-fbb-train
```
resnet_640_ctscp-train-resize_640-sub_8-lac-batch_40-seq3k-voc20-fbb-gdez/ckpt-*-ctscp-val-resize_640-sub_8-lac/masks-batch_32
```
<a id="on_val___r_640_sub_8_lac_fbb_trai_n_"></a>
### on-val       @ r-640-sub-8-lac-fbb/train-->stitch-ctscp
python stitch.py cfg=ctscp:val:r-640:sub-8:lac:batch-32:logits:vis-0:img:_in_-resnet_640_ctscp-train-resize_640-sub_8-lac-batch_40-seq3k-voc20-fbb-gdez/ckpt-__var__:_out_-ctscp-r-640-sub-8-lac-fbb
<a id="gt_80___on_val_r_640_sub_8_lac_fbb_train_"></a>
#### gt_80       @ on-val/r-640-sub-8-lac-fbb/train-->stitch-ctscp
python stitch.py cfg=ctscp:val:r-80:batch-32:logits:vis-0:img:_in_-resnet_640_ctscp-train-resize_640-sub_8-lac-batch_40-seq3k-voc20-fbb-gdez/ckpt-86950-ctscp-val-resize_640-sub_8-lac/:_out_-ctscp-r-640-sub-8-lac-fbb-gt_80:dbg

<a id="r_1280_640_p_640_sub_8_lac_fbb___trai_n_"></a>
## r-1280_640-p-640-sub-8-lac-fbb       @ train-->stitch-ctscp
<a id="on_train___r_1280_640_p_640_sub_8_lac_fbb_train_"></a>
<a id="on_val___r_1280_640_p_640_sub_8_lac_fbb_train_"></a>
### on-val       @ r-1280_640-p-640-sub-8-lac-fbb/train-->stitch-ctscp
python stitch.py cfg=ctscp:val:r-1280_640:p-640:sub-8:lac:batch-16:logits:vis-0:img:_in_-resnet_640_ctscp-train-resize_1280x640-640_640-640_640-sub_8-lac-batch_80-seq2k-voc8192-fbb-gdez/ckpt-__var__:_out_-ctscp-r-1280_640-p-640-sub-8-lac-fbb
```
resnet_640_ctscp-train-resize_1280x640-640_640-640_640-sub_8-lac-batch_80-seq2k-voc8192-fbb-gdez/ckpt-*-ctscp-val-resize_1280x640-640_640-640_640-sub_8-lac/masks-batch_16
```
<a id="gt_80___on_val_r_1280_640_p_640_sub_8_lac_fbb_trai_n_"></a>
#### gt_80       @ on-val/r-1280_640-p-640-sub-8-lac-fbb/train-->stitch-ctscp
python stitch.py cfg=ctscp:val:r-160_80:p-80:lac:batch-16:logits:vis-0:img:_in_-resnet_640_ctscp-train-resize_1280x640-640_640-640_640-sub_8-lac-batch_80-seq2k-voc8192-fbb-gdez/ckpt-152810-ctscp-val-resize_1280x640-640_640-640_640-sub_8-lac/:_out_-ctscp-r-1280_640-p-640-sub-8-lac-fbb-gt_80:dbg


<a id="train_rfm_"></a>
# train-rfm
<a id="p_1024_sub_8_lac_fbb___train_rf_m_"></a>
## p-1024-sub-8-lac-fbb       @ train-rfm-->stitch-ctscp
<a id="on_train___p_1024_sub_8_lac_fbb_train_rfm_"></a>
### on-train       @ p-1024-sub-8-lac-fbb/train-rfm-->stitch-ctscp
python stitch.py cfg=ctscp:train:p-1024:sub-8:lac:batch-32:vis-0:img:_in_-resnet_1024_train-1024_1024-1024_1024-ctscp_train-rfm-rot-flip-batch_32-seq3k-fbb-voc20-cls_eq_1-zedg/ckpt-__var__:_out_-ctscp-p-1024-rfm-sub-8-lac-cls_eq-fbb:dbg
`seq-0`
```
log/seg/resnet_1024_train-1024_1024-1024_1024-ctscp_train-rfm-rot-flip-batch_32-seq3k-fbb-voc20-cls_eq_1-zedg/ckpt-129828-ctscp-train-1024_1024-1024_1024-seq_0_0-sub_8-lac/vid_info.json.gz
```
python stitch.py cfg=ctscp:train:p-1024:sub-8:lac:batch-32:vis-0:img:_in_-resnet_1024_train-1024_1024-1024_1024-ctscp_train-rfm-rot-flip-batch_32-seq3k-fbb-voc20-cls_eq_1-zedg/ckpt-__var__:_out_-ctscp-p-1024-rfm-sub-8-lac-cls_eq-fbb:dbg:seq-0


<a id="on_val___p_1024_sub_8_lac_fbb_train_rfm_"></a>
### on-val       @ p-1024-sub-8-lac-fbb/train-rfm-->stitch-ctscp
python stitch.py cfg=ctscp:val:p-1024:sub-8:lac:batch-32:vis-0:img:_in_-resnet_1024_train-1024_1024-1024_1024-ctscp_train-rfm-rot-flip-batch_32-seq3k-fbb-voc20-cls_eq_1-zedg/ckpt-__var__:_out_-ctscp-p-1024-rfm-sub-8-lac-cls_eq-fbb-rerun:dbg
`dbg`
python stitch.py cfg=ctscp:val:p-1024:sub-8:lac:batch-32:vis-1:img:_in_-resnet_1024_train-1024_1024-1024_1024-ctscp_train-rfm-rot-flip-batch_32-seq3k-fbb-voc20-cls_eq_1-zedg/ckpt-__var__:_out_-ctscp-p-1024-rfm-sub-8-lac-cls_eq-fbb-dbg:dbg

<a id="ssgt___on_val_p_1024_sub_8_lac_fbb_train_rf_m_"></a>
#### ssgt       @ on-val/p-1024-sub-8-lac-fbb/train-rfm-->stitch-ctscp
python stitch.py cfg=ctscp:val:p-1024:sub-8:lac:batch-32:vis-0:img:_in_-resnet_1024_train-1024_1024-1024_1024-ctscp_train-rfm-rot-flip-batch_32-seq3k-fbb-voc20-cls_eq_1-zedg/ckpt-__var__:_out_-ctscp-p-1024-rfm-sub-8-lac-cls_eq-fbb-ssgt-dbg:ssgt:dbg

<a id="r_1280_640_p_640_sub_8_mhd_fbb___train_rf_m_"></a>
## r-1280_640-p-640-sub-8-mhd-fbb       @ train-rfm-->stitch-ctscp
<a id="on_val___r_1280_640_p_640_sub_8_mhd_fbb_train_rfm_"></a>
### on-val       @ r-1280_640-p-640-sub-8-mhd-fbb/train-rfm-->stitch-ctscp
python stitch.py cfg=ctscp:val:r-1280_640:p-640:sub-8:mhd:batch-32:vis-0:img:_in_-resnet_640_ctscp-train-resize_1280x640-640_640-640_640-mhd-rfm-rot-flip-batch_72-seq1k-fbb-gdez/ckpt-__var__:_out_-ctscp-r-1280_640-p-640-mhd-rfm-sub-8-fbb
```
resnet_640_ctscp-train-resize_1280x640-640_640-640_640-mhd-rfm-rot-flip-batch_72-seq1k-fbb-gdez/ckpt-*-ctscp-val-resize_1280x640-640_640-640_640-sub_8-2d-mc/masks-batch_32
```


<a id="r_1280_640_p_640_sub_8_mhd_1241_fbb___train_rf_m_"></a>
## r-1280_640-p-640-sub-8-mhd_1241-fbb       @ train-rfm-->stitch-ctscp
<a id="on_val___r_1280_640_p_640_sub_8_mhd_1241_fbb_train_rf_m_"></a>
### on-val       @ r-1280_640-p-640-sub-8-mhd_1241-fbb/train-rfm-->stitch-ctscp
python stitch.py cfg=ctscp:val:r-1280_640:p-640:sub-8:mhd:batch-32:vis-0:img:_in_-resnet_640_ctscp-train-resize_1280x640-640_640-640_640-mhd_1241-rfm-rot-flip-batch_72-seq1k-fbb-gdez/ckpt-__var__:_out_-ctscp-r-1280_640-p-640-mhd_1241-rfm-sub-8-fbb

<a id="r_1280_640_p_640_sub_8_mhd_1241_fbb_b54_zeg___train_rf_m_"></a>
## r-1280_640-p-640-sub-8-mhd_1241-fbb-b54-zeg       @ train-rfm-->stitch-ctscp
<a id="on_val___r_1280_640_p_640_sub_8_mhd_1241_fbb_b54_zeg_train_rf_m_"></a>
### on-val       @ r-1280_640-p-640-sub-8-mhd_1241-fbb-b54-zeg/train-rfm-->stitch-ctscp
python stitch.py cfg=ctscp:val:r-1280_640:p-640:sub-8:mhd:batch-32:vis-0:img:_in_-resnet_640_ctscp-train-resize_1280x640-640_640-640_640-mhd_1241-rfm-rot-flip-batch_54-seq1k-fbb-zeg/ckpt-__var__:_out_-ctscp-r-1280_640-p-640-mhd_1241-rfm-sub-8-fbb-b54

