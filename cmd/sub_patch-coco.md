<!-- MarkdownTOC -->

- [train](#train_)
    - [r-640       @ train](#r_640___trai_n_)
        - [p-640       @ r-640/train](#p_640___r_640_trai_n_)
            - [end-2000       @ p-640/r-640/train](#end_2000___p_640_r_640_trai_n_)
        - [p-0-aug       @ r-640/train](#p_0_aug___r_640_trai_n_)
            - [end-100       @ p-0-aug/r-640/train](#end_100___p_0_aug_r_640_trai_n_)
    - [r-1280       @ train](#r_1280___trai_n_)
        - [p-640       @ r-1280/train](#p_640___r_1280_train_)
            - [end-100       @ p-640/r-1280/train](#end_100___p_640_r_1280_train_)
            - [end-2000       @ p-640/r-1280/train](#end_2000___p_640_r_1280_train_)
            - [end-600       @ p-640/r-1280/train](#end_600___p_640_r_1280_train_)
        - [p-640-aug       @ r-1280/train](#p_640_aug___r_1280_train_)
- [val](#val_)
    - [r-640       @ val](#r_640___va_l_)
        - [p-640       @ r-640/val](#p_640___r_640_va_l_)
            - [end-1000       @ p-640/r-640/val](#end_1000___p_640_r_640_va_l_)
    - [r-1280       @ val](#r_1280___va_l_)
        - [p-640       @ r-1280/val](#p_640___r_1280_val_)
            - [end-100       @ p-640/r-1280/val](#end_100___p_640_r_1280_val_)
            - [end-2000       @ p-640/r-1280/val](#end_2000___p_640_r_1280_val_)
            - [end-500       @ p-640/r-1280/val](#end_500___p_640_r_1280_val_)

<!-- /MarkdownTOC -->

<a id="train_"></a>
# train
<a id="r_640___trai_n_"></a>
## r-640       @ train-->sub_patch-coco
<a id="p_640___r_640_trai_n_"></a>
### p-640       @ r-640/train-->sub_patch-coco
python sub_patch_multi.py cfg=coco:p-640:r-640:vid-0:img
<a id="end_2000___p_640_r_640_trai_n_"></a>
#### end-2000       @ p-640/r-640/train-->sub_patch-coco
python sub_patch_multi.py cfg=coco:p-640:r-640:vid-0:img:end-2000

<a id="p_0_aug___r_640_trai_n_"></a>
### p-0-aug       @ r-640/train-->sub_patch-coco
python sub_patch_multi.py cfg=coco:p-0:r-640:rot-15_345_4:vid-0:img
<a id="end_100___p_0_aug_r_640_trai_n_"></a>
#### end-100       @ p-0-aug/r-640/train-->sub_patch-coco
python sub_patch_multi.py cfg=coco:r-640:p-0:rot-15_345_4:vid-0:img:end-100:bkg-50
python sub_patch_multi.py cfg=coco:r-640:p-640:rot-15_345_4:vid-0:img:end-100:bkg-50

<a id="r_1280___trai_n_"></a>
## r-1280       @ train-->sub_patch-coco
<a id="p_640___r_1280_train_"></a>
### p-640       @ r-1280/train-->sub_patch-coco
python sub_patch_multi.py cfg=coco:p-640:r-1280:vid-0:img
<a id="end_100___p_640_r_1280_train_"></a>
#### end-100       @ p-640/r-1280/train-->sub_patch-coco
python sub_patch_multi.py cfg=coco:p-640:r-1280:vid-0:img:end-100
<a id="end_2000___p_640_r_1280_train_"></a>
#### end-2000       @ p-640/r-1280/train-->sub_patch-coco
python sub_patch_multi.py cfg=coco:p-640:r-1280:vid-0:img:end-2000
<a id="end_600___p_640_r_1280_train_"></a>
#### end-600       @ p-640/r-1280/train-->sub_patch-coco
python sub_patch_multi.py cfg=coco:p-640:r-1280:vid-0:img:end-600

<a id="p_640_aug___r_1280_train_"></a>
### p-640-aug       @ r-1280/train-->sub_patch-coco
python sub_patch_multi.py cfg=coco:p-640:r-1280:rot-15_345_4:vid-0:img:end-100

<a id="val_"></a>
# val
<a id="r_640___va_l_"></a>
## r-640       @ val-->sub_patch-coco
<a id="p_640___r_640_va_l_"></a>
### p-640       @ r-640/val-->sub_patch-coco
python sub_patch_multi.py cfg=coco:val:p-640:r-640:vid-0:img
`dbg`
python sub_patch_multi.py cfg=coco:val:p-640:r-640:vid-0:img:end-100
<a id="end_1000___p_640_r_640_va_l_"></a>
#### end-1000       @ p-640/r-640/val-->sub_patch-coco
python sub_patch_multi.py cfg=coco:p-640:r-640:vid-0:img:end-1000

<a id="r_1280___va_l_"></a>
## r-1280       @ val-->sub_patch-coco
<a id="p_640___r_1280_val_"></a>
### p-640       @ r-1280/val-->sub_patch-coco
python sub_patch_multi.py cfg=coco:val:p-640:r-1280:vid-0:img
<a id="end_100___p_640_r_1280_val_"></a>
#### end-100       @ p-640/r-1280/val-->sub_patch-coco
python sub_patch_multi.py cfg=coco:val:p-640:r-1280:vid-0:img:end-100:vis
<a id="end_2000___p_640_r_1280_val_"></a>
#### end-2000       @ p-640/r-1280/val-->sub_patch-coco
python sub_patch_multi.py cfg=coco:val:p-640:r-1280:vid-0:img:end-2000
<a id="end_500___p_640_r_1280_val_"></a>
#### end-500       @ p-640/r-1280/val-->sub_patch-coco
python sub_patch_multi.py cfg=coco:val:p-640:r-1280:vid-0:img:end-500

