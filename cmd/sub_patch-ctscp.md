<!-- MarkdownTOC -->

- [val](#val_)
    - [orig       @ val](#orig___va_l_)
        - [p-1024       @ orig/val](#p_1024___orig_val_)
    - [r-640       @ val](#r_640___va_l_)
    - [r-80       @ val](#r_80___va_l_)
    - [r-128       @ val](#r_128___va_l_)
    - [r-160       @ val](#r_160___va_l_)
    - [r-256       @ val](#r_256___va_l_)
    - [r-160_80       @ val](#r_160_80___va_l_)
        - [p-80       @ r-160_80/val](#p_80___r_160_80_val_)
    - [r-256_128       @ val](#r_256_128___va_l_)
    - [r-512_256       @ val](#r_512_256___va_l_)
    - [r-1280_640       @ val](#r_1280_640___va_l_)
        - [p-640       @ r-1280_640/val](#p_640___r_1280_640_val_)
- [train](#train_)
    - [orig       @ train](#orig___trai_n_)
        - [p-1024       @ orig/train](#p_1024___orig_train_)
        - [p-640       @ orig/train](#p_640___orig_train_)
    - [r-1024       @ train](#r_1024___trai_n_)
    - [r-640       @ train](#r_640___trai_n_)
        - [vid       @ r-640/train](#vid___r_640_trai_n_)
        - [img       @ r-640/train](#img___r_640_trai_n_)
    - [r-640_1280       @ train](#r_640_1280___trai_n_)
        - [p-640       @ r-640_1280/train](#p_640___r_640_1280_train_)

<!-- /MarkdownTOC -->

<a id="val_"></a>
# val
<a id="orig___va_l_"></a>
## orig       @ val-->sub_patch-ctscp
python sub_patch_multi.py cfg=ctscp:val:vid-0:img-1 save_palette=0 save_mapped=1

<a id="p_1024___orig_val_"></a>
### p-1024       @ orig/val-->sub_patch-ctscp
python sub_patch_multi.py cfg=ctscp:val:p-1024:vid-0:img-1

<a id="r_640___va_l_"></a>
## r-640       @ val-->sub_patch-ctscp
python sub_patch_multi.py cfg=ctscp:val:r-640:vid-0:img-1

<a id="r_80___va_l_"></a>
## r-80       @ val-->sub_patch-ctscp
python sub_patch_multi.py cfg=ctscp:val:r-80:vid-0:img-1 save_palette=0 save_mapped=1
<a id="r_128___va_l_"></a>
## r-128       @ val-->sub_patch-ctscp
python sub_patch_multi.py cfg=ctscp:val:r-128:vid-0:img-1 save_palette=0 save_mapped=1
<a id="r_160___va_l_"></a>
## r-160       @ val-->sub_patch-ctscp
python sub_patch_multi.py cfg=ctscp:val:r-160:vid-0:img-1 save_palette=0 save_mapped=1
<a id="r_256___va_l_"></a>
## r-256       @ val-->sub_patch-ctscp
python sub_patch_multi.py cfg=ctscp:val:r-256:vid-0:img-1 save_palette=0 save_mapped=1

<a id="r_160_80___va_l_"></a>
## r-160_80       @ val-->sub_patch-ctscp
python sub_patch_multi.py cfg=ctscp:val:r-160_80:vid-0:img-1 save_palette=0 save_mapped=1
<a id="p_80___r_160_80_val_"></a>
### p-80       @ r-160_80/val-->sub_patch-ctscp
python sub_patch_multi.py cfg=ctscp:val:r-160_80:p-80:vid-0:img-1 save_palette=0 save_mapped=1
<a id="r_256_128___va_l_"></a>
## r-256_128       @ val-->sub_patch-ctscp
python sub_patch_multi.py cfg=ctscp:val:r-256_128:vid-0:img-1 save_palette=0 save_mapped=1
<a id="r_512_256___va_l_"></a>
## r-512_256       @ val-->sub_patch-ctscp
python sub_patch_multi.py cfg=ctscp:val:r-512_256:vid-0:img-1 save_palette=0 save_mapped=1

<a id="r_1280_640___va_l_"></a>
## r-1280_640       @ val-->sub_patch-ctscp
python sub_patch_multi.py cfg=ctscp:val:r-1280_640:vid-0:img-1
<a id="p_640___r_1280_640_val_"></a>
### p-640       @ r-1280_640/val-->sub_patch-ctscp
python sub_patch_multi.py cfg=ctscp:val:r-1280_640:p-640:vid-0:img-1



<a id="train_"></a>
# train
<a id="orig___trai_n_"></a>
## orig       @ train-->sub_patch-ctscp
<a id="p_1024___orig_train_"></a>
### p-1024       @ orig/train-->sub_patch-ctscp
python sub_patch_multi.py cfg=ctscp:train:p-1024:vid-0:img-1
`seq-0`
python sub_patch_multi.py cfg=ctscp:train:p-1024:vid-0:img-1:seq-0
<a id="p_640___orig_train_"></a>
### p-640       @ orig/train-->sub_patch-ctscp
python sub_patch_multi.py cfg=ctscp:train:p-640:vid-0:img-1

<a id="r_1024___trai_n_"></a>
## r-1024       @ train-->sub_patch-ctscp
python sub_patch_multi.py cfg=ctscp:train:r-1024:vid-1:img-0
python sub_patch_multi.py cfg=ctscp:train:r-1024:vid-0:img-1
python sub_patch_multi.py cfg=ctscp:val:r-1024:vid-0:img-1

<a id="r_640___trai_n_"></a>
## r-640       @ train-->sub_patch-ctscp
<a id="vid___r_640_trai_n_"></a>
### vid       @ r-640/train-->sub_patch-ctscp
`compression artifacts showing up in labels saved as video files when running subpatch in wsl but not when running it in windows; tf_seg run on wsl in both cases; apparently crf=0 does not equate lossless in wsl ffmpeg`
python sub_patch_multi.py cfg=ctscp:train:r-640:vid-1:img-0
`dbg`
python sub_patch_multi.py cfg=ctscp:train:r-640:vid-1:img-0:seq-0
<a id="img___r_640_trai_n_"></a>
### img       @ r-640/train-->sub_patch-ctscp
python sub_patch_multi.py cfg=ctscp:train:r-640:vid-0:img-1

<a id="r_640_1280___trai_n_"></a>
## r-640_1280       @ train-->sub_patch-ctscp
python sub_patch_multi.py cfg=ctscp:train:r-1280_640:vid-0:img-1
<a id="p_640___r_640_1280_train_"></a>
### p-640       @ r-640_1280/train-->sub_patch-ctscp
python sub_patch_multi.py cfg=ctscp:train:r-1280_640:p-640:vid-0:img-1




