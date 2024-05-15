<!-- MarkdownTOC -->

- [16_53](#16_53_)
    - [res-5120       @ 16_53](#res_5120___16_5_3_)
        - [sz-640       @ res-5120/16_53](#sz_640___res_5120_16_53_)
    - [resize-2560       @ 16_53](#resize_2560___16_5_3_)
        - [sz-640       @ resize-2560/16_53](#sz_640___resize_2560_16_5_3_)
        - [sz-320       @ resize-2560/16_53](#sz_320___resize_2560_16_5_3_)
        - [sz-640-aug       @ resize-2560/16_53](#sz_640_aug___resize_2560_16_5_3_)
    - [res-640       @ 16_53](#res_640___16_5_3_)
        - [sz-80       @ res-640/16_53](#sz_80___res_640_16_5_3_)
            - [seq-0       @ sz-80/res-640/16_53](#seq_0___sz_80_res_640_16_5_3_)
            - [seq-1       @ sz-80/res-640/16_53](#seq_1___sz_80_res_640_16_5_3_)
        - [sz-160       @ res-640/16_53](#sz_160___res_640_16_5_3_)
        - [sz-640       @ res-640/16_53](#sz_640___res_640_16_5_3_)
    - [res-320       @ 16_53](#res_320___16_5_3_)
        - [sz-80       @ res-320/16_53](#sz_80___res_320_16_5_3_)
        - [sz-80-aug       @ res-320/16_53](#sz_80_aug___res_320_16_5_3_)
        - [sz-160       @ res-320/16_53](#sz_160___res_320_16_5_3_)
        - [sz-160-aug       @ res-320/16_53](#sz_160_aug___res_320_16_5_3_)
- [54_126](#54_12_6_)
    - [res-2560       @ 54_126](#res_2560___54_126_)
        - [sz-640       @ res-2560/54_126](#sz_640___res_2560_54_12_6_)
    - [res-640       @ 54_126](#res_640___54_126_)
        - [sz-80       @ res-640/54_126](#sz_80___res_640_54_126_)
    - [res-320       @ 54_126](#res_320___54_126_)
        - [sz-80       @ res-320/54_126](#sz_80___res_320_54_126_)
        - [sz-160       @ res-320/54_126](#sz_160___res_320_54_126_)
- [0_126](#0_126_)
    - [resize-640       @ 0_126](#resize_640___0_12_6_)
        - [max_rle-1k       @ resize-640/0_126](#max_rle_1k___resize_640_0_126_)
    - [640       @ 0_126](#640___0_12_6_)
    - [320       @ 0_126](#320___0_12_6_)
    - [160       @ 0_126](#160___0_12_6_)

<!-- /MarkdownTOC -->

<a id="16_53_"></a>
# 16_53
<a id="res_5120___16_5_3_"></a>
## res-5120       @ 16_53-->sub_patch-ipsc
<a id="sz_640___res_5120_16_53_"></a>
### sz-640       @ res-5120/16_53-->sub_patch-ipsc
python subPatchBatch.py cfg=ipsc:sz-640:res-5120:proc-1:vid-1:frame-16_53

<a id="resize_2560___16_5_3_"></a>
## resize-2560       @ 16_53-->sub_patch-ipsc
<a id="sz_640___resize_2560_16_5_3_"></a>
### sz-640       @ resize-2560/16_53-->sub_patch-ipsc
python subPatchBatch.py cfg=ipsc:sz-640:res-2560:proc-1:vid-1:frame-16_53
`seq-0`
python subPatchBatch.py cfg=ipsc:sz-640:res-2560:proc-1:vid-1:frame-16_53:seq-0
<a id="sz_320___resize_2560_16_5_3_"></a>
### sz-320       @ resize-2560/16_53-->sub_patch-ipsc
python subPatchBatch.py cfg=ipsc:sz-320:res-2560:proc-1:vid-1:frame-16_53

<a id="sz_640_aug___resize_2560_16_5_3_"></a>
### sz-640-aug       @ resize-2560/16_53-->sub_patch-ipsc
python subPatchBatch.py cfg=ipsc:sz-640:res-2560:rot-15_345_4:proc-1:vid-1:frame-16_53
`seq-0`
python subPatchBatch.py cfg=ipsc:sz-640:res-2560:rot-15_345_4:proc-1:vid-1:frame-16_53:seq-0:show-0

<a id="res_640___16_5_3_"></a>
## res-640       @ 16_53-->sub_patch-ipsc
<a id="sz_80___res_640_16_5_3_"></a>
### sz-80       @ res-640/16_53-->sub_patch-ipsc
python subPatchBatch.py cfg=ipsc:sz-80:res-640:proc-1:vid-1:frame-16_53
<a id="seq_0___sz_80_res_640_16_5_3_"></a>
#### seq-0       @ sz-80/res-640/16_53-->sub_patch-ipsc
python subPatchBatch.py cfg=ipsc:sz-80:res-640:proc-1:vid-0:frame-16_53:seq-0
<a id="seq_1___sz_80_res_640_16_5_3_"></a>
#### seq-1       @ sz-80/res-640/16_53-->sub_patch-ipsc
python subPatchBatch.py cfg=ipsc:sz-80:res-640:proc-1:vid-0:frame-16_53:seq-1
<a id="sz_160___res_640_16_5_3_"></a>
### sz-160       @ res-640/16_53-->sub_patch-ipsc
python subPatchBatch.py cfg=ipsc:sz-160:res-640:proc-1:vid-1:frame-16_53
<a id="sz_640___res_640_16_5_3_"></a>
### sz-640       @ res-640/16_53-->sub_patch-ipsc
python subPatchBatch.py cfg=ipsc:sz-640:res-640:proc-1:vid-1:frame-16_53

<a id="res_320___16_5_3_"></a>
## res-320       @ 16_53-->sub_patch-ipsc
<a id="sz_80___res_320_16_5_3_"></a>
### sz-80       @ res-320/16_53-->sub_patch-ipsc
python subPatchBatch.py cfg=ipsc:sz-80:res-320:proc-1:vid-1:frame-16_53
`seq-0`
python subPatchBatch.py cfg=ipsc:sz-80:res-320:proc-1:vid-0:frame-16_53:seq-0

<a id="sz_80_aug___res_320_16_5_3_"></a>
### sz-80-aug       @ res-320/16_53-->sub_patch-ipsc
python subPatchBatch.py cfg=ipsc:sz-80:res-320:strd-40_80:rot-15_345_4:flip-1:proc-1:vid-1:frame-16_53
`seq-0`
python subPatchBatch.py cfg=ipsc:sz-80:res-320:strd-40_80:rot-15_345_4:flip-1:proc-1:vid-1:frame-16_53:seq-0

<a id="sz_160___res_320_16_5_3_"></a>
### sz-160       @ res-320/16_53-->sub_patch-ipsc
python subPatchBatch.py cfg=ipsc:sz-160:res-320:proc-1:vid-1:frame-16_53
`seq-0`
python subPatchBatch.py cfg=ipsc:sz-160:res-320:proc-1:vid-0:frame-16_53:seq-0

<a id="sz_160_aug___res_320_16_5_3_"></a>
### sz-160-aug       @ res-320/16_53-->sub_patch-ipsc
python subPatchBatch.py cfg=ipsc:sz-160:res-320:strd-40_160:rot-15_345_4:flip-1:proc-1:vid-1:frame-16_53

<a id="54_12_6_"></a>
# 54_126
<a id="res_2560___54_126_"></a>
## res-2560       @ 54_126-->sub_patch-ipsc
<a id="sz_640___res_2560_54_12_6_"></a>
### sz-640       @ res-2560/54_126-->sub_patch-ipsc
python subPatchBatch.py cfg=ipsc:sz-640:res-2560:proc-1:vid-1:frame-54_126

<a id="res_640___54_126_"></a>
## res-640       @ 54_126-->sub_patch-ipsc
<a id="sz_80___res_640_54_126_"></a>
### sz-80       @ res-640/54_126-->sub_patch-ipsc
python subPatchBatch.py cfg=ipsc:sz-80:res-640:proc-1:vid-1:frame-54_126
<a id="res_320___54_126_"></a>
## res-320       @ 54_126-->sub_patch-ipsc
<a id="sz_80___res_320_54_126_"></a>
### sz-80       @ res-320/54_126-->sub_patch-ipsc
python subPatchBatch.py cfg=ipsc:sz-80:res-320:proc-1:vid-1:frame-54_126

<a id="sz_160___res_320_54_126_"></a>
### sz-160       @ res-320/54_126-->sub_patch-ipsc
python subPatchBatch.py cfg=ipsc:sz-160:res-320:proc-1:vid-1:frame-54_126

<a id="0_126_"></a>
# 0_126
<a id="resize_640___0_12_6_"></a>
## resize-640       @ 0_126-->sub_patch-ipsc
python subPatchBatch.py cfg=ipsc:sz-640:log:seq:res-640:proc-1:vid-1

python subPatchBatch.py cfg=ipsc:sz-320:log:seq:res-640:proc-1:vid-1

python subPatchBatch.py cfg=ipsc:sz-160:log:seq:res-640:proc-1:vid-1

python subPatchBatch.py cfg=ipsc:sz-80:log:seq:res-640:proc-1:vid-1

<a id="max_rle_1k___resize_640_0_126_"></a>
### max_rle-1k       @ resize-640/0_126-->sub_patch-ipsc
python subPatchBatch.py cfg=ipsc:sz-640:log:seq:res-640:proc-1:vid-0:max_rle-1000

python subPatchBatch.py cfg=ipsc:sz-320:log:seq:res-640:proc-1:vid-0:max_rle-1000

python subPatchBatch.py cfg=ipsc:sz-160:log:seq:res-640:proc-1:vid-0:max_rle-1000

python subPatchBatch.py cfg=ipsc:sz-80:log:seq:res-640:proc-1:vid-0:max_rle-1000

<a id="640___0_12_6_"></a>
## 640       @ 0_126-->sub_patch-ipsc
python subPatchBatch.py cfg=ipsc:sz-640:log:seq:proc-1

python subPatchBatch.py cfg=ipsc:sz-640:smin-0:smax-0:rmin-15:rmax-345:rnum-0:log:seq-0

<a id="320___0_12_6_"></a>
## 320       @ 0_126-->sub_patch-ipsc
python subPatchBatch.py cfg=ipsc:sz-320:log:seq
<a id="160___0_12_6_"></a>
## 160       @ 0_126-->sub_patch-ipsc
python subPatchBatch.py cfg=ipsc:sz-160:log:seq
