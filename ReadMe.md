This repository provides python and matlab code for all experiments reported in [this paper](https://arxiv.org/abs/1901.04412).
It contains modified versions of several open source repositories that were used for experimentation though not all of these were reported in the paper.
These are the reported models and their corresponding folders:
1. DenseNet: [[densenet]](densenet)   [python/tensorflow]
2. DeepLab: [[old_deeplab (Xception65)]](old_deeplab), [[new_deeplab (Auto DeepLab, ResNet101 PSP)]](new_deeplab)  [python/tensorflow]
3. UNet, SegNet: [[image-segmentation-keras]](image-segmentation-keras) [python/keras]
4. SVM: [[svm]](svm) [matlab]

[unet](unet), [tf_unet](tf_unet) and [image-segmentation-keras/deeplab_keras](image-segmentation-keras/deeplab_keras) contain other implementations of these models that did not work as well as the above. 

Unreported models:

1. FCN: [[image-segmentation-keras]](image-segmentation-keras)
2. Video Segmentation: [[video]](video)

The commands for running each model are provided in a markdown file in the corresponding folder. For example, commands for UNet and DenseNet are in [image-segmentation-keras/unet.md](image-segmentation-keras/unet.md) and [densenet/densenet.md](densenet/densenet.md).
The commands are organized hierarchically into categories of experiments and a table of contents is included for easier navigation.

Following scripts can be used for data preparation and results generation:

1. Data augmentation / sub patch generation: [subPatchDataset.py](subPatchDataset.py), [subPatchBatch.py](subPatchBatch.py)
2. Stitching (and optionally evaluating) sub-patch segmentation results: [stitchSubPatchDataset.py](stitchSubPatchDataset.py)
3. Generating ice concentration plots: [plotIceConcentration.py](plotIceConcentration.py)
4. Visualizing and evaluating segmentation results: [visDataset.py](visDataset.py)

Commands for running these are in the markdown files in the [cmd](cmd) folder as well as in the individual model markdown files.

Some commands might require general utility scripts available in the [python tracking framework](https://github.com/abhineet123/PTF), e.g. [videoToImgSeq.py](https://github.com/abhineet123/PTF/blob/master/videoToImgSeq.py).

If a command does not work,  the command corresponding to some experiment cannot be found or the meaning of some command is not clear, please create an issue and we will do our best to address it.

All the accompanying data is available at [IEEE DataPort](http://dx.doi.org/10.21227/ebax-1h44)

All commands assume that the data is present under _/data/617/_.


The code and data are released under [creative commons attribution license](https://creativecommons.org/licenses/by/4.0/) and are free for research and commercial applications. 
Also, individual repositories used here might have their own licenses that might be more restrictive so please refer to them as well.

If you find this work useful, please consider citing [this paper](https://arxiv.org/abs/1901.04412) [[bibtex](misc/bibtex.txt)].






