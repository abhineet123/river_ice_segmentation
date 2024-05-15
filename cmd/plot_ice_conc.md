<!-- MarkdownTOC -->

- [training_32_49](#training_32_4_9_)
   - [deeplab       @ training_32_49](#deeplab___training_32_49_)
      - [anchor       @ deeplab/training_32_49](#anchor___deeplab_training_32_49_)
      - [frazil       @ deeplab/training_32_49](#frazil___deeplab_training_32_49_)
   - [auto_deeplab       @ training_32_49](#auto_deeplab___training_32_49_)
      - [anchor       @ auto_deeplab/training_32_49](#anchor___auto_deeplab_training_32_4_9_)
      - [frazil       @ auto_deeplab/training_32_49](#frazil___auto_deeplab_training_32_4_9_)
   - [resnet101_psp       @ training_32_49](#resnet101_psp___training_32_49_)
      - [anchor       @ resnet101_psp/training_32_49](#anchor___resnet101_psp_training_32_49_)
      - [frazil       @ resnet101_psp/training_32_49](#frazil___resnet101_psp_training_32_49_)
   - [segnet       @ training_32_49](#segnet___training_32_49_)
      - [max_acc       @ segnet/training_32_49](#max_acc___segnet_training_32_4_9_)
   - [unet       @ training_32_49](#unet___training_32_49_)
   - [densenet       @ training_32_49](#densenet___training_32_49_)
      - [anchor       @ densenet/training_32_49](#anchor___densenet_training_32_4_9_)
      - [frazil       @ densenet/training_32_49](#frazil___densenet_training_32_4_9_)
   - [svm       @ training_32_49](#svm___training_32_49_)
      - [anchor       @ svm/training_32_49](#anchor___svm_training_32_49_)
      - [frazil       @ svm/training_32_49](#frazil___svm_training_32_49_)
   - [svm_deeplab       @ training_32_49](#svm_deeplab___training_32_49_)
      - [no_labels       @ svm_deeplab/training_32_49](#no_labels___svm_deeplab_training_32_49_)
   - [svm_deeplab_densenet       @ training_32_49](#svm_deeplab_densenet___training_32_49_)
   - [svm_deeplab_unet_densenet_segnet       @ training_32_49](#svm_deeplab_unet_densenet_segnet___training_32_49_)
      - [Combined       @ svm_deeplab_unet_densenet_segnet/training_32_49](#combined___svm_deeplab_unet_densenet_segnet_training_32_4_9_)
      - [anchor       @ svm_deeplab_unet_densenet_segnet/training_32_49](#anchor___svm_deeplab_unet_densenet_segnet_training_32_4_9_)
      - [frazil       @ svm_deeplab_unet_densenet_segnet/training_32_49](#frazil___svm_deeplab_unet_densenet_segnet_training_32_4_9_)
- [training_4_49](#training_4_49_)
   - [svm_deeplab_unet_densenet_segnet       @ training_4_49](#svm_deeplab_unet_densenet_segnet___training_4_4_9_)
      - [Combined       @ svm_deeplab_unet_densenet_segnet/training_4_49](#combined___svm_deeplab_unet_densenet_segnet_training_4_49_)
- [video](#video_)
   - [YUN00001_3600       @ video](#yun00001_3600___vide_o_)
      - [combined       @ YUN00001_3600/video](#combined___yun00001_3600_vide_o_)
         - [svm       @ combined/YUN00001_3600/video](#svm___combined_yun00001_3600_video_)
      - [frazil       @ YUN00001_3600/video](#frazil___yun00001_3600_vide_o_)
         - [svm       @ frazil/YUN00001_3600/video](#svm___frazil_yun00001_3600_video_)
      - [anchor       @ YUN00001_3600/video](#anchor___yun00001_3600_vide_o_)
         - [svm       @ anchor/YUN00001_3600/video](#svm___anchor_yun00001_3600_video_)
   - [20160122_YUN00002_700_2500       @ video](#20160122_yun00002_700_2500___vide_o_)
      - [combined       @ 20160122_YUN00002_700_2500/video](#combined___20160122_yun00002_700_2500_video_)
         - [plot_changed_seg_count       @ combined/20160122_YUN00002_700_2500/video](#plot_changed_seg_count___combined_20160122_yun00002_700_2500_vide_o_)
      - [frazil       @ 20160122_YUN00002_700_2500/video](#frazil___20160122_yun00002_700_2500_video_)
      - [anchor       @ 20160122_YUN00002_700_2500/video](#anchor___20160122_yun00002_700_2500_video_)
   - [20160122_YUN00020_2000_3800       @ video](#20160122_yun00020_2000_3800___vide_o_)
      - [combined       @ 20160122_YUN00020_2000_3800/video](#combined___20160122_yun00020_2000_3800_vide_o_)
         - [svm       @ combined/20160122_YUN00020_2000_3800/video](#svm___combined_20160122_yun00020_2000_3800_video_)
      - [frazil       @ 20160122_YUN00020_2000_3800/video](#frazil___20160122_yun00020_2000_3800_vide_o_)
         - [svm       @ frazil/20160122_YUN00020_2000_3800/video](#svm___frazil_20160122_yun00020_2000_3800_video_)
      - [anchor       @ 20160122_YUN00020_2000_3800/video](#anchor___20160122_yun00020_2000_3800_vide_o_)
         - [svm       @ anchor/20160122_YUN00020_2000_3800/video](#svm___anchor_20160122_yun00020_2000_3800_video_)
   - [20161203_Deployment_1_YUN00001_900_2700       @ video](#20161203_deployment_1_yun00001_900_2700___vide_o_)
      - [combined       @ 20161203_Deployment_1_YUN00001_900_2700/video](#combined___20161203_deployment_1_yun00001_900_2700_vide_o_)
         - [svm       @ combined/20161203_Deployment_1_YUN00001_900_2700/video](#svm___combined_20161203_deployment_1_yun00001_900_2700_video_)
            - [20161203_Deployment_1_YUN00001_900_1200       @ svm/combined/20161203_Deployment_1_YUN00001_900_2700/video](#20161203_deployment_1_yun00001_900_1200___svm_combined_20161203_deployment_1_yun00001_900_2700_video_)
      - [frazil       @ 20161203_Deployment_1_YUN00001_900_2700/video](#frazil___20161203_deployment_1_yun00001_900_2700_vide_o_)
         - [svm       @ frazil/20161203_Deployment_1_YUN00001_900_2700/video](#svm___frazil_20161203_deployment_1_yun00001_900_2700_video_)
            - [20161203_Deployment_1_YUN00001_900_1200       @ svm/frazil/20161203_Deployment_1_YUN00001_900_2700/video](#20161203_deployment_1_yun00001_900_1200___svm_frazil_20161203_deployment_1_yun00001_900_2700_video_)
      - [anchor       @ 20161203_Deployment_1_YUN00001_900_2700/video](#anchor___20161203_deployment_1_yun00001_900_2700_vide_o_)
         - [svm       @ anchor/20161203_Deployment_1_YUN00001_900_2700/video](#svm___anchor_20161203_deployment_1_yun00001_900_2700_video_)
            - [20161203_Deployment_1_YUN00001_2000_2300       @ svm/anchor/20161203_Deployment_1_YUN00001_900_2700/video](#20161203_deployment_1_yun00001_2000_2300___svm_anchor_20161203_deployment_1_yun00001_900_2700_video_)
            - [20161203_Deployment_1_YUN00001_900_1200       @ svm/anchor/20161203_Deployment_1_YUN00001_900_2700/video](#20161203_deployment_1_yun00001_900_1200___svm_anchor_20161203_deployment_1_yun00001_900_2700_video_)
   - [20161203_Deployment_1_YUN00002_1800       @ video](#20161203_deployment_1_yun00002_1800___vide_o_)
      - [combined       @ 20161203_Deployment_1_YUN00002_1800/video](#combined___20161203_deployment_1_yun00002_1800_vide_o_)
      - [frazil       @ 20161203_Deployment_1_YUN00002_1800/video](#frazil___20161203_deployment_1_yun00002_1800_vide_o_)
      - [anchor       @ 20161203_Deployment_1_YUN00002_1800/video](#anchor___20161203_deployment_1_yun00002_1800_vide_o_)

<!-- /MarkdownTOC -->

<a id="training_32_4_9_"></a>
# training_32_49
python3 plotIceConcentration.py --images_path=/data/617/images/training/images --labels_path=/data/617/images/training/labels --images_ext=tif --labels_ext=tif --n_classes=3
<a id="deeplab___training_32_49_"></a>
## deeplab       @ training_32_49-->plot_ice_conc
python3 plotIceConcentration.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --seg_paths=deeplab\log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_training_32_49_raw_z370_190408_200424 --images_ext=png --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=deeplab

<a id="anchor___deeplab_training_32_49_"></a>
### anchor       @ deeplab/training_32_49-->plot_ice_conc

python3 plotIceConcentration.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --seg_paths=deeplab\log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_training_32_49_raw_z370_190408_200424 --images_ext=png --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=deeplab --ice_type=1

<a id="frazil___deeplab_training_32_49_"></a>
### frazil       @ deeplab/training_32_49-->plot_ice_conc

python3 plotIceConcentration.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --seg_paths=deeplab\log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_training_32_49_raw_z370_190408_200424 --images_ext=png --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=deeplab --ice_type=2

<a id="auto_deeplab___training_32_49_"></a>
## auto_deeplab       @ training_32_49-->plot_ice_conc

python3 plotIceConcentration.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --seg_paths=new_deeplab\nas_hnasnet_0_31_ade20k_training_32_49_raw__orca_190909_083554 --images_ext=png --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=auto_deeplab

<a id="anchor___auto_deeplab_training_32_4_9_"></a>
### anchor       @ auto_deeplab/training_32_49-->plot_ice_conc

python3 plotIceConcentration.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --seg_paths=new_deeplab\nas_hnasnet_0_31_ade20k_training_32_49_raw__orca_190909_083554 --images_ext=png --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=auto_deeplab --ice_type=1

<a id="frazil___auto_deeplab_training_32_4_9_"></a>
### frazil       @ auto_deeplab/training_32_49-->plot_ice_conc

python3 plotIceConcentration.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --seg_paths=new_deeplab\nas_hnasnet_0_31_ade20k_training_32_49_raw__orca_190909_083554 --images_ext=png --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=auto_deeplab --ice_type=2

<a id="resnet101_psp___training_32_49_"></a>
## resnet101_psp       @ training_32_49-->plot_ice_conc

python3 plotIceConcentration.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --seg_paths=new_deeplab\resnet_v1_101_0_31_training_32_49_raw_grs_190909_101731 --images_ext=png --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=resnet101_psp

<a id="anchor___resnet101_psp_training_32_49_"></a>
### anchor       @ resnet101_psp/training_32_49-->plot_ice_conc

python3 plotIceConcentration.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --seg_paths=new_deeplab\resnet_v1_101_0_31_training_32_49_raw_grs_190909_101731 --images_ext=png --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=resnet101_psp --ice_type=1

<a id="frazil___resnet101_psp_training_32_49_"></a>
### frazil       @ resnet101_psp/training_32_49-->plot_ice_conc

python3 plotIceConcentration.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --seg_paths=new_deeplab\resnet_v1_101_0_31_training_32_49_raw_grs_190909_101731 --images_ext=png --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=resnet101_psp --ice_type=2

<a id="segnet___training_32_49_"></a>
## segnet       @ training_32_49-->plot_ice_conc

<a id="max_acc___segnet_training_32_4_9_"></a>
### max_acc       @ segnet/training_32_49-->plot_ice_conc

python3 plotIceConcentration.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_root_dir=H:\UofA\617\Project\617_proj_code\log\segnet --seg_paths=log_vgg_segnet_0_31_640_640_64_256_rot_15_345_4_flip_training_32_49_max_acc_raw_grs_190524_145647,log_vgg_segnet_0_31_640_640_64_256_rot_15_345_4_flip_training_32_49_max_val_acc_raw_grs_190524_154518,log_vgg_segnet_0_31_640_640_64_256_rot_15_345_4_flip_training_32_49_min_loss_raw_grs_190524_154535 --images_ext=png --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=max_acc,max_val_acc,min_loss


<a id="unet___training_32_49_"></a>
## unet       @ training_32_49-->plot_ice_conc

python3 plotIceConcentration.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --seg_paths=unet\log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_training_32_49_max_val_acc_grs_190413_102443\raw --images_ext=png --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=unet

<a id="densenet___training_32_49_"></a>
## densenet       @ training_32_49-->plot_ice_conc

python3 plotIceConcentration.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --seg_paths=densenet\log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_training_32_49_raw_z370_190413_084638 --images_ext=png --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=densenet

<a id="anchor___densenet_training_32_4_9_"></a>
### anchor       @ densenet/training_32_49-->plot_ice_conc

python3 plotIceConcentration.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --seg_paths=densenet\log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_training_32_49_raw_z370_190413_084638 --images_ext=png --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=densenet --ice_type=1

<a id="frazil___densenet_training_32_4_9_"></a>
### frazil       @ densenet/training_32_49-->plot_ice_conc

python3 plotIceConcentration.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --seg_paths=densenet\log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_training_32_49_raw_z370_190413_084638 --images_ext=png --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=densenet --ice_type=2


<a id="svm___training_32_49_"></a>
## svm       @ training_32_49-->plot_ice_conc

python3 plotIceConcentration.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --seg_paths=H:\UofA\617\Project\617_proj_code\svm\log\svm_1_32_1 --images_ext=png --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm_1

python3 plotIceConcentration.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --seg_paths=H:\UofA\617\Project\617_proj_code\svm\log\svm_1_32_2 --images_ext=png --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm_2


<a id="anchor___svm_training_32_49_"></a>
### anchor       @ svm/training_32_49-->plot_ice_conc

python3 plotIceConcentration.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --seg_paths=H:\UofA\617\Project\617_proj_code\svm\log\svm_1_32_2 --images_ext=png --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm_2 --ice_type=1

<a id="frazil___svm_training_32_49_"></a>
### frazil       @ svm/training_32_49-->plot_ice_conc

python3 plotIceConcentration.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --seg_paths=H:\UofA\617\Project\617_proj_code\svm\log\svm_1_32_2 --images_ext=png --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm_2 --ice_type=2

<a id="svm_deeplab___training_32_49_"></a>
## svm_deeplab       @ training_32_49-->plot_ice_conc

python3 plotIceConcentration.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --seg_paths=svm\svm_1_32_2,deeplab\log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_training_32_49_raw_z370_190408_200424 --images_ext=png --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,deeplab --ice_type=1

<a id="no_labels___svm_deeplab_training_32_49_"></a>
### no_labels       @ svm_deeplab/training_32_49-->plot_ice_conc

python3 plotIceConcentration.py --images_path=/data/617/images/training_32_49/images --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --seg_paths=svm\svm_1_32_2,deeplab\log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_training_32_49_raw_z370_190408_200424 --images_ext=png --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,deeplab --ice_type=1

<a id="svm_deeplab_densenet___training_32_49_"></a>
## svm_deeplab_densenet       @ training_32_49-->plot_ice_conc

python3 plotIceConcentration.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --seg_paths=svm\svm_1_32_2,deeplab\log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_training_32_49_raw_z370_190408_200424,densenet\log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_training_32_49_raw_z370_190413_084638 --images_ext=png --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,deeplab,densenet --ice_type=0

<a id="svm_deeplab_unet_densenet_segnet___training_32_49_"></a>
## svm_deeplab_unet_densenet_segnet       @ training_32_49-->plot_ice_conc

<a id="combined___svm_deeplab_unet_densenet_segnet_training_32_4_9_"></a>
### Combined       @ svm_deeplab_unet_densenet_segnet/training_32_49-->plot_ice_conc

python3 plotIceConcentration.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --seg_paths=svm\svm_1_32_2,deeplab\log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_training_32_49_raw_z370_190408_200424,unet\log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_training_32_49_max_val_acc_grs_190413_102443\raw,densenet\log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_training_32_49_raw_z370_190413_084638,segnet/log_vgg_segnet_0_31_640_640_64_256_rot_15_345_4_flip_training_32_49_max_acc_raw_grs_190524_145647 --images_ext=png --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,deeplab,unet,densenet,segnet --ice_type=0

<a id="anchor___svm_deeplab_unet_densenet_segnet_training_32_4_9_"></a>
### anchor       @ svm_deeplab_unet_densenet_segnet/training_32_49-->plot_ice_conc

python3 plotIceConcentration.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --seg_paths=svm\svm_1_32_2,deeplab\log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_training_32_49_raw_z370_190408_200424,unet\log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_training_32_49_max_val_acc_grs_190413_102443\raw,densenet\log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_training_32_49_raw_z370_190413_084638,segnet/log_vgg_segnet_0_31_640_640_64_256_rot_15_345_4_flip_training_32_49_max_acc_raw_grs_190524_145647 --images_ext=png --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,deeplab,unet,densenet,segnet --ice_type=1

<a id="frazil___svm_deeplab_unet_densenet_segnet_training_32_4_9_"></a>
### frazil       @ svm_deeplab_unet_densenet_segnet/training_32_49-->plot_ice_conc

python3 plotIceConcentration.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --seg_paths=svm\svm_1_32_2,deeplab\log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_training_32_49_raw_z370_190408_200424,unet\log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_training_32_49_max_val_acc_grs_190413_102443\raw,densenet\log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_training_32_49_raw_z370_190413_084638,segnet/log_vgg_segnet_0_31_640_640_64_256_rot_15_345_4_flip_training_32_49_max_acc_raw_grs_190524_145647 --images_ext=png --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,deeplab,unet,densenet,segnet --ice_type=2

<a id="training_4_49_"></a>
# training_4_49
<a id="svm_deeplab_unet_densenet_segnet___training_4_4_9_"></a>
## svm_deeplab_unet_densenet_segnet       @ training_4_49-->plot_ice_conc

<a id="combined___svm_deeplab_unet_densenet_segnet_training_4_49_"></a>
### Combined       @ svm_deeplab_unet_densenet_segnet/training_4_49-->plot_ice_conc

python3 plotIceConcentration.py --images_path=/data/617/images/training_4_49/images --labels_path=/data/617/images/training_4_49/labels --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --seg_paths=svm\svm_1_4_2,deeplab\log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_3_training_4_49_raw_grs_190524_173504,unet\log_vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_training_4_49_max_val_acc_raw_grs_190524_172038,densenet\log_rt2_training_0_3_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_320_4_elu_predict_acc_training_4_49_raw_z370_190524_173744,segnet/log_vgg_segnet_0_3_640_640_64_256_rot_15_345_4_flip_training_4_49_max_acc_raw_grs_190524_173006 --images_ext=png --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,deeplab,unet,densenet,segnet --ice_type=0

python3 plotIceConcentration.py --images_path=/data/617/images/training_4_49/images --labels_path=/data/617/images/training_4_49/labels --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --seg_paths=svm\svm_1_4_2,deeplab\log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_3_training_4_49_raw_grs_190524_173504,unet\log_vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_training_4_49_max_val_acc_raw_grs_190524_172038,densenet\log_rt2_training_0_3_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_320_4_elu_predict_acc_training_4_49_raw_z370_190524_173744,segnet/log_vgg_segnet_0_3_640_640_64_256_rot_15_345_4_flip_training_4_49_max_acc_raw_grs_190524_173006 --images_ext=png --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,deeplab,unet,densenet,segnet --ice_type=1

python3 plotIceConcentration.py --images_path=/data/617/images/training_4_49/images --labels_path=/data/617/images/training_4_49/labels --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --seg_paths=svm\svm_1_4_2,deeplab\log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_3_training_4_49_raw_grs_190524_173504,unet\log_vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_training_4_49_max_val_acc_raw_grs_190524_172038,densenet\log_rt2_training_0_3_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_320_4_elu_predict_acc_training_4_49_raw_z370_190524_173744,segnet/log_vgg_segnet_0_3_640_640_64_256_rot_15_345_4_flip_training_4_49_max_acc_raw_grs_190524_173006 --images_ext=png --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,deeplab,unet,densenet,segnet --ice_type=2



<a id="video_"></a>
# video
<a id="yun00001_3600___vide_o_"></a>
## YUN00001_3600       @ video-->plot_ice_conc

<a id="combined___yun00001_3600_vide_o_"></a>
### combined       @ YUN00001_3600/video-->plot_ice_conc
python3 plotIceConcentration.py --images_path=/data/617/images/YUN00001_3600/images --seg_paths=deeplab/log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_YUN00001_3600_z370_190427_174853,unet/log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_YUN00001_3600_max_val_acc_z370_190427_173056,densenet/log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_YUN00001_3600_z370_190427_173049 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=deeplab,unet,densenet --seg_cols=blue,forest_green,magenta --ice_type=0 --out_path=log/ice_concentration/YUN00001_3600  --out_size=1920x720 --enable_plotting=0

<a id="svm___combined_yun00001_3600_video_"></a>
#### svm       @ combined/YUN00001_3600/video-->plot_ice_conc
python3 plotIceConcentration.py --images_path=/data/617/images/YUN00001_3600/images --seg_paths=svm\svm_1_32_2\20160121_YUN00001_900,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_YUN00001_3600_max_val_acc_z370_190427_173056,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_YUN00001_3600_z370_190427_173049 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,unet,densenet --seg_cols=red,forest_green,magenta --ice_type=0 --out_path=log/ice_concentration/YUN00001_3600  --out_size=1920x720 --end_id=899

python3 plotIceConcentration.py --images_path=/data/617/images/YUN00001_3600/images --seg_paths=svm\svm_1_32_2\20160121_YUN00001_900,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_YUN00001_3600_max_val_acc_z370_190427_173056,log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_YUN00001_3600_z370_190427_174853 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,unet,deeplab --seg_cols=red,forest_green,blue --ice_type=0 --out_path=log/ice_concentration/YUN00001_3600_deeplab  --out_size=1920x720 --end_id=899

<a id="frazil___yun00001_3600_vide_o_"></a>
### frazil       @ YUN00001_3600/video-->plot_ice_conc

python3 plotIceConcentration.py --images_path=/data/617/images/YUN00001_3600/images --seg_paths=log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_YUN00001_3600_z370_190427_174853,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_YUN00001_3600_max_val_acc_z370_190427_173056,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_YUN00001_3600_z370_190427_173049 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=deeplab,unet,densenet --seg_cols=blue,forest_green,magenta --ice_type=2 --out_path=log/frazil_ice_concentration/YUN00001_3600  --out_size=1920x720 

<a id="svm___frazil_yun00001_3600_video_"></a>
#### svm       @ frazil/YUN00001_3600/video-->plot_ice_conc

python3 plotIceConcentration.py --images_path=/data/617/images/YUN00001_3600/images --seg_paths=svm\svm_1_32_2\20160121_YUN00001_900,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_YUN00001_3600_max_val_acc_z370_190427_173056,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_YUN00001_3600_z370_190427_173049 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,unet,densenet --seg_cols=red,forest_green,magenta --ice_type=2 --out_path=log/frazil_ice_concentration/YUN00001_3600  --out_size=1920x720 --end_id=899

python3 plotIceConcentration.py --images_path=/data/617/images/YUN00001_3600/images --seg_paths=svm\svm_1_32_2\20160121_YUN00001_900,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_YUN00001_3600_max_val_acc_z370_190427_173056,log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_YUN00001_3600_z370_190427_174853 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,unet,deeplab --seg_cols=red,forest_green,blue --ice_type=2 --out_path=log/frazil_ice_concentration/YUN00001_3600_deeplab  --out_size=1920x720 --end_id=899


<a id="anchor___yun00001_3600_vide_o_"></a>
### anchor       @ YUN00001_3600/video-->plot_ice_conc

python3 plotIceConcentration.py --images_path=/data/617/images/YUN00001_3600/images --seg_paths=log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_YUN00001_3600_z370_190427_174853,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_YUN00001_3600_max_val_acc_z370_190427_173056,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_YUN00001_3600_z370_190427_173049 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=deeplab,unet,densenet --seg_cols=blue,forest_green,magenta --ice_type=1 --out_path=log/anchor_ice_concentration/YUN00001_3600  --out_size=1920x720 

<a id="svm___anchor_yun00001_3600_video_"></a>
#### svm       @ anchor/YUN00001_3600/video-->plot_ice_conc

python3 plotIceConcentration.py --images_path=/data/617/images/YUN00001_3600/images --seg_paths=svm\svm_1_32_2\20160121_YUN00001_900,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_YUN00001_3600_max_val_acc_z370_190427_173056,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_YUN00001_3600_z370_190427_173049 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,unet,densenet --seg_cols=red,forest_green,magenta --ice_type=1 --out_path=log/anchor_ice_concentration/YUN00001_3600  --out_size=1920x720 --end_id=899

python3 plotIceConcentration.py --images_path=/data/617/images/YUN00001_3600/images --seg_paths=svm\svm_1_32_2\20160121_YUN00001_900,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_YUN00001_3600_max_val_acc_z370_190427_173056,log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_YUN00001_3600_z370_190427_174853 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,unet,deeplab --seg_cols=red,forest_green,blue --ice_type=1 --out_path=log/anchor_ice_concentration/YUN00001_3600_deeplab  --out_size=1920x720 --end_id=899


<a id="20160122_yun00002_700_2500___vide_o_"></a>
## 20160122_YUN00002_700_2500       @ video-->plot_ice_conc

<a id="combined___20160122_yun00002_700_2500_video_"></a>
### combined       @ 20160122_YUN00002_700_2500/video-->plot_ice_conc

python3 plotIceConcentration.py --images_path=/data/617/images/20160122_YUN00002_700_2500/images --seg_paths=log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_20160122_YUN00002_700_2500_z370_190423_065334,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20160122_YUN00002_700_2500_max_val_acc_z370_190424_162012,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_20160122_YUN00002_700_2500_z370_190421_161512 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=deeplab,unet,densenet --seg_cols=blue,forest_green,magenta --ice_type=0 --out_path=log/ice_concentration/20160122_YUN00002_700_2500  --out_size=1920x720 

<a id="plot_changed_seg_count___combined_20160122_yun00002_700_2500_vide_o_"></a>
#### plot_changed_seg_count       @ combined/20160122_YUN00002_700_2500/video-->plot_ice_conc

python3 plotIceConcentration.py --images_path=/data/617/images/20160122_YUN00002_700_2500/images --seg_paths=log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_20160122_YUN00002_700_2500_z370_190423_065334,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20160122_YUN00002_700_2500_max_val_acc_z370_190424_162012,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_20160122_YUN00002_700_2500_z370_190421_161512 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=deeplab,unet,densenet --seg_cols=blue,forest_green,magenta --ice_type=0 --out_path=log/ice_concentration/20160122_YUN00002_700_2500  --out_size=1920x720 --plot_changed_seg_count=1

<a id="frazil___20160122_yun00002_700_2500_video_"></a>
### frazil       @ 20160122_YUN00002_700_2500/video-->plot_ice_conc

python3 plotIceConcentration.py --images_path=/data/617/images/20160122_YUN00002_700_2500/images --seg_paths=log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_20160122_YUN00002_700_2500_z370_190423_065334,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20160122_YUN00002_700_2500_max_val_acc_z370_190424_162012,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_20160122_YUN00002_700_2500_z370_190421_161512 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=deeplab,unet,densenet --seg_cols=blue,forest_green,magenta --ice_type=2 --out_path=log/frazil_ice_concentration/20160122_YUN00002_700_2500  --out_size=1920x720 

<a id="anchor___20160122_yun00002_700_2500_video_"></a>
### anchor       @ 20160122_YUN00002_700_2500/video-->plot_ice_conc

python3 plotIceConcentration.py --images_path=/data/617/images/20160122_YUN00002_700_2500/images --seg_paths=log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_20160122_YUN00002_700_2500_z370_190423_065334,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20160122_YUN00002_700_2500_max_val_acc_z370_190424_162012,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_20160122_YUN00002_700_2500_z370_190421_161512 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=deeplab,unet,densenet --seg_cols=blue,forest_green,magenta --ice_type=1 --out_path=log/anchor_ice_concentration/20160122_YUN00002_700_2500  --out_size=1920x720 

<a id="20160122_yun00020_2000_3800___vide_o_"></a>
## 20160122_YUN00020_2000_3800       @ video-->plot_ice_conc

<a id="combined___20160122_yun00020_2000_3800_vide_o_"></a>
### combined       @ 20160122_YUN00020_2000_3800/video-->plot_ice_conc

python3 plotIceConcentration.py --images_path=/data/617/images/20160122_YUN00020_2000_3800/images --seg_paths=log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_20160122_YUN00020_2000_3800_z370_190423_165809,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20160122_YUN00020_2000_3800_max_val_acc_z370_190424_162006,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_20160122_YUN00020_2000_3800_z370_190424_161211 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=deeplab,unet,densenet --seg_cols=blue,forest_green,magenta --ice_type=0 --out_path=log/ice_concentration/20160122_YUN00020_2000_3800  --out_size=1920x720 

<a id="svm___combined_20160122_yun00020_2000_3800_video_"></a>
#### svm       @ combined/20160122_YUN00020_2000_3800/video-->plot_ice_conc

python3 plotIceConcentration.py --images_path=/data/617/images/20160122_YUN00020_2000_3800/images --seg_paths=svm\svm_1_32_2\20160122_YUN00020_2000_2300,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20160122_YUN00020_2000_3800_max_val_acc_z370_190424_162006,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_20160122_YUN00020_2000_3800_z370_190424_161211 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,unet,densenet --seg_cols=red,forest_green,magenta --ice_type=0 --out_path=log/ice_concentration/20160122_YUN00020_2000_3800  --out_size=1920x720 --end_id=299

python3 plotIceConcentration.py --images_path=/data/617/images/20160122_YUN00020_2000_3800/images --seg_paths=svm\svm_1_32_2\20160122_YUN00020_2000_2300,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20160122_YUN00020_2000_3800_max_val_acc_z370_190424_162006,log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_20160122_YUN00020_2000_3800_z370_190423_165809 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,unet,deeplab --seg_cols=red,forest_green,blue --ice_type=0 --out_path=log/ice_concentration/20160122_YUN00020_2000_3800_deeplab  --out_size=1920x720 --end_id=299


<a id="frazil___20160122_yun00020_2000_3800_vide_o_"></a>
### frazil       @ 20160122_YUN00020_2000_3800/video-->plot_ice_conc

python3 plotIceConcentration.py --images_path=/data/617/images/20160122_YUN00020_2000_3800/images --seg_paths=log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_20160122_YUN00020_2000_3800_z370_190423_165809,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20160122_YUN00020_2000_3800_max_val_acc_z370_190424_162006,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_20160122_YUN00020_2000_3800_z370_190424_161211 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=deeplab,unet,densenet --seg_cols=blue,forest_green,magenta --ice_type=2 --out_path=log/frazil_ice_concentration/20160122_YUN00020_2000_3800  --out_size=1920x720

<a id="svm___frazil_20160122_yun00020_2000_3800_video_"></a>
#### svm       @ frazil/20160122_YUN00020_2000_3800/video-->plot_ice_conc

python3 plotIceConcentration.py --images_path=/data/617/images/20160122_YUN00020_2000_3800/images --seg_paths=svm\svm_1_32_2\20160122_YUN00020_2000_2300,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20160122_YUN00020_2000_3800_max_val_acc_z370_190424_162006,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_20160122_YUN00020_2000_3800_z370_190424_161211 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,unet,densenet --seg_cols=red,forest_green,magenta --ice_type=2 --out_path=log/frazil_ice_concentration/20160122_YUN00020_2000_3800  --out_size=1920x720 --end_id=299

python3 plotIceConcentration.py --images_path=/data/617/images/20160122_YUN00020_2000_3800/images --seg_paths=svm\svm_1_32_2\20160122_YUN00020_2000_2300,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20160122_YUN00020_2000_3800_max_val_acc_z370_190424_162006,log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_20160122_YUN00020_2000_3800_z370_190423_165809 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,unet,deeplab --seg_cols=red,forest_green,blue --ice_type=2 --out_path=log/frazil_ice_concentration/20160122_YUN00020_2000_3800_deeplab  --out_size=1920x720 --end_id=299


<a id="anchor___20160122_yun00020_2000_3800_vide_o_"></a>
### anchor       @ 20160122_YUN00020_2000_3800/video-->plot_ice_conc

python3 plotIceConcentration.py --images_path=/data/617/images/20160122_YUN00020_2000_3800/images --seg_paths=log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_20160122_YUN00020_2000_3800_z370_190423_165809,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20160122_YUN00020_2000_3800_max_val_acc_z370_190424_162006,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_20160122_YUN00020_2000_3800_z370_190424_161211 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=deeplab,unet,densenet --seg_cols=blue,forest_green,magenta --ice_type=1 --out_path=log/anchor_ice_concentration/20160122_YUN00020_2000_3800  --out_size=1920x720 

<a id="svm___anchor_20160122_yun00020_2000_3800_video_"></a>
#### svm       @ anchor/20160122_YUN00020_2000_3800/video-->plot_ice_conc

python3 plotIceConcentration.py --images_path=/data/617/images/20160122_YUN00020_2000_3800/images --seg_paths=svm\svm_1_32_2\20160122_YUN00020_2000_2300,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20160122_YUN00020_2000_3800_max_val_acc_z370_190424_162006,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_20160122_YUN00020_2000_3800_z370_190424_161211 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,unet,densenet --seg_cols=red,forest_green,magenta --ice_type=1 --out_path=log/anchor_ice_concentration/20160122_YUN00020_2000_3800  --out_size=1920x720 --end_id=299

python3 plotIceConcentration.py --images_path=/data/617/images/20160122_YUN00020_2000_3800/images --seg_paths=svm\svm_1_32_2\20160122_YUN00020_2000_2300,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20160122_YUN00020_2000_3800_max_val_acc_z370_190424_162006,log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_20160122_YUN00020_2000_3800_z370_190423_165809 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,unet,deeplab --seg_cols=red,forest_green,blue --ice_type=1 --out_path=log/anchor_ice_concentration/20160122_YUN00020_2000_3800_deeplab  --out_size=1920x720 --end_id=299



<a id="20161203_deployment_1_yun00001_900_2700___vide_o_"></a>
## 20161203_Deployment_1_YUN00001_900_2700       @ video-->plot_ice_conc

<a id="combined___20161203_deployment_1_yun00001_900_2700_vide_o_"></a>
### combined       @ 20161203_Deployment_1_YUN00001_900_2700/video-->plot_ice_conc

python3 plotIceConcentration.py --images_path=/data/617/images/20161203_Deployment_1_YUN00001_900_2700/images --seg_paths=log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_20161203_Deployment_1_YUN00001_900_2700_z370_190423_165756,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20161203_Deployment_1_YUN00001_900_2700_max_val_acc_z370_190424_162002,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_20161203_Deployment_1_YUN00001_900_2700_z370_190424_161230 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=deeplab,unet,densenet --seg_cols=blue,forest_green,magenta --ice_type=0 --out_path=log/ice_concentration/20161203_Deployment_1_YUN00001_900_2700  --out_size=1920x720 

<a id="svm___combined_20161203_deployment_1_yun00001_900_2700_video_"></a>
#### svm       @ combined/20161203_Deployment_1_YUN00001_900_2700/video-->plot_ice_conc

python3 plotIceConcentration.py --images_path=/data/617/images/20161203_Deployment_1_YUN00001_2000_2300/images --seg_paths=svm\svm_1_32_2\20161203_Deployment_1_YUN00001_2000_2300,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20161203_Deployment_1_YUN00001_2000_2300_max_val_acc_z370_190424_162002,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_20161203_Deployment_1_YUN00001_2000_2300_z370_190424_161230 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,unet,densenet --seg_cols=red,forest_green,magenta --ice_type=0 --out_path=log/ice_concentration/20161203_Deployment_1_YUN00001_2000_2300_densenet  --out_size=1920x720 --start_id=0 --end_id=299

python3 plotIceConcentration.py --images_path=/data/617/images/20161203_Deployment_1_YUN00001_2000_2300/images --seg_paths=svm\svm_1_32_2\20161203_Deployment_1_YUN00001_2000_2300,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20161203_Deployment_1_YUN00001_2000_2300_max_val_acc_z370_190424_162002,log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_20161203_Deployment_1_YUN00001_2000_2300_z370_190423_165756 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,unet,deeplab --seg_cols=red,forest_green,blue --ice_type=0 --out_path=log/ice_concentration/20161203_Deployment_1_YUN00001_2000_2300_deeplab  --out_size=1920x720 --start_id=0 --end_id=299

<a id="20161203_deployment_1_yun00001_900_1200___svm_combined_20161203_deployment_1_yun00001_900_2700_video_"></a>
##### 20161203_Deployment_1_YUN00001_900_1200       @ svm/combined/20161203_Deployment_1_YUN00001_900_2700/video-->plot_ice_conc

python3 plotIceConcentration.py --images_path=/data/617/images/20161203_Deployment_1_YUN00001_900_2700/images --seg_paths=svm\svm_1_32_2\20161203_Deployment_1_YUN00001_900_1200,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20161203_Deployment_1_YUN00001_900_2700_max_val_acc_z370_190424_162002,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_20161203_Deployment_1_YUN00001_900_2700_z370_190424_161230 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,unet,densenet --seg_cols=red,forest_green,magenta --ice_type=0 --out_path=log/ice_concentration/20161203_Deployment_1_YUN00001_900_1200_densenet  --out_size=1920x720 --start_id=0 --end_id=299

python3 plotIceConcentration.py --images_path=/data/617/images/20161203_Deployment_1_YUN00001_900_2700/images --seg_paths=svm\svm_1_32_2\20161203_Deployment_1_YUN00001_900_1200,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20161203_Deployment_1_YUN00001_900_2700_max_val_acc_z370_190424_162002,log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_20161203_Deployment_1_YUN00001_900_2700_z370_190423_165756 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,unet,deeplab --seg_cols=red,forest_green,blue --ice_type=0 --out_path=log/ice_concentration/20161203_Deployment_1_YUN00001_900_1200_deeplab  --out_size=1920x720 --start_id=0 --end_id=299

<a id="frazil___20161203_deployment_1_yun00001_900_2700_vide_o_"></a>
### frazil       @ 20161203_Deployment_1_YUN00001_900_2700/video-->plot_ice_conc

python3 plotIceConcentration.py --images_path=/data/617/images/20161203_Deployment_1_YUN00001_900_2700/images --seg_paths=log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_20161203_Deployment_1_YUN00001_900_2700_z370_190423_165756,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20161203_Deployment_1_YUN00001_900_2700_max_val_acc_z370_190424_162002,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_20161203_Deployment_1_YUN00001_900_2700_z370_190424_161230 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=deeplab,unet,densenet --seg_cols=blue,forest_green,magenta --ice_type=2 --out_path=log/frazil_ice_concentration/20161203_Deployment_1_YUN00001_900_2700  --out_size=1920x720 

<a id="svm___frazil_20161203_deployment_1_yun00001_900_2700_video_"></a>
#### svm       @ frazil/20161203_Deployment_1_YUN00001_900_2700/video-->plot_ice_conc

python3 plotIceConcentration.py --images_path=/data/617/images/20161203_Deployment_1_YUN00001_2000_2300/images --seg_paths=svm\svm_1_32_2\20161203_Deployment_1_YUN00001_2000_2300,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20161203_Deployment_1_YUN00001_2000_2300_max_val_acc_z370_190424_162002,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_20161203_Deployment_1_YUN00001_2000_2300_z370_190424_161230 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,unet,densenet --seg_cols=red,forest_green,magenta --ice_type=2 --out_path=log/frazil_ice_concentration/20161203_Deployment_1_YUN00001_2000_2300_densenet  --out_size=1920x720 --start_id=0 --end_id=299

python3 plotIceConcentration.py --images_path=/data/617/images/20161203_Deployment_1_YUN00001_2000_2300/images --seg_paths=svm\svm_1_32_2\20161203_Deployment_1_YUN00001_2000_2300,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20161203_Deployment_1_YUN00001_2000_2300_max_val_acc_z370_190424_162002,log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_20161203_Deployment_1_YUN00001_2000_2300_z370_190423_165756 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,unet,deeplab --seg_cols=red,forest_green,blue --ice_type=2 --out_path=log/frazil_ice_concentration/20161203_Deployment_1_YUN00001_900_1200_deeplab  --out_size=1920x720 --start_id=0 --end_id=299

<a id="20161203_deployment_1_yun00001_900_1200___svm_frazil_20161203_deployment_1_yun00001_900_2700_video_"></a>
##### 20161203_Deployment_1_YUN00001_900_1200       @ svm/frazil/20161203_Deployment_1_YUN00001_900_2700/video-->plot_ice_conc

python3 plotIceConcentration.py --images_path=/data/617/images/20161203_Deployment_1_YUN00001_900_2700/images --seg_paths=svm\svm_1_32_2\20161203_Deployment_1_YUN00001_900_1200,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20161203_Deployment_1_YUN00001_900_2700_max_val_acc_z370_190424_162002,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_20161203_Deployment_1_YUN00001_900_2700_z370_190424_161230 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,unet,densenet --seg_cols=red,forest_green,magenta --ice_type=2 --out_path=log/frazil_ice_concentration/20161203_Deployment_1_YUN00001_2000_2300_densenet  --out_size=1920x720 --start_id=0 --end_id=299

python3 plotIceConcentration.py --images_path=/data/617/images/20161203_Deployment_1_YUN00001_900_2700/images --seg_paths=svm\svm_1_32_2\20161203_Deployment_1_YUN00001_900_1200,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20161203_Deployment_1_YUN00001_900_2700_max_val_acc_z370_190424_162002,log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_20161203_Deployment_1_YUN00001_900_2700_z370_190423_165756 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,unet,deeplab --seg_cols=red,forest_green,blue --ice_type=2 --out_path=log/frazil_ice_concentration/20161203_Deployment_1_YUN00001_900_1200_deeplab  --out_size=1920x720 --start_id=0 --end_id=299


<a id="anchor___20161203_deployment_1_yun00001_900_2700_vide_o_"></a>
### anchor       @ 20161203_Deployment_1_YUN00001_900_2700/video-->plot_ice_conc

python3 plotIceConcentration.py --images_path=/data/617/images/20161203_Deployment_1_YUN00001_900_2700/images --seg_paths=log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_20161203_Deployment_1_YUN00001_900_2700_z370_190423_165756,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20161203_Deployment_1_YUN00001_900_2700_max_val_acc_z370_190424_162002,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_20161203_Deployment_1_YUN00001_900_2700_z370_190424_161230 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=deeplab,unet,densenet --seg_cols=blue,forest_green,magenta --ice_type=1 --out_path=log/anchor_ice_concentration/20161203_Deployment_1_YUN00001_900_2700  --out_size=1920x720 

<a id="svm___anchor_20161203_deployment_1_yun00001_900_2700_video_"></a>
#### svm       @ anchor/20161203_Deployment_1_YUN00001_900_2700/video-->plot_ice_conc

<a id="20161203_deployment_1_yun00001_2000_2300___svm_anchor_20161203_deployment_1_yun00001_900_2700_video_"></a>
##### 20161203_Deployment_1_YUN00001_2000_2300       @ svm/anchor/20161203_Deployment_1_YUN00001_900_2700/video-->plot_ice_conc

python3 plotIceConcentration.py --images_path=/data/617/images/20161203_Deployment_1_YUN00001_2000_2300/images --seg_paths=svm\svm_1_32_2\20161203_Deployment_1_YUN00001_2000_2300,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20161203_Deployment_1_YUN00001_2000_2300_max_val_acc_z370_190424_162002,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_20161203_Deployment_1_YUN00001_2000_2300_z370_190424_161230 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,unet,densenet --seg_cols=red,forest_green,magenta --ice_type=1 --out_path=log/anchor_ice_concentration/20161203_Deployment_1_YUN00001_2000_2300_densenet  --out_size=1920x720 --start_id=0 --end_id=299

python3 plotIceConcentration.py --images_path=/data/617/images/20161203_Deployment_1_YUN00001_2000_2300/images --seg_paths=svm\svm_1_32_2\20161203_Deployment_1_YUN00001_2000_2300,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20161203_Deployment_1_YUN00001_2000_2300_max_val_acc_z370_190424_162002,log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_20161203_Deployment_1_YUN00001_2000_2300_z370_190423_165756 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,unet,deeplab --seg_cols=red,forest_green,blue --ice_type=1 --out_path=log/anchor_ice_concentration/20161203_Deployment_1_YUN00001_2000_2300_deeplab  --out_size=1920x720 --start_id=0 --end_id=299

<a id="20161203_deployment_1_yun00001_900_1200___svm_anchor_20161203_deployment_1_yun00001_900_2700_video_"></a>
##### 20161203_Deployment_1_YUN00001_900_1200       @ svm/anchor/20161203_Deployment_1_YUN00001_900_2700/video-->plot_ice_conc

python3 plotIceConcentration.py --images_path=/data/617/images/20161203_Deployment_1_YUN00001_900_2700/images --seg_paths=svm\svm_1_32_2\20161203_Deployment_1_YUN00001_900_1200,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20161203_Deployment_1_YUN00001_900_2700_max_val_acc_z370_190424_162002,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_20161203_Deployment_1_YUN00001_900_2700_z370_190424_161230 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,unet,densenet --seg_cols=red,forest_green,magenta --ice_type=1 --out_path=log/anchor_ice_concentration/20161203_Deployment_1_YUN00001_2000_2300_densenet  --out_size=1920x720 --start_id=0 --end_id=299

python3 plotIceConcentration.py --images_path=/data/617/images/20161203_Deployment_1_YUN00001_900_2700/images --seg_paths=svm\svm_1_32_2\20161203_Deployment_1_YUN00001_900_1200,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20161203_Deployment_1_YUN00001_900_2700_max_val_acc_z370_190424_162002,log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_20161203_Deployment_1_YUN00001_900_2700_z370_190423_165756 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=svm,unet,deeplab --seg_cols=red,forest_green,blue --ice_type=1 --out_path=log/anchor_ice_concentration/20161203_Deployment_1_YUN00001_2000_2300_deeplab  --out_size=1920x720 --start_id=0 --end_id=299



<a id="20161203_deployment_1_yun00002_1800___vide_o_"></a>
## 20161203_Deployment_1_YUN00002_1800       @ video-->plot_ice_conc

<a id="combined___20161203_deployment_1_yun00002_1800_vide_o_"></a>
### combined       @ 20161203_Deployment_1_YUN00002_1800/video-->plot_ice_conc

python3 plotIceConcentration.py --images_path=/data/617/images/20161203_Deployment_1_YUN00002_1800/images --seg_paths=log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_20161203_Deployment_1_YUN00002_1800_z370_190423_165800,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20161203_Deployment_1_YUN00002_1800_max_val_acc_z370_190424_161957,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_20161203_Deployment_1_YUN00002_1800_z370_190424_161903 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=deeplab,unet,densenet --seg_cols=blue,forest_green,magenta --ice_type=0 --out_path=log/ice_concentration/20161203_Deployment_1_YUN00002_1800  --out_size=1920x720 

<a id="frazil___20161203_deployment_1_yun00002_1800_vide_o_"></a>
### frazil       @ 20161203_Deployment_1_YUN00002_1800/video-->plot_ice_conc

python3 plotIceConcentration.py --images_path=/data/617/images/20161203_Deployment_1_YUN00002_1800/images --seg_paths=log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_20161203_Deployment_1_YUN00002_1800_z370_190423_165800,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20161203_Deployment_1_YUN00002_1800_max_val_acc_z370_190424_161957,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_20161203_Deployment_1_YUN00002_1800_z370_190424_161903 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=deeplab,unet,densenet --seg_cols=blue,forest_green,magenta --ice_type=2 --out_path=log/frazil_ice_concentration/20161203_Deployment_1_YUN00002_1800  --out_size=1920x720 

<a id="anchor___20161203_deployment_1_yun00002_1800_vide_o_"></a>
### anchor       @ 20161203_Deployment_1_YUN00002_1800/video-->plot_ice_conc
python3 plotIceConcentration.py --images_path=/data/617/images/20161203_Deployment_1_YUN00002_1800/images --seg_paths=log_training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_31_20161203_Deployment_1_YUN00002_1800_z370_190423_165800,log_vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_20161203_Deployment_1_YUN00002_1800_max_val_acc_z370_190424_161957,log_rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu_predict_acc_20161203_Deployment_1_YUN00002_1800_z370_190424_161903 --seg_root_dir=H:\UofA\617\Project\617_proj_code\log --images_ext=jpg --labels_ext=png --seg_ext=png --n_classes=3 --seg_labels=deeplab,unet,densenet --seg_cols=blue,forest_green,magenta --ice_type=1 --out_path=log/anchor_ice_concentration/20161203_Deployment_1_YUN00002_1800  --out_size=1920x720 










