# il

zrr il_videos log/rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu predict 20160122_YUN00002_700_2500.mkv,20161203_Deployment_1_YUN00001_900_2700.mkv,20160122_YUN00020_2000_3800.mkv,20161203_Deployment_1_YUN00002_1800.mkv

zrr il_videos log/50_10000_10000_800_random_200_4/predict __n__ YUN00001_3600.mkv,20160122_YUN00002_700_2500.mkv,20161203_Deployment_1_YUN00001_900_2700.mkv,20160122_YUN00020_2000_3800.mkv,20161203_Deployment_1_YUN00002_1800.mkv

zrr __n__ log/rt2_training_0_3_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_320_4_elu/predict __n__ YUN00001_3600.mkv,20160122_YUN00002_700_2500.mkv,20161203_Deployment_1_YUN00001_900_2700.mkv,20160122_YUN00020_2000_3800.mkv,20161203_Deployment_1_YUN00002_1800.mkv

zrr il_sel_videos log  _640_640_640_640_0_2_2_,_640_640_640_640_0_100_100_,_640_640_640_640_0_1000_1000_ *.mkv



# unet

zrr unet_videos log/vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip __n__ "20160122_YUN00002_700_2500_*.mkv,20161203_Deployment_1_YUN00001_900_2700_*.mkv,20160122_YUN00020_2000_3800_*.mkv,20161203_Deployment_1_YUN00002_1800_*.mkv"

zrr unet_sel_videos log flip_1K,flip_2,flip_100 *YUN00*.mkv 0

zrr unet_sel_videos log vgg_unet2_0_3_640_640_640_640_1000,vgg_unet2_0_3_640_640_640_640_100,vgg_unet2_0_3_640_640_640_640_2 *20160122_YUN00020_2000_3800*.mkv,*YUN00001_3600_0_3599_640_640*.mkv 0

# deeplab

## 50

zrr __n__ log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_49 __n__ *.mkv 

## 4

zrr __n__ log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_3 __n__ model.ckpt-366751* 
zrr __n__ log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_3 __n__ *.mkv

 
zrr __n__ log/training_0_31_49_640_640_64_256_rot_15_345_4_flip xception,_sel_ *YUN00001_3600*.mkv,*20160122_YUN00020_2000_3800*.mkv

