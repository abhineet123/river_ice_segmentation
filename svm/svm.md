# svm_1_4_2

SVM_Predict(5, 32, 2, 1, 4, 'P:\Datasets\617\images\training_4_49', 'img_%d', 'png', 'log\training_4_49');

# svm_1_32_2

## YUN00001_3600       @ svm_1_32_2

SVM_Predict(1, 300, 2, 1, 32, 'P:\Datasets\617\images\20160121_YUN00001_900', 'image%06d', 'jpg', 'log\20160121_YUN00001_900');

SVM_Predict(301, 600, 2, 1, 32, 'P:\Datasets\617\images\20160121_YUN00001_900', 'image%06d', 'jpg', 'log\20160121_YUN00001_900');

SVM_Predict(601, 900, 2, 1, 32, 'P:\Datasets\617\images\20160121_YUN00001_900', 'image%06d', 'jpg', 'log\20160121_YUN00001_900');

### stitching       @ YUN00001_3600/svm_1_32_2

python ../stitchSubPatchDataset.py src_path=E:\Datasets\617\images\YUN00001_3600\images patch_seq_path=H:\UofA\617\Project\617_proj_code\svm\log\svm_1_32_2\20160121_YUN00001_900 stitched_seq_path=H:\UofA\617\Project\617_proj_code\svm\log\svm_1_32_2\20160121_YUN00001_900 patch_height=800 patch_width=800 start_id=0 end_id=899 show_img=1 stacked=0 method=-1 normalize_patches=0 img_ext=png del_patch_seq=0 out_ext=mkv width=1920 height=1080

## 20160122_YUN00020_2000_2300       @ svm_1_32_2

SVM_Predict(1, 100, 2, 1, 32, 'P:\Datasets\617\images\20160122_YUN00020_2000_2300', 'image%06d', 'jpg', 'log\20160122_YUN00020_2000_2300');

SVM_Predict(101, 200, 2, 1, 32, 'P:\Datasets\617\images\20160122_YUN00020_2000_2300', 'image%06d', 'jpg', 'log\20160122_YUN00020_2000_2300');

SVM_Predict(201, 300, 2, 1, 32, 'P:\Datasets\617\images\20160122_YUN00020_2000_2300', 'image%06d', 'jpg', 'log\20160122_YUN00020_2000_2300');

### stitching       @ 20160122_YUN00020_2000_2300/svm_1_32_2

python ../stitchSubPatchDataset.py src_path=H:\UofA\617\Project\617_proj_code\svm\log\svm_1_32_2\20160122_YUN00020_2000_2300 patch_seq_path=H:\UofA\617\Project\617_proj_code\svm\log\svm_1_32_2\20160122_YUN00020_2000_2300 stitched_seq_path=H:\UofA\617\Project\617_proj_code\svm\log\svm_1_32_2\20160122_YUN00020_2000_2300 patch_height=800 patch_width=800 start_id=0 end_id=-1 show_img=1 stacked=0 method=-1 normalize_patches=0 img_ext=png del_patch_seq=0 out_ext=mkv width=1920 height=1080

## 20161203_Deployment_1_YUN00001_900_1200       @ svm_1_32_2

SVM_Predict(1, 100, 2, 1, 32, 'P:\Datasets\617\images\20161203_Deployment_1_YUN00001_900_1200', 'image%06d', 'jpg', 'log\20161203_Deployment_1_YUN00001_900_1200');

SVM_Predict(101, 200, 2, 1, 32, 'P:\Datasets\617\images\20161203_Deployment_1_YUN00001_900_1200', 'image%06d', 'jpg', 'log\20161203_Deployment_1_YUN00001_900_1200');

SVM_Predict(201, 300, 2, 1, 32, 'P:\Datasets\617\images\20161203_Deployment_1_YUN00001_900_1200', 'image%06d', 'jpg', 'log\20161203_Deployment_1_YUN00001_900_1200');

### stitching       @ 20161203_Deployment_1_YUN00001_900_1200/svm_1_32_2

python ../stitchSubPatchDataset.py src_path=H:\UofA\617\Project\617_proj_code\svm\log\svm_1_32_2\20161203_Deployment_1_YUN00001_900_1200 patch_seq_path=H:\UofA\617\Project\617_proj_code\svm\log\svm_1_32_2\20161203_Deployment_1_YUN00001_900_1200 stitched_seq_path=H:\UofA\617\Project\617_proj_code\svm\log\svm_1_32_2\20161203_Deployment_1_YUN00001_900_1200 patch_height=800 patch_width=800 start_id=0 end_id=-1 show_img=1 stacked=0 method=-1 normalize_patches=0 img_ext=png del_patch_seq=0 out_ext=mkv width=1920 height=1080

