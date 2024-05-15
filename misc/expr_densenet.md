# densenet
	384
		10k/10k/600-random/32
			training
				9:40 PM 4/23/2018 started on grs g1;
					348 images
				6:55 AM 4/24/2018 `done` 1500 epochs
					50_10000_10000_384_random_601_4 :: epoch:   1580 loss: 3371.8801269531 min_loss: 3364.5822753906 pix_acc: 0.9519604492 mean_acc: 0.8045273719 mean_IU: 0.6873729924 fw_IU: 0.9325010022
			evaluation
				7:04 AM 4/24/2018 started batch job on grs g0
					batch_predict_384_201804240701.sh
				7:10 AM 4/24/2018 restarted after fixing command
				7:15 AM 4/24/2018 restarted yet again in eval mode
				8:36 AM 4/24/2018 `done`
					pix_acc: 0.8423283204 mean_acc: 0.7937176733 mean_IU: 0.6464628243 fw_IU: 0.7798132670
			validation
				7:04 AM 4/24/2018 started batch job on grs g0
					batch_predict_384_201804240701.sh
				7:06 AM 4/24/2018 `done`
					
					
	512
		10k/10k/500-random/32
			training
				9:40 PM 4/23/2018 started on grs g0 a while ago;
					361 images
				6:55 AM 4/24/2018 `done` 1330 epochs
					50_10000_10000_512_random_501_4 :: epoch:   1330 loss: 3357.5942382812 min_loss: 3351.0498046875 pix_acc: 0.9644547653 mean_acc: 0.8511160057 mean_IU: 0.7036533022 fw_IU: 0.9465249728
			evaluation
				7:04 AM 4/24/2018 started batch job on grs g1
					batch_predict_512_201804240701.sh
				7:10 AM 4/24/2018 restarted after fixing command
				7:15 AM 4/24/2018 restarted yet again in eval mode
				8:35 AM 4/24/2018 `done`
					pix_acc: 0.8497684871 mean_acc: 0.8128675727 mean_IU: 0.6604735600 fw_IU: 0.7837015512
			validation
				7:04 AM 4/24/2018 started batch job on grs g1
					batch_predict_512_201804240701.sh
				7:06 AM 4/24/2018 `done`
	640
		10k/10k/400-random
			training
				12:57 PM 4/22/2018 started on grs g1;
				2:40 PM 4/23/2018 stopped after 1350 epochs/ 349 images;
					50_10000_10000_640_random_401_4 :: epoch:   1350 loss: 3384.2785644531 min_loss: 3383.9147949219 pix_acc: 0.9674024414 mean_acc: 0.8703747205 mean_IU: 0.7274847911 fw_IU: 0.9479425185
			evaluation
				2:50 PM 4/23/2018 started on grs g0
				4:14 PM 4/23/2018 `done`
					50_10000_10000_640_random_401_4
			validation
			video
			examination	
				4:20 PM 4/23/2018 started on grs
				4:28 PM 4/23/2018 done
					pix_acc: 0.8725457676 mean_acc: 0.8412989248 mean_IU: 0.6971669251 fw_IU: 0.8093490764
		4_non_aug
			training - sel_2
				12:39 PM 10/30/2018 started on z370_1;
				6:03 PM 10/30/2018 completed ~1100 epochs;
	800
		10k/10k/100		
			training
				`done`/~1600 epochs
			evaluation
				9:50 AM 4/21/2018 running on e5g
				10:19 AM 4/21/2018 `completed`
			validation
				9:48 AM 4/21/2018 running on grs
				11:48 AM 4/21/2018 restarted on grs g1				
			video			
		10k/10k/200-random
			training
				started on e5g but got stuck between 0-10 epochs, turns out that it was actually running on the CPU
				11:49 AM 4/21/2018 restarted on grs g0
				8:01 AM 4/22/2018 `done`/~1400 epochs
			evaluation
				9:09 AM 4/22/2018 Started on grey shark GPU 0
				9:13 AM 4/22/2018 `done`
			validation
				9:24 AM 4/22/2018 Started a while ago on grey shark
				9:29 AM 4/22/2018 `done`
			video
				11:11 AM 4/24/2018 started on grs g0
				11:47 AM 4/24/2018 done stitching and zipping
					50_10000_10000_800_random_200_4_predict_YUN00001_0_239_800_800_800_800_stitched_grs_201804241144.zip
				5:21 PM 6/21/2018
					20160122_YUN00002_700_0_1799	GRS
				7:05 PM 6/21/2018
					20160122_YUN00002_700_0_1799_800_800_800_800	GRS		
					
			examination			
				50_10000_10000_800_random_200_4_predict_training_0_49_800_800_100_200_1_10_grs_201804220958.zip
				50_10000_10000_800_random_200_4_predict_validation_0_563_800_800_800_800_grs_201804221007.zip
				pix_acc: 0.5834871217 mean_acc: 0.3230916377 mean_IU: 0.3174556230 fw_IU: 0.5753989708
				12:41 PM 4/22/2018 restarted on GRS GPU 1 after realizing that the processed labels cannot be compared with the GT because of their different class labels
				pix_acc: 0.8915896919 mean_acc: 0.8557205494 mean_IU: 0.7226243696 fw_IU: 0.8303422430
		10k/10k/all		
			training
				10:44 AM 4/22/2018 Started on GRS0;
				12:48 PM 4/22/2018 restarted after fixing the stitching bug in the script
				2:45 PM 4/22/2018 Stopped after 30 epochs
			evaluation
			validation
			video
		
		4
			training
				4:06 PM 9/1/2018 done >1000 epochs on Z370
			evaluation
				6:47 PM 10/2/2018 `done`
			training - 2
				2:39 PM 10/23/2018 started on z370_1
			training - 5
				2:10 PM 10/23/2018 started on z370_0
			training - 10
				3:38 PM 10/22/2018 started on z370_1
				2:08 PM 10/23/2018 stopped after 322 epochs
			training - 100
				1:08 AM 10/19/2018 started on z370_1
				2:50 PM 10/19/2018 started on z370_1 after disabling preload_images
			training - 1K
				1:17 AM 10/19/2018 done 872 epochs on Z370
					Testing...
					Done   831/  831 frames in epoch   871 ( 12.28( 12.28,   9.34) fps) pix_acc: 0.8405685601
					rt2_training_0_3_800_800_80_320_rot_15_345_4_flip_0_1000_1000_800_0_320_4_elu :: epoch:  871 loss: 150.0467588212 min_loss: 149.4092276021(838) pix_acc: 0.8405685601 max_pix_acc:  0.8491891245(18) lr: 0.0000115
					333
					Done   318/  318 frames in epoch   872 (  4.16/  4.14 fps) avg_loss: 150.041773
			training - 5K
				1:03 AM 10/19/2018 started on z370_0
				2:50 PM 10/19/2018 started on z370_0 after disabling preload_images
				11:15 PM 10/20/2018 stopped after 808 epochs
				
		8
			training
				4:06 PM 9/1/2018 started on Z370
				10:51 AM 9/3/2018 done 1064 epochs on Z370
					50_8000_8000_800_0_431_4 :: epoch:   1060 loss: 1838.6987304688 min_loss: 1795.9851074219 pix_acc: 0.8583818594 max_pix_acc:  0.8606304531 mean_acc: 0.8273964191 mean_IU: 0.6877002815 fw_IU: 0.7778425296
			evaluation
				6:47 PM 10/2/2018 `done`
				
		16
			training
				10:52 AM 9/3/2018 started on Z370
				11:07 AM 9/3/2018 Restarted after incorporating FPS in the training script;
				9/8/2018 killed itself after 590 epochs; resuming causes NaN loss
				9/8/2018 restarted on grs
				8:52 AM 9/11/2018 done 1020 epochs:
					Done   846/  846 frames in epoch  1020 (  4.24 fps)
					50_8000_8000_800_0_887_4 :: epoch:   1020 loss: 2219.5410156250 min_loss: 2218.8395996094(960) pix_acc: 0.8717792187 max_pix_acc:  0.8749398906(150) mean_acc: 0.8301249636 mean_IU: 0.7039399987 fw_IU: 0.7949288783
			evaluation
				6:47 PM 10/2/2018 `done`
			training - 100
				11:15 PM 10/20/2018 started on z370_0
				2:08 PM 10/23/2018 completed 684 epochs
				
		24
			training
				9/3/2018 started on grs
				9/8/2018 completed 1400 epochs
			evaluation
				6:47 PM 10/2/2018 `done`
			retraining
				11:37 PM 10/4/2018 `started` on GRS_2
				
		32
			training
				6:50 PM 10/2/2018 seem to have completed 870 epochs
			evaluation
				6:47 PM 10/2/2018 `done`
			retraining
				11:37 PM 10/4/2018 `started` on GRS_2
				
		10k/10k/all/augmented	
			training
				12:55 PM 4/28/2018 Started on GRS1;
			evaluation
			validation
			video
	1000
		10k/10k/400-random
			training
				7:02 AM 4/26/2018 started on e5g g1
				7:09 AM 4/26/2018 started on grs g1 as e5g seems to have hung up;
					381 images
				10:17 PM 4/26/2018 `done` with 360 epochs
					50_10000_10000_1000_random_401_4 :: epoch:    360 loss: 3448.2766113281 min_loss: 3448.2766113281 pix_acc: 0.9273357700 mean_acc: 0.8790904955 mean_IU: 0.7635891667 fw_IU: 0.8772128062      
			evaluation
				10:19 PM 4/26/2018 started prediction
				10:23 PM 4/26/2018 started vis
				10:26 PM 4/26/2018 `done`
					pix_acc: 0.8641319173 mean_acc: 0.8402155914 mean_IU: 0.6995547771 fw_IU: 0.7931711310
			video
				5:16 PM 6/21/2018
					YUN00002_2000_0_1799
					20161201_YUN00002_0_1799
				5:31 PM 6/21/2018 imgSeqToVideo on grs 
					YUN00001_1920x1080_0_1799_1000_1000_1000_1000
					YUN00002_1920x1080_0_1799_1000_1000_1000_1000
		10k/10k/800-random
			training	
				5:25 PM 6/30/2018 completed 1250 steps / 765 images
					50_10000_10000_1000_random_801_4 :: epoch:   1250 loss: 3308.8352050781 min_loss: 3306.3183593750 pix_acc: 0.9335003700 mean_acc: 0.8924250439 mean_IU: 0.7856740653 fw_IU: 0.8858462636
			evaluation
				11:26 AM 7/1/2018 done
					pix_acc: 0.7006610720 mean_acc: 0.6693179808 mean_IU: 0.4934615151 fw_IU: 0.5940034899
			video
				11:30 AM 7/1/2018 done quite a few:
					20170114_YUN00005_0_1799_1000_1000_1000_1000_stitched
					YUN00001_1920x1080_0_1799_1000_1000_1000_1000_stitched
					YUN00002_2000_0_1799_1000_1000_1000_1000_stitched
					20160122_YUN00002_700_0_1799_1000_1000_1000_1000_stitched
					20161201_YUN00002_0_1799_1000_1000_1000_1000_stitched
				11:32 AM 7/1/2018 started YUN00001_0_1799_1000_1000_1000_1000