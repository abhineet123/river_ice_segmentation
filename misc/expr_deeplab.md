	
# deeplab
	256	
		50
			stride 16
				training
					`done`
				evaluation
					6:34 PM 4/23/2018 started in a batch job on grs g0
					8:39 PM 4/23/2018 done prediction a while ago
					11:13 PM 4/23/2018 `done`
						pix_acc: 0.5947307221 mean_acc: 0.5481592878 mean_IU: 0.3538180877 fw_IU: 0.4856798606
				validation
					9:30 PM 4/23/2018 started in a batch job on grs g0
						batch_vis_validation_0_20_201804232121.sh;
					9:35 PM 4/23/2018 `done`
				video
			stride 8
				training
					partially `done` on grs/58440 steps
				evaluation
					running on e5g cpu
					5:51 PM 4/22/2018 `completed`
						miou_1.0[0.799446404]
					11:15 PM 4/23/2018 started on e5g cpu
					9:32 PM 4/24/2018 e5g gpu finally became free so started on g0;
					12:02 AM 4/25/2018 started vis on grs after transferring over the segs;
					10:03 AM 4/25/2018 finally `done`
						pix_acc: 0.8429068424 mean_acc: 0.7390817076 mean_IU: 0.6134979497 fw_IU: 0.7710768508						
				validation
					5:51 PM 4/22/2018 started on e5g cpu
					7:29 AM 4/23/2018 `done` on e5g					
					
		32
			training
				`done`
	384
		50
			batch size 6
				stride 16
					training
						`done`
						12:41 AM 4/25/2018 restarted on e5g g0
						11:23 AM 4/25/2018 Stopped second attempt;						
					evaluation
						6:34 PM 4/23/2018 started in a batch job on grs g0
						8:39 PM 4/23/2018 done prediction
						11:16 PM 4/23/2018 done
							pix_acc: 0.2852180532 mean_acc: 0.3339480287 mean_IU: 0.1349384901 fw_IU: 0.1656293053							
					validation
						9:30 PM 4/23/2018 started in a batch job on grs g0
							batch_vis_validation_0_20_201804232121.sh;
						9:35 PM 4/23/2018 `done`
				stride 8
					training
						12:41 AM 4/25/2018 started on e5g g1
						12:43 AM 4/25/2018 ran into resource exhausted error as is typical;	
			batch size 8
				training
					11:23 AM 4/25/2018 started on e5g g0 
					6:33 AM 4/26/2018 `done`
				evaluation
					6:39 AM 4/26/2018 started on e5g g0;
					10:42 AM 4/26/2018 restarted after the earlier one crashed
					12:09 PM 4/26/2018 `done` prediction
					3:09 PM 4/26/2018 finally started vis on grs;
					4:39 PM 4/26/2018 done
						pix_acc: 0.2667742556 mean_acc: 0.3363098762 mean_IU: 0.1145519463 fw_IU: 0.1324344936
			
		32
			training
				`done`
	512
		50
			batch size 6
				training
					`done`
				evaluation
					6:34 PM 4/23/2018 started in a batch job on grs g0
					9:08 PM 4/23/2018 `done` prediction
					5:07 AM 4/24/2018 `done`
						pix_acc: 0.5120640124 mean_acc: 0.3412964797 mean_IU: 0.2057745358 fw_IU: 0.3422805291
				validation
					9:30 PM 4/23/2018 started in a batch job on grs g0
						batch_vis_validation_0_20_201804232121.sh;
					9:35 PM 4/23/2018 `done`				
			batch size 2
				training
					8:22 PM 4/25/2018 done on e5g g1
					8:25 PM 4/25/2018 Started prediction on e5g g1
					6:33 AM 4/26/2018 `done` prediction
					9:11 AM 4/26/2018 E5G started responding again so started evaluation of the predictions
					10:34 AM 4/26/2018 `done`
						pix_acc: 0.8881209743 mean_acc: 0.7990793359 mean_IU: 0.7057903068 fw_IU: 0.8192340437
			
		32
			training
				`done`
	640
		50
			training
				1:33 PM 4/21/2018 started on grs g1 with batch size 2
				7:26 AM 4/22/2018 `done` on grs
			evaluation
				7:48 AM 4/22/2018 started on grs g1
				7:53 AM 4/22/2018 `done`
				11:57 AM 4/22/2018 redid it on grs after discovering that eval.py does not actually save the results so vis.py has to be used instead
					training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_49_validation_0_563_640_640_640_640_1_25_grs_201804221134.zip
					training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_49_validation_0_563_640_640_640_640_raw_1_25_grs_201804221136.zip
				pix_acc: 0.8847400938 mean_acc: 0.8086450872 mean_IU: 0.7193816555 fw_IU: 0.8192872585				
			validation
				7:54 AM 4/22/2018 started on grs g1
				8:00 AM 4/22/2018 `done`
				stitching:
					started on grs				
			validation_20
				9:30 PM 4/23/2018 started in a batch job on grs g0
					batch_vis_validation_0_20_201804232121.sh;
			video
				11:11 AM 4/24/2018 started on grs g1
		4
			training
				10:30 AM 9/10/2018 `started` on z370_0 with batch size 2;
				7:18 PM 9/10/2018 stopped after multiple attempts to get it working without hogging up the system but getting no success and hearing alarming sounds from the CPU;
				9:11 AM 9/14/2018 `completed` 100K + 40K epochs on z370_0
				10:11 AM 9/27/2018 `resumed` on GRS_1
				11:21 PM 9/27/2018 `done` 72K more iterations on GRS_1
				10:38 PM 9/30/2018 `resumed` on GRS_2
				6:57 AM 10/1/2018 `resumed` at 100K iterations on GRS_2
				5:39 PM 10/1/2018 `stopped` after 175K iterations on GRS_2
				1:19 PM 10/2/2018 `resumed` at 175K iterations on GRS_2
				4:19 PM 10/3/2018 `stopped` after 366K iterations on GRS_2
			evaluation
				3:17 PM 10/4/2018 done 0.865344316	0.814421193	0.694913526	0.795548749
		8
			training
				9:33 AM 9/14/2018 started on z370_0 with batch size 2;
				early morning 9/15/2018 started on z370_0 with batch size 2;
				11:31 PM 9/27/2018 `resumed` on GRS_1
				11:34 PM 9/28/2018 `done` 230K iterations on GRS_1
				6:57 AM 10/1/2018 `resumed` on GRS_0
				8:19 AM 10/2/2018 `stopped` 404K iterations on GRS_0
				2:25 PM 10/3/2018 `resumed` on GRS_0
				7:45 PM 10/3/2018 `stopped` after 438K iterations on GRS_2 to change PSU
				9:24 PM 10/3/2018 `resumed` on GRS_0
				8:45 AM 10/4/2018 `stopped` after 516K iterations on GRS_0
			evaluation
				3:17 PM 10/4/2018 done 0.885127123	0.830072479	0.721393762	0.821320539
		16
			training
				late night 9/15/2018 Completed on z370_0 with batch size 2;	
				11:50 PM 9/28/2018 `resumed` on GRS_1
				7:43 AM 9/30/2018 `done` 273K iterations on GRS_1
				5:40 PM 10/1/2018 `resumed` after 272K iterations on GRS_2
				1:19 PM 10/2/2018 `stopped` after 412K iterations on GRS_2
				4:20 PM 10/3/2018 `resumed` after 410K iterations on GRS_2
				7:44 PM 10/3/2018 `stopped` after 433K iterations on GRS_2 to change PSU
				9:24 PM 10/3/2018 `resumed` after 433K iterations on GRS_2
				8:45 AM 10/4/2018 `stopped` after 515K iterations on GRS_2
			evaluation
				3:17 PM 10/4/2018 done 0.844020258	0.748278433	0.646771214	0.759152969
		24
			training
				10:32 AM 9/16/2018 Completed on z370_0 with batch size 2;	
				5:02 PM 9/29/2018 `resumed` on GRS_2
				10:36 PM 9/30/2018 `done` 310K iterations on GRS_2
				8:20 AM 10/2/2018 `resumed` on GRS_0
				2:25 PM 10/3/2018 `stopped` after 514K iterations on GRS_0
			evaluation
				3:17 PM 10/4/2018 done 0.890282616	0.831576893	0.726620716	0.82335421
		32
			training
				10:54 AM 9/16/2018 started on z370_0 with batch size 2;	
				7:47 AM 9/30/2018 `resumed` on GRS_1
				sometime before 2:29 PM 10/3/2018 `stopped` after 406K iterations on GRS_1
				9:24 PM 10/3/2018 `resumed` on GRS_1
				3:17 PM 10/4/2018 `stopped` after 502K iterations on GRS_1
			evaluation
				3:18 PM 10/4/2018 `started` on GRS_0
				3:17 PM 10/4/2018 done 0.865344316	0.814421193	0.694913526	0.795548749
	800
		50
			training
				6:40 PM 4/24/2018 started on grs g1
				9:05 PM 4/24/2018 restarted after correctly building data on grs g1; 
			evaluation
				9:00 PM 4/26/2018 started building training_4_49_800_800_80_320_rot_15_345_4_flip tfrecord on grs g0
				9:33 PM 4/26/2018 started prediction
				9:36 PM 4/26/2018 done prediction
				9:37 PM 4/26/2018 started vis
				9:42 PM 4/26/2018 `done`
					pix_acc: 0.8619450530 mean_acc: 0.7717653977 mean_IU: 0.6714336220 fw_IU: 0.7834769640
		4
			training
				10:14 AM 9/9/2018 started on z370 with batch size 1 as 2 causes out of memory error;
				9:57 AM 9/10/2018 trying to restart it after it ran into nan summary histogram error on step 1158