# segnet

	256
		training
			`done`
			10:07 AM 4/24/2018
				max_mean_acc: [0.9820459020137786, 266, {'acc': [0.99025315046310425], 'loss': [0.024298261704554847], 'val_acc': [0.97383865356445309], 'val_loss': [0.12450393697847098]}]
				max_val_acc: [0.9791737365722656, 685, {'acc': [0.96886163949966431], 'loss': [0.08248276897858986], 'val_acc': [0.97917373657226559], 'val_loss': [0.063531718317826738]}]
		evaluation
			10:13 AM 4/24/2018 started on grs g1
			10:46 AM 4/24/2018 started vis
			11:20 AM 4/24/2018 `done`
				pix_acc: 0.8569880616 mean_acc: 0.7453307296 mean_IU: 0.6324283621 fw_IU: 0.7935499443

		validation
			`done`
		video
			`done`
	384
		training
			`done`
			max_mean_acc: [0.9878571625612675, 494, {'acc': [0.99086067359894514], 'loss': [0.022691741252977238], 'val_acc': [0.98485365152359006], 'val_loss': [0.06106311171224206]}]
			max_val_acc: [0.9852371528744698, 583, {'acc': [0.97692727961111814], 'loss': [0.064230987140639684], 'val_acc': [0.98523715287446978], 'val_loss': [0.053684177661425567]}]
		evaluation
			10:13 AM 4/24/2018 started a while ago on grs g0
			10:46 AM 4/24/2018 started vis
			11:20 AM 4/24/2018 `done`
				pix_acc: 0.8673494160 mean_acc: 0.8012017656 mean_IU: 0.6697248589 fw_IU: 0.8051855060
		validation
			`done`
		video
			`done`
	512
		training
			11:42 PM 4/22/2018 started on hml
			11:31 PM 4/23/2018 stopped after 391 epochs
				max_acc: 0.998145 in epoch 353  acc: 0.9981     val_acc: 0.9631 loss: 0.0044    val_loss: 0.2322 
				max_val_acc: 0.985008 in epoch 352      acc: 0.9924     val_acc: 0.9850 loss: 0.0184    val_loss: 0.0706
				max_mean_acc: 0.988727 in epoch 352     acc: 0.9924     val_acc: 0.9850 loss: 0.0184    val_loss: 0.0706
				min_loss: 0.004360 in epoch 353 acc: 0.9981     val_acc: 0.9631 loss: 0.0044    val_loss: 0.2322
				min_val_loss: 0.070002 in epoch 139     acc: 0.9950     val_acc: 0.9802 loss: 0.0131    val_loss: 0.0700
				min_mean_loss: 0.041532 in epoch 139    acc: 0.9950     val_acc: 0.9802 loss: 0.0131    val_loss: 0.0700 
		evaluation
			11:36 PM 4/23/2018 started on hml gpu in batch
				batch_predict_512_unet_segnet_201804232329.sh
			5:06 AM 4/24/2018 `done` prediction
			7:29 AM 4/24/2018 restarted after fixing command
			10:15 AM 4/24/2018 `done`
				pix_acc: 0.8759492947 mean_acc: 0.8317681320 mean_IU: 0.6971081121 fw_IU: 0.8106137736
		validation
			8:24 AM 4/24/2018 done on hml
			
			
	640
		training
			5:44 PM 4/24/2018 started on grs g1
			6:41 PM 4/24/2018 stopped after 10 epochs
			9:08 PM 4/24/2018 started on hml;
			9:16 PM 4/24/2018 Restarted yet again after generating the 640 data one HML;
			8:29 PM 4/26/2018 done after 519 epochs
				max_acc: 0.993718 in epoch 451  acc: 0.9937     val_acc: 0.8668 loss: 0.0152    val_loss: 1.1544 
				max_val_acc: 0.962039 in epoch 305      acc: 0.9914     val_acc: 0.9620 loss: 0.0213    val_loss: 0.2087  
				max_mean_acc: 0.977116 in epoch 411     acc: 0.9925     val_acc: 0.9617 loss: 0.0181    val_loss: 0.2307  
				min_loss: 0.015190 in epoch 451 acc: 0.9937     val_acc: 0.8668 loss: 0.0152    val_loss: 1.1544 
				min_val_loss: 0.119342 in epoch 93      acc: 0.9790     val_acc: 0.9592 loss: 0.0551    val_loss: 0.1193 
				min_mean_loss: 0.087198 in epoch 93     acc: 0.9790     val_acc: 0.9592 loss: 0.0551    val_loss: 0.1193  
		evaluation
			8:32 PM 4/26/2018 started prediction on hml
			8:52 PM 4/26/2018 started vis too
			9:23 PM 4/26/2018 `done` a while ago
		validation
			9:23 PM 4/26/2018 started on hml
			9:24 PM 4/26/2018 `done` too
			9:26 PM 4/26/2018 stitching and getting in the results as well;
	640 - 4
		training
			8:58 AM 9/22/2018 `started` on GRS0
			2:17 PM 9/25/2018 `resumed` at epoch 909 on GRS_2
			11:36 PM 9/25/2018 `done` 1000 epochs on GRS_2	
			
	640 - 8
		training	
			9:58 AM 9/27/2018 `started` on GRS_0	
			
	640 - 16
		training	
			11:40 PM 9/25/2018 `started` on GRS_2		
			
	640 - 24
		training
			8:38 PM 9/22/2018 `started` on Z370_1			
			2:32 PM 9/23/2018 `resumed` at epoch 146 on Z370_1			
			1:49 PM 9/24/2018 `resumed` at epoch 342 on Z370_0			
			2:17 PM 9/25/2018 `resumed` at epoch 535 on GRS_0	