# unet
	256
		training
			`done`
		validation
			`done`		
		video
			`done`
		evaluation
			training_256_256_100_200_rot_90_180_flip
				5:14 PM 4/22/2018 started on hml: 
				5:47 PM 4/22/2018 completed
					pix_acc: 0.5863670237 mean_acc: 0.3884212477 mean_IU: 0.3724332862 fw_IU: 0.5695124158
				6:29 PM 4/22/2018 the above is apparently wrong
				8:43 PM 4/22/2018 `done` finally
					pix_acc: 0.8908632070 mean_acc: 0.8183751752 mean_IU: 0.7168346737 fw_IU: 0.8381554005
			training_32_49_256_256_25_100_rot_15_125_235_345_flip			
				6:30 PM 4/22/2018 started on hml
				8:44 PM 4/22/2018 `done`
				9:24 PM 4/22/2018 done evaluating
					pix_acc: 0.8659944935 mean_acc: 0.7836832797 mean_IU: 0.6710072995 fw_IU: 0.8048341307				
			
	384
		training
			`done`
		validation
			`done`
		video
			`done`
		evaluation
			training_32_49_384_384_25_100_rot_15_345_4_flip
				6:30 PM 4/22/2018 started on hml: training_32_49_384_384_25_100_rot_15_345_4_flip
				8:46 PM 4/22/2018 Restarted after fixing bugs in the command
				9:33 PM 4/22/2018 done generating predictions and started evaluating
				11:37 PM 4/22/2018 `done`
					pix_acc: 0.8677582586 mean_acc: 0.8155066878 mean_IU: 0.6883941679 fw_IU: 0.8075073098
			
	512
		training
			3:07 PM 4/22/2018 finally started on GRS0 after fixes in theano flags and reinstalling h5py;
			6:27 PM 4/23/2018 stopped after 410 epochs
				vgg_unet2_0_31_512_512_25_100_rot_15_345_4_flip_410e_grs_201804231831.zip
				max_acc: [0.9984657615423203, 311, {'acc': [0.9984657615423203], 'loss': [0.003678515428141793], 'val_acc': [0.7775251770019531], 'val_loss': [2.1743075180053713]}]
				max_val_acc: [0.985575065612793, 140, {'acc': [0.9903854876756668], 'loss': [0.023903482635432738], 'val_acc': [0.985575065612793], 'val_loss': [0.06675477372496971]}]
				max_mean_acc: [0.9879802766442298, 140, {'acc': [0.9903854876756668], 'loss': [0.023903482635432738], 'val_acc': [0.985575065612793], 'val_loss': [0.06675477372496971]}]
		evaluation
			11:36 PM 4/23/2018 started on hml gpu in batch
				batch_predict_512_unet_segnet_201804232329.sh
			7:22 AM 4/24/2018 `done` prediction
			7:29 AM 4/24/2018 restarted after fixing command
			7:53 AM 4/24/2018 `done`
			7:57 AM 4/24/2018 started vis
			9:02 AM 4/24/2018 
				pix_acc: 0.8649815896 mean_acc: 0.8246868429 mean_IU: 0.6863907490 fw_IU: 0.7983977621
		validation
			8:24 AM 4/24/2018 done on hml
			
			
	640 - 32
		training
			5:41 PM 4/24/2018 started on grs g0
			8:22 PM 4/26/2018 `done` after 538 epochs
				max_acc: 0.994655 in epoch 514  acc: 0.9947     val_acc: 0.8989 loss: 0.0128    val_loss: 0.8316
				max_val_acc: 0.970356 in epoch 189      acc: 0.9849     val_acc: 0.9704 loss: 0.0397    val_loss: 0.1405
				max_mean_acc: 0.977900 in epoch 481     acc: 0.9929     val_acc: 0.9629 loss: 0.0173    val_loss: 0.2430 
				min_loss: 0.012769 in epoch 514 acc: 0.9947     val_acc: 0.8989 loss: 0.0128    val_loss: 0.8316 
				min_val_loss: 0.133594 in epoch 161     acc: 0.9847     val_acc: 0.9689 loss: 0.0405    val_loss: 0.1336 
				min_mean_loss: 0.087051 in epoch 161    acc: 0.9847     val_acc: 0.9689 loss: 0.0405    val_loss: 0.1336
		evaluation
			8:28 PM 4/26/2018 started predictions on grs
			8:34 PM 4/26/2018 started vis too
			8:50 PM 4/26/2018 `done`
				pix_acc: 0.8877982202 mean_acc: 0.8467476490 mean_IU: 0.7209879922 fw_IU: 0.8236038126 
			9:00 PM 4/26/2018 `done`
				pix_acc: 0.8850818522 mean_acc: 0.8623441917 mean_IU: 0.7264261381 fw_IU: 0.8240310859 
		validation
			9:17 PM 4/26/2018 started on grs
			9:17 PM 4/26/2018 `done` too
			9:22 PM 4/26/2018 stitching and getting in the results as well;
	640 - 4
		training
			8:58 AM 9/22/2018 `done` on GRS0 after 1000 epochs
				Epoch 1000/1000
				Epoch 1/1
				512/512 [==============================] - 297s 580ms/step - loss: 0.0013 - acc: 0.9996 - val_loss: 2.8044 - val_acc: 0.8027
				max_acc: 0.999730 in epoch 935  acc: 0.9997     val_acc: 0.8644 loss: 0.0009    val_loss: 1.9804
				max_val_acc: 0.950050 in epoch 144      acc: 0.9925     val_acc: 0.9500 loss: 0.0207    val_loss: 0.4572
				max_mean_acc: 0.971293 in epoch 144     acc: 0.9925     val_acc: 0.9500 loss: 0.0207    val_loss: 0.4572
				min_loss: 0.000864 in epoch 935 acc: 0.9997     val_acc: 0.8644 loss: 0.0009    val_loss: 1.9804
				min_val_loss: 0.214141 in epoch 4       acc: 0.9255     val_acc: 0.9332 loss: 0.1795    val_loss: 0.2141
				min_mean_loss: 0.160947 in epoch 15     acc: 0.9692     val_acc: 0.9419 loss: 0.0799    val_loss: 0.2420
		training - 2
			2:36 PM 10/23/2018 started on GRS1
		training - 5
			2:13 PM 10/23/2018 started on GRS0
		training - 10
			3:37 PM 10/22/2018 started on GRS1
			2:32 PM 10/23/2018 `stopped` after 368 epochs
		training - 100
			1:16 AM 10/19/2018 started on GRS1
			7:19 AM 10/22/2018 done
		training - 1K
			1:10 AM 10/19/2018 done 836 epochs on GRS0
				min_loss: 0.668878 in epoch 1   acc: 0.6689     val_acc: 0.8710 loss: 53613.9370        val_loss: 2576222.9438
				min_val_loss: 0.668878 in epoch 1       acc: 0.6689     val_acc: 0.8710 loss: 53613.9370        val_loss: 2576222.9438
				min_mean_loss: 0.668878 in epoch 1      acc: 0.6689     val_acc: 0.8710 loss: 53613.9370        val_loss: 2576222.9438
				max_acc: 0.675878 in epoch 228  acc: 0.6759     val_acc: 0.9116 loss: 52237.1704        val_loss: 2572194.1463
				max_val_acc: 0.618268 in epoch 836      acc: 0.6183     val_acc: 0.9224 loss: 52237.7285        val_loss: 2562699.4700
				max_mean_acc: 0.618268 in epoch 836     acc: 0.6183     val_acc: 0.9224 loss: 52237.7285        val_loss: 2562699.4700
		training - 5K
			1:15 AM 10/19/2018 resumed at epoch 437  on GRS0
			7:25 AM 10/20/2018 done at 935 epochs on GRS0
	640 - 8
		training
			10:49 AM 9/22/2018 `started` on Z370_0
			2:30 PM 9/23/2018 `resumed` at epoch 287 on Z370_0
			1:27 PM 9/24/2018 `resumed` at epoch 542 on Z370_1
			2:19 PM 9/25/2018 `resumed` at epoch 694 on GRS_1
			9:55 AM 9/27/2018 `done` at 1000 epochs on GRS_1
				Epoch 1000/1000
				Epoch 1/1
				512/512 [==============================] - 377s 737ms/step - loss: 0.0048 - acc: 0.9981 - val_loss: 2.5246 - val_acc: 0.8093
				min_loss: 0.998121 in epoch 1000        acc: 0.9981     val_acc: 0.8093 loss: 0.0048    val_loss: 2.5246
				min_val_loss: 0.997641 in epoch 992     acc: 0.9976     val_acc: 0.9281 loss: 0.0060    val_loss: 0.8481
				min_mean_loss: 0.997243 in epoch 999    acc: 0.9972     val_acc: 0.8816 loss: 0.0074    val_loss: 1.3996
				max_acc: 0.998325 in epoch 969  acc: 0.9983     val_acc: 0.8762 loss: 0.0043    val_loss: 1.5035
				max_val_acc: 0.970238 in epoch 16       acc: 0.9702     val_acc: 0.9431 loss: 0.0780    val_loss: 0.2097
				max_mean_acc: 0.976175 in epoch 27      acc: 0.9762     val_acc: 0.9531 loss: 0.0621    val_loss: 0.2166
				
			
	640 - 16
		training
			8:31 PM 9/22/2018 `done` on Z370 after unknown epochs though probably > 750
				Epoch 276/1000
				Epoch 1/1
				512/512 [==============================] - 377s 737ms/step - loss: 0.0164 - acc: 0.9932 - val_loss: 0.4704 - val_acc: 0.9495
				min_loss: 0.993238 in epoch 276 acc: 0.9932     val_acc: 0.9495 loss: 0.0164    val_loss: 0.4704
				min_val_loss: 0.358094 in epoch 54      acc: 0.9883     val_acc: 0.9396 loss: 0.0336    val_loss: 0.3581
				min_mean_loss: 0.186566 in epoch 4      acc: 0.9948     val_acc: 0.9527 loss: 0.0125    val_loss: 0.3606
				max_acc: 0.996191 in epoch 259  acc: 0.9962     val_acc: 0.8943 loss: 0.0091    val_loss: 1.2635
				max_val_acc: 0.956573 in epoch 115      acc: 0.9955     val_acc: 0.9566 loss: 0.0108    val_loss: 0.3819
				max_mean_acc: 0.976044 in epoch 115     acc: 0.9955     val_acc: 0.9566 loss: 0.0108    val_loss: 0.3819
		training - 100
			7:34 AM 10/20/2018 started on GRS0
			2:11 PM 10/23/2018 `done` at 382 epochs
				Epoch 382/1000
				Epoch 1/1
				1347/1347 [==============================] - 742s 551ms/step - loss: 3839.7384 - acc: 0.6414 - val_loss: 2638929.7750 - val_acc: 0.8207
				min_loss: 3839.618131 in epoch 351      acc: 0.6421     val_acc: 0.8762 loss: 3839.6181 val_loss: 2597664.8900
				min_val_loss: 2529492.811250 in epoch 15        acc: 0.6437     val_acc: 0.9574 loss: 3886.1949 val_loss: 2529492.8112
				min_mean_loss: 1266689.503055 in epoch 15       acc: 0.6437     val_acc: 0.9574 loss: 3886.1949 val_loss: 2529492.8112
				max_acc: 0.647601 in epoch 322  acc: 0.6476     val_acc: 0.8672 loss: 3840.3683 val_loss: 2604635.1463
				max_val_acc: 0.957368 in epoch 15       acc: 0.6437     val_acc: 0.9574 loss: 3886.1949 val_loss: 2529492.8112
				max_mean_acc: 0.800557 in epoch 15      acc: 0.6437     val_acc: 0.9574 loss: 3886.1949 val_loss: 2529492.8112
			
			
	640 - 24
		training
			10:00 AM 9/22/2018 `done` on Z370 after unknown epochs though probably > 750		
		
			
			
	

				
				
			

				
				

					
				

