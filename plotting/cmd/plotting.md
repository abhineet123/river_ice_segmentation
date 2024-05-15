# mAP plotting pipeline
1. coco_to_xml to convert json to mAP compatible csv format with optional nms sweeping
2. mAP to generate results summary again with optional nms sweeping
    C:\Datasets\mAP
    Z:\UofA\Acamp\acamp_code\mAP\log\seg
    Y:\UofA\617\Project\617_proj_code\plotting\log\seg
3. create list of files to concat in
    Y:\UofA\617\Project\617_proj_code\plotting\cmd\concat
4. concat_metrics to generate plot compatible data
    Y:\UofA\617\Project\617_proj_code\plotting\log
5. run plot_summary with
    rec_prec_mode = 2;
    thresh_mode = 3;
6. plots go to:
    Z:\UofA\PhD\Reports\plots