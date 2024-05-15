clear param;

param.prefix = '/mnt/e0.2/scratch/brian/releasedata/graz';
param.db_path = '~/data/graz02/cats' ;

type = 'ipldsift'
switch type
case 'vlfeat'
param.feat_tag        = 'vlfeat';
param.feat_detector   = 'sift';
param.feat_descriptor = 'simipld';
case 'ipld'
param.feat_tag        = 'basms' ;
param.feat_detector   = 'ipld';
param.feat_descriptor = 'simipld';
case 'ipldsift'
param.feat_tag        = 'ipldsift' ;
param.feat_detector   = 'ipld';
param.feat_descriptor = 'sift';
case 'vlparams'
param.feat_tag        = 'vlparams' ;
param.feat_detector   = 'sift';
param.feat_descriptor = 'simipld';
param.feat_min_sigma  = 1.5;
param.feat_detector_params = {'peakthresh', 0, 'edgethresh', 10};
otherwise
  error('no idea');
end

cats = {'bike', 'cars', 'person'};
methods = {'ikm', 'hikm', 'aib'};
%methods = {'ikm', 'hikm'};
cuts = {20, 200, 1000};
for c = 1:length(cats)
    param.fg_cat = cats{c};

    for m = 1:length(methods)
        param.dict_type = methods{m};

        switch methods{m}
        case 'aib'
            for t = 1:length(cuts)
                param.dict_K   = 20;
                param.dict_size = 8000;
                param.aib_nwords = cuts{t};
                aibgraz;
            end
        case 'hikm'
            param.aib_nwords = [];
            param.dict_K   = 20;
            param.dict_size = 8000;
            aibgraz;
        case 'ikm'
            param.aib_nwords = [];
            param.dict_size = 200;
            aibgraz;
        otherwise
            error('Unknown dict method %s', methods{m});
        end
    end
end
