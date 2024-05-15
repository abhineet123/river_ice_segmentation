clear param
cats = {'bike', 'cars', 'person'};

%detector_types = {'dsift', 'dsift-color'};
detector_types = {'dsift-color'};
reg_params = {{16, 4, 1}, {16, 4, 5}};
atonce = 1;
onlyleaves = 0;

%dictionary_types = {'hikm', 'ikm'};
dictionary_types = {'hikm'};

param.prefix = '/mnt/e0.2/scratch/brian/releasedata/aibloc';
param.db_path = '~/data/graz02/cats' ;
param.db_seg_prefix = '~/data/graz02/segmentations/';

param.ikm_nwords   = 100;
param.dict_hikm_K       = 10;
param.dict_hikm_nleaves = 10000;
param.db_tag = 'gz'; % Graz 2 train 1 test
%param.db_tag = 'gzodd' ; % Graz 1 train 1 test

param.partition_data = 0; % Use 50 images to train the dictionary

% 0 means disabled
aib_types = {0, 5, 40, 200, 400};

for s = 1:1
param.use_segs = s;

for c = 1:length(cats)
    for d = 1:length(detector_types)
        for q = 1:length(dictionary_types)
            for a = 1:length(aib_types)

        param.use_aib = aib_types{a} > 0;
        param.aib_nwords = aib_types{a};
        param.dict_dictionary = dictionary_types{q};
        param.dict_tag = ['_' dictionary_types{q}];

        if strcmp(param.dict_dictionary, 'hikm') && param.dict_hikm_K ~= 10
            param.dict_tag = sprintf('%sK%02d', param.dict_tag, param.dict_hikm_K);
        end

        if strcmp(param.dict_dictionary, 'ikm')
            param.dict_tag = sprintf('_ikm%03d', param.ikm_nwords);
        end

        if strcmp(param.dict_dictionary, 'ikm') && atonce
            param.dict_at_once = 1;
            param.dict_tag = [param.dict_tag '_atonce'];
        else
            param.dict_at_once = 0;
        end

        if strcmp(param.dict_dictionary, 'hikm') && ~onlyleaves
            param.dict_hikm_only_leaves = 0;
            param.dict_tag = [param.dict_tag '_fulltree'];
        else
            param.dict_hikm_only_leaves = 1;
        end

        if param.use_aib && strcmp(param.dict_dictionary, 'ikm')
            continue; % don't want aib ikm config
        end

        if ~param.use_aib && strcmp(param.dict_dictionary, 'hikm')
            continue; % don't want this config
        end

        param.feat_detector = detector_types{d};
        if strcmp(param.feat_detector, 'dsift')
            param.feat_descriptor = 'dsift';
        elseif strcmp(param.feat_detector, 'dsift-color')
            param.feat_descriptor = 'dsift-color';
        else
            param.feat_descriptor = 'simipld';
        end
        param.fg_cat = cats{c};
        param = clearfields(param, {'feat_scales', 'feat_spacing', ...
                    'feat_patchwidth', 'hist_min_sigma', 'dict_at_once'});
        if strcmp(detector_types{d}, 'regular')
            for r = 1:length(reg_params)
                param.feat_patchwidth = reg_params{r}{1};
                param.feat_spacing    = reg_params{r}{2};
                param.feat_scales     = reg_params{r}{3};
                param.feat_tag   = sprintf('regular_%d_%d_%d', ...
                    reg_params{r}{1}, reg_params{r}{2}, reg_params{r}{3});
                param.hist_min_sigma = 0 ;

                aibloc;
            end
        elseif strcmp(detector_types(d), 'dsift') || ...
               strcmp(detector_types(d), 'dsift-color')
            param.feat_tag       = param.feat_detector;
            param.hist_min_sigma = 0 ;
            aibloc;
        else
            param.feat_tag       = param.feat_detector;
            aibloc;
        end

            end
        end
    end
end
end
