% AIBLOC Driver
    
defparam.prefix         = '' ;
defparam.db_path        = '' ;
defparam.db_seg_prefix  = '' ;

defparam.db_type        = 'graz02' ;
defparam.db_tag         = 'gz' ;
defparam.fg_cat         = 'bike';

defparam.feat_tag       = '' ;
defparam.feat_detector  = 'ipld';
defparam.feat_descriptor= 'simipld' ;
defparam.feat_min_sigma = 0 ;
defparam.feat_rescale   = 6 ;
defparam.feat_ref_size  = [] ;
defparam.feat_patchwidth = 16;
defparam.feat_spacing   = 4;
defparam.feat_scales    = 1;

defparam.hist_min_sigma = 2.5 ;

defparam.ker_tag        = '_chi2' ;
defparam.ker_type       = 'dchi2' ;
defparam.ker_normalize  = 'l1' ;
defparam.ker_metric     = 1 ;

defparam.dict_ntrials   = 1 ;
defparam.dict_tag       = '' ;
defparam.dict_size      = 200 ;
defparam.dict_dictionary= 'ikm' ;
defparam.dict_ikm_nwords   = 200;
defparam.dict_hikm_K       = 10;
defparam.dict_hikm_nleaves = 10000;
defparam.dict_hikm_only_leaves = 1;
defparam.dict_at_once   = 0;

defparam.use_aib        = 0;
defparam.aib_nwords     = 40;

defparam.use_segs       = 0;

% Number of images used to train dictionary. Remainder will be used to train
% the svm if 0, partitioning is disabled
defparam.partition_data     = 0;

if ~exist('param'), param = struct ; end
param = vl_override(defparam, param) ;

if length(param.prefix) == 0, error('prefix must be specified'); end
if length(param.db_path) == 0, error('path to the database must be specified'); end
if length(param.db_seg_prefix) == 0, error('path to the segmentations must be specified'); end

fprintf('*** PARAMETERS ***\n') ;
disp(param) ;

clear global wrd ;
clear wrd ;
global wrd ;

wrd.prefix       = param.prefix  ;
wrd.enable_split = 1 ;
wrd.bless_all    = 0 ;
wrd.pretend      = 0 ;

cat_ids    = {{'bike', 1}, {'cars', 2}, {'person', 3}, {'none', 0}};
for i = 1:length(cat_ids)
    if strcmp(cat_ids{i}{1}, param.fg_cat)
        param.fg_id = cat_ids{i}{2};
    end
end

% --------------------------------------------------------------------
% construct database
% --------------------------------------------------------------------
clear ex;
ex.db           = block_db ;
ex.db.tag       = ['db@' param.db_tag] ;
ex.db.db_type   = param.db_type ; 
ex.db.db_prefix = param.db_path ;

ex.db = block_db(ex.db) ;

% ------------------------------------------------------------------
% feature extraction
% ------------------------------------------------------------------

ex.feat            = block_feat ;
ex.feat.tag        = ['feat@' param.db_tag param.feat_tag] ;
ex.feat.min_sigma  = param.feat_min_sigma ;
ex.feat.rescale    = param.feat_rescale ;
ex.feat.ref_size   = param.feat_ref_size ;
ex.feat.max_num    = +inf ;
ex.feat.detector   = param.feat_detector ;
ex.feat.descriptor = param.feat_descriptor ;
ex.feat.spacing    = param.feat_spacing ;
ex.feat.scales     = param.feat_scales ;
ex.feat.patchwidth = param.feat_patchwidth ;
if strcmp(param.feat_detector, 'dsift') || ...
   strcmp(param.feat_detector, 'dsift-color')
    ex.feat.dsift_size = 4;
    ex.feat.dsift_step = 4;
    ex.feat.dsift_minnorm = 0.015;
end
ex.feat.split      = 30 ;
ex.feat.rand_seed  = 1 ;

ex.feat = bkplug(ex.feat, 'db', ex.db.tag) ;
ex.feat = block_feat(ex.feat) ;

% ------------------------------------------------------------------
% select category/training/testing
% ------------------------------------------------------------------

ex.dbpart           = block_dbpart ;
ex.dbpart.tag       = ['dbpart@' param.db_tag '_' param.fg_cat] ;
ex.dbpart.fg_cat    = param.fg_cat ;
ex.dbpart.db_prefix = ex.db.db_prefix ;
ex.dbpart.db_type   = ex.db.db_type ;

ex.dbpart = bkplug(ex.dbpart, 'db', ex.db.tag) ;
ex.dbpart = block_dbpart(ex.dbpart) ;


% ------------------------------------------------------------------
% partition training data
% ------------------------------------------------------------------

if param.partition_data
    rand('twister', 1);
    db    = bkfetch(ex.dbpart, 'db') ;
    seg_ids = find([db.segs.flag] == db.TRAIN) ;
    dict_seg_ids = [];
    for i = 1:length(db.cat_ids)
        cat_ids = find([db.segs.flag] == db.TRAIN & ... 
                       [db.segs.cat] == db.cat_ids(i));
        perm = randperm(length(cat_ids));
        dict_seg_ids = [dict_seg_ids cat_ids(perm(1:param.partition_data))];
    end
    train_seg_ids = setdiff(seg_ids, dict_seg_ids);
end

% ------------------------------------------------------------------
% train dictionary
% ------------------------------------------------------------------

ex.dict             = block_dictionary ;
if param.partition_data
    param.dict_tag = [param.dict_tag '_part'];
end
ex.dict.tag         = ['dict@' bkver(ex.dbpart) param.feat_tag param.dict_tag] ;
ex.dict.dictionary  = param.dict_dictionary ;
ex.dict.nfeats      = 2*50000 ;
ex.dict.rand_seed   = 1 ;
ex.dict.ntrials     = param.dict_ntrials ;
ex.dict.split       = ex.dict.ntrials ;

if param.partition_data
    ex.dict.seg_ids = dict_seg_ids ;
end

ex.dict.ikm_nwords   = param.dict_ikm_nwords ;
ex.dict.hikm_K       = param.dict_hikm_K;
ex.dict.hikm_nleaves = param.dict_hikm_nleaves;
ex.dict.hikm_only_leaves = param.dict_hikm_only_leaves;

ex.dict.ikm_at_once  = param.dict_at_once ;

ex.dict = bkplug(ex.dict, 'db',   ex.dbpart.tag) ;
ex.dict = bkplug(ex.dict, 'feat', ex.feat.tag) ;
ex.dict = block_dictionary(ex.dict) ;  

% ------------------------------------------------------------------
% repeat: kernels, learn svm, test svm, visualize
% ------------------------------------------------------------------

for trial = 1:param.dict_ntrials
  
  % ----------------------------------------------------------------
  % select a dictionary
  % ----------------------------------------------------------------
  ex.dictsel           = block_dict_sel ;
  ex.dictsel.tag       = ['dictsel@' bkver(ex.dict.tag) ...
                    sprintf('_tri%d', trial)] ;
  ex.dictsel.selection = trial ;
  
  ex.dictsel = bkplug(ex.dictsel, 'dict', ex.dict.tag);
  ex.dictsel = block_dict_sel(ex.dictsel) ;

  % ------------------------------------------------------------------
  % compute histograms
  % ------------------------------------------------------------------

  ex.hist           = block_hist() ;
  if param.use_segs
    ex.hist.tag       = ['hist@' bkver(ex.dictsel.tag) '_seg'] ;
    ex.hist.seg_prefix = param.db_seg_prefix;
    ex.hist.seg_ext    = 'png';
    ex.hist.fg_id      = param.fg_id;
    ex.hist.fg_cat     = param.fg_cat;
  else
    ex.hist.tag       = ['hist@' bkver(ex.dictsel.tag)] ;
  end
  ex.hist.split     = 30 ;
  ex.hist.min_sigma = param.hist_min_sigma ;
  ex.hist = bkplug(ex.hist, 'db',   ex.dbpart.tag) ;
  ex.hist = bkplug(ex.hist, 'feat', ex.feat.tag) ;
  ex.hist = bkplug(ex.hist, 'dict', ex.dictsel.tag) ;
  ex.hist = block_hist(ex.hist) ;

  if param.use_aib
      ex.aib            = block_aib();
      ex.aib.tag        = ['aib@' bkver(ex.hist.tag)];

      if param.partition_data
          ex.aib.seg_ids = dict_seg_ids;
      end
      ex.aib            = bkplug(ex.aib, 'db',   ex.dbpart.tag) ;
      ex.aib            = bkplug(ex.aib, 'hist', ex.hist.tag) ;
      ex.aib            = block_aib(ex.aib);

      ex.aibdict        = block_aibdict();
      ex.aibdict.nwords = param.aib_nwords;
      ex.aibdict.tag    = sprintf('aibdict@%s_aib%d', bkver(ex.aib), ...
                                  param.aib_nwords);

      ex.aibdict        = bkplug(ex.aibdict, 'aib',  ex.aib.tag) ;
      ex.aibdict        = bkplug(ex.aibdict, 'dict', ex.dictsel.tag) ;
      ex.aibdict        = block_aibdict(ex.aibdict) ;

      ex.hist_noaib     = ex.hist;
      ex.hist           = block_hist() ;
      if param.use_segs
        ex.hist.tag       = ['hist@' bkver(ex.aibdict.tag) '_seg'] ;
        ex.hist.seg_prefix = param.db_seg_prefix;
        ex.hist.seg_ext    = 'png';
        ex.hist.fg_id      = param.fg_id;
        ex.hist.fg_cat     = param.fg_cat;
      else
        ex.hist.tag       = ['hist@' bkver(ex.aibdict.tag)] ;
      end
      ex.hist.split     = 30 ;
      ex.hist.min_sigma = param.hist_min_sigma ;
      ex.hist = bkplug(ex.hist, 'db',   ex.dbpart.tag) ;
      ex.hist = bkplug(ex.hist, 'feat', ex.feat.tag) ;
      ex.hist = bkplug(ex.hist, 'dict', ex.aibdict.tag) ;
      ex.hist = block_hist(ex.hist) ;
  end

  % --------------------------------------------------------------------
  % compute kernel
  % --------------------------------------------------------------------

  ker_block = @block_ker ;
  
  if ~ wrd.pretend
    db    = bkfetch(ex.dbpart, 'db') ;
    seltr = find([db.segs.flag] == db.TRAIN) ;
    if param.partition_data
        seltr = train_seg_ids;
    else
        seltr = find([db.segs.flag] == db.TRAIN) ;
    end
    selts = find([db.segs.flag] == db.TEST) ;
    idstr = [db.segs(seltr).seg] ;
    idsts = [db.segs(selts).seg] ;
  else
    idstr = [] ;
    idsts = []  ;
  end
    
  % train
  ex.ktr             = ker_block() ;
  ex.ktr.tag         = ['ktr@' bkver(ex.hist.tag) param.ker_tag] ;
  ex.ktr.row_seg_ids = idstr ;
  ex.ktr.col_seg_ids = idstr ;
  ex.ktr.kernel      = param.ker_type ;
  ex.ktr.metric      = param.ker_metric ;
  ex.ktr.split       = 20 ;
  
  ex.ktr.normalize = param.ker_normalize;
  ex.ktr = bkplug(ex.ktr, 'hist', ex.hist.tag) ;
      
  ex.ktr = ker_block(ex.ktr) ;

  % --------------------------------------------------------------------
  % train SVM
  % --------------------------------------------------------------------

  ex.train            = block_train_svm ;
  ex.train.tag        = ['train@' bkver(ex.ktr.tag)] ;
  ex.train.svm_cross  = 10 ;
  ex.train.svm_gamma  = [] ;
  ex.train.svm_rbf    = param.ker_metric ;
  ex.train.debug      = 1 ;

  if param.partition_data
      ex.train.seg_ids = train_seg_ids;
  end

  ex.train = bkplug(ex.train, 'db',      ex.dbpart.tag) ;
  ex.train = bkplug(ex.train, 'kernel',  ex.ktr.tag) ;
  ex.train = block_train_svm(ex.train) ;

  % --------------------------------------------------------------------
  % test SVM
  % --------------------------------------------------------------------

  %TODO: Integrate mask images into training data extraction
  ex.loctest = block_test_bruteloc ;
  ex.loctest.split = 30 ;
  ex.loctest.classifier = 'svm';
  %ex.loctest.classifier = 'info';
  %ex.loctest.classifier = 'nn';
  if param.partition_data
      ex.loctest.seg_ids = train_seg_ids;
  end

  ex.loctest.tag = ['loctest@' bkver(ex.train.tag) '_' ex.loctest.classifier];

  ex.loctest = bkplug(ex.loctest, 'db',     ex.dbpart.tag) ;
  ex.loctest = bkplug(ex.loctest, 'feat',   ex.feat.tag) ;
  ex.loctest = bkplug(ex.loctest, 'hist',   ex.hist.tag) ;
  ex.loctest = bkplug(ex.loctest, 'svm',    ex.train.tag) ;
  ex.loctest = bkplug(ex.loctest, 'kernel', ex.ktr.tag) ;
  
  ex.loctest = block_test_bruteloc(ex.loctest) ;

  ex.visloc  = block_visloc;
  %ex.visloc.tag  = ['visloc@' bkver(ex.loctest.tag) '_fgonly'] ;
  %ex.visloc.fgonly = 1;
  ex.visloc.tag  = ['visloc@' bkver(ex.loctest.tag)] ;
  ex.visloc.fgonly = 0;
  ex.visloc.seg_prefix = param.db_seg_prefix;
  ex.visloc.seg_ext    = 'png';
  ex.visloc.cat_ids    = {{'bike', 1}, {'cars', 2}, {'person', 3}, {'none', 0}};
  ex.visloc.fg_cat     = param.fg_cat;
  ex.visloc.split  = 0;

  ex.visloc = bkplug(ex.visloc, 'db',         ex.dbpart.tag) ;
  ex.visloc = bkplug(ex.visloc, 'prediction', ex.loctest.tag) ;
  ex.visloc = block_visloc(ex.visloc) ;

  % Test bag of features on whole images
  if 1

  % test
  ex.kts             = ex.ktr ;
  ex.kts.tag         = ['kts@' bkver(ex.ktr.tag)] ;
  ex.kts.row_seg_ids = idsts ;
  ex.kts.col_seg_ids = idstr ;
  ex.kts.split       = 30 ;

  ex.kts = bkplug(ex.kts, 'hist', ex.hist.tag) ;
    
  ex.kts = ker_block(ex.kts) ;

  % --------------------------------------------------------------------
  % test SVM
  % --------------------------------------------------------------------

  ex.test            = block_test_svm ;
  ex.test.tag        = ['test@' bkver(ex.train.tag)] ;

  ex.test = bkplug(ex.test, 'svm',    ex.train.tag) ;
  ex.test = bkplug(ex.test, 'kernel', ex.kts.tag) ;
  ex.test = block_test_svm(ex.test) ;

  % --------------------------------------------------------------------
  % results
  % --------------------------------------------------------------------

  ex.vis             = block_vis ;
  ex.vis.tag         = ['vis@' bkver(ex.test.tag)] ;

  ex.vis = bkplug(ex.vis, 'db',         ex.dbpart.tag) ;
  ex.vis = bkplug(ex.vis, 'prediction', ex.test.tag) ;
  ex.vis = block_vis(ex.vis) ;


  end

end % next dictionary
