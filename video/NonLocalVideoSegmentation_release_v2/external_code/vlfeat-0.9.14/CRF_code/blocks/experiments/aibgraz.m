
% Required parameters
defparam.prefix         = '' ;
defparam.db_path        = '' ;

defparam.db_type        = 'graz02' ;
defparam.db_tag         = 'gz' ;
defparam.fg_cat         = 'cars';
defparam.feat_tag       = 'basms' ;
defparam.feat_detector  = 'ipld';
defparam.feat_descriptor = 'simipld';
defparam.feat_min_sigma = 2.5 ;
defparam.feat_rescale   = 6 ;
defparam.feat_ref_size  = [] ;
defparam.dict_tag       = 'bas' ;
defparam.dict_size      = 200 ;
defparam.dict_type      = 'ikm' ;
defparam.dict_K         = 10 ;
defparam.aib_nwords     = [];

if ~exist('param'), param = struct ; end
param = vl_override(defparam, param) ;

if length(param.prefix) == 0, error('prefix must be specified'); end
if length(param.db_path) == 0, error('path to the database must be specified'); end

fprintf('*** PARAMETERS ***\n') ;
disp(param) ;

clear global wrd ;
clear wrd ;
global wrd ;

wrd.prefix       = param.prefix  ;
wrd.enable_split = 0 ;
wrd.bless_all = 0;

% --------------------------------------------------------------------
% construct database
% --------------------------------------------------------------------
ex.db           = block_db ;
ex.db.tag       = ['db@' param.db_tag] ;
ex.db.db_type   = param.db_type ; 
ex.db.db_prefix = param.db_path ;

ex.db = block_db(ex.db) ;

% ------------------------------------------------------------------
% feature extraction
% ------------------------------------------------------------------
ex.feat            = block_feat ;
ex.feat.min_sigma  = param.feat_min_sigma ;
ex.feat.rescale    = param.feat_rescale ;
ex.feat.ref_size   = param.feat_ref_size ;
ex.feat.max_num    = +inf ;
ex.feat.detector   = param.feat_detector ;
if isfield(param, 'feat_detector_params')
  ex.feat.detector_params = param.feat_detector_params;
end
ex.feat.descriptor = param.feat_descriptor ;
ex.feat.split      = 30 ;
ex.feat.rand_seed  = 1 ;

ex.feat.tag       = ...
  ['feat@' bkver(ex.db) '_' param.feat_tag ] ;

ex.feat = bkplug(ex.feat, 'db', ex.db.tag) ;
ex.feat = block_feat(ex.feat) ;

% ------------------------------------------------------------------
% select category/training/testing
% ------------------------------------------------------------------

ex.dbpart           = block_dbpart ;
ex.dbpart.fg_cat    = param.fg_cat ;
ex.dbpart.tag       = ['dbpart@' bkver(ex.db) '_' ex.dbpart.fg_cat ] ;
ex.dbpart.db_prefix = ex.db.db_prefix ;
ex.dbpart.db_type   = ex.db.db_type ;

ex.dbpart = bkplug(ex.dbpart, 'db', ex.db.tag) ;
ex.dbpart = block_dbpart(ex.dbpart) ;

% ------------------------------------------------------------------
% train dictionary
% ------------------------------------------------------------------

nwords = param.dict_size ;

switch param.dict_type
  
  case 'ikm'
    ex.dict             = block_dictionary ;
    ex.dict.tag         = ['dict@' bkver(ex.dbpart) param.feat_tag sprintf('_ikm%d', nwords)] ;
    ex.dict.dictionary  = 'ikm';
    ex.dict.nfeats      = 50000 ;
    ex.dict.ikm_nwords  = nwords ;
    ex.dict.rand_seed   = 1 ;
    ex.dict.ntrials     = 5 ;
    ex.dict.all_at_once = 0 ;
    ex.dict.split       = ex.dict.ntrials ;
    
    ex.dict = bkplug(ex.dict, 'db',   ex.dbpart.tag) ;
    ex.dict = bkplug(ex.dict, 'feat', ex.feat.tag) ;
    ex.dict = block_dictionary(ex.dict) ;  
    
  case {'hikm', 'aib'}
    ex.dict             = block_dictionary ;
    ex.dict.method      = 'hikm';
    ex.dict.tag         = ['dict@' bkver(ex.dbpart) param.feat_tag sprintf('_hikm_%d', nwords)] ;
    ex.dict.nfeats      = 150000 ;
    ex.dict.dictionary  = 'hikm';
    ex.dict.hikm_K      = param.dict_K;
    ex.dict.hikm_nleaves = param.dict_size ;
    ex.dict.rand_seed   = 1 ;
    ex.dict.ntrials     = 5 ;
    ex.dict.all_at_once = 1 ;
    ex.dict.split       = 5 ;
    
    ex.dict = bkplug(ex.dict, 'db',    ex.dbpart.tag) ;
    ex.dict = bkplug(ex.dict, 'feat',  ex.feat.tag) ;
    ex.dict = block_dictionary(ex.dict) ;  
    
  otherwise
    error('Dictionary %s unknown\n', param.dict_type);
end

% ------------------------------------------------------------------
% repeat: histograms, learn svm, test svm, visualize
% ------------------------------------------------------------------

for trial = 1:5
  
  % ----------------------------------------------------------------
  % select a dictionary
  % ----------------------------------------------------------------
  ex.dictsel           = block_dict_sel ;
  ex.dictsel.tag       = ['dictsel@' ...
                bkver(ex.dict.tag) sprintf('_tri%d', trial)] ;
  ex.dictsel.selection = trial ;
  
  ex.dictsel = bkplug(ex.dictsel, 'dict', ex.dict.tag);
  ex.dictsel = block_dict_sel(ex.dictsel) ;

  ex.hist        = block_hist ;
  ex.hist.tag    = ['hist@' bkver(ex.dictsel.tag)] ;
  ex.hist.split  = 30 ;

  ex.hist = bkplug(ex.hist, 'db',   ex.dbpart.tag) ;
  ex.hist = bkplug(ex.hist, 'feat', ex.feat.tag) ;
  ex.hist = bkplug(ex.hist, 'dict', ex.dictsel.tag) ;
  ex.hist = block_hist(ex.hist) ;

  if strcmp(param.dict_type, 'aib')
    ex.aib            = block_aib();
    ex.aib.tag        = ['aib@' bkver(ex.hist.tag)];

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
    ex.hist.tag       = ['hist@' bkver(ex.aibdict.tag)] ;
    
    ex.hist.split     = 30 ;
    ex.hist = bkplug(ex.hist, 'db',   ex.dbpart.tag) ;
    ex.hist = bkplug(ex.hist, 'feat', ex.feat.tag) ;
    ex.hist = bkplug(ex.hist, 'dict', ex.aibdict.tag) ;
    ex.hist = block_hist(ex.hist) ;
  end

  % --------------------------------------------------------------------
  % compute kernel
  % --------------------------------------------------------------------
  db    = bkfetch(ex.dbpart, 'db') ;
  seltr = find([db.segs.flag] == db.TRAIN) ;
  selts = find([db.segs.flag] == db.TEST) ;
  idstr = [db.segs(seltr).seg] ;
  idsts = [db.segs(selts).seg] ;

  % train
  ex.ktr             = block_ker ;
  ex.ktr.tag         = ['ktr@' bkver(ex.hist.tag)] ;
  ex.ktr.row_seg_ids = idstr ;
  ex.ktr.col_seg_ids = idstr ;
  ex.ktr.kernel      = 'dchi2' ;
  ex.ktr.normalize   = 'l1' ;
  ex.ktr.split       = 14 ;
  ex.ktr.use_segs    = 0;

  ex.ktr = bkplug(ex.ktr, 'hist', ex.hist.tag) ;
  ex.ktr = block_ker(ex.ktr) ;

  % test
  ex.kts             = ex.ktr ;
  ex.kts.tag         = ['kts@' bkver(ex.ktr.tag)] ;
  ex.kts.row_seg_ids = idsts ;
  ex.kts.col_seg_ids = idstr ;
  ex.kts.split       = 30 ;

  ex.kts = bkplug(ex.kts, 'hist', ex.hist.tag) ;
  ex.kts = block_ker(ex.kts) ;

  % --------------------------------------------------------------------
  % train SVM
  % --------------------------------------------------------------------

  ex.train            = block_train_svm ;
  ex.train.tag        = ['train@' bkver(ex.ktr.tag)] ;
  ex.train.svm_cross  = 10 ;
  ex.train.svm_gamma  = [] ;
  ex.train.svm_rbf    = 1 ;
  ex.train.debug      = 1 ;

  ex.train = bkplug(ex.train, 'db',      ex.dbpart.tag) ;
  ex.train = bkplug(ex.train, 'kernel',  ex.ktr.tag) ;
  ex.train = block_train_svm(ex.train) ;

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

end % next dictionary
