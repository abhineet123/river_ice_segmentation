% required parameters
defparam.prefix         = '' ;
defparam.db_path        = '' ;
defparam.seg_prefix     = '' ;
defparam.obj_prefix     = '' ;
defparam.db_type        = '' ;
defparam.db_tag         = '' ;
defparam.fg_cat         = '' ;

% parameters with defaults
defparam.neighbors      = 0;
defparam.qseg_tag       = 'def';
defparam.qseg_sigma     = 2;
defparam.qseg_ratio     = 0.5;
defparam.qseg_tau       = 8;

defparam.dict_tag       = '' ;
defparam.dict_size      = 400 ;
defparam.dict_dictionary= 'ikm' ;

defparam.hists_per_image = 5 ;
defparam.hists_per_cat = [] ;
defparam.classifier     = 'svm';

defparam.feat_tag = 'dsift';
defparam.feat_dsift_size = 12;

defparam.testall = 0;

defparam.crf = 0;
defparam.crf_trainmethod = 'gridsearch';
defparam.crf_traingoal = 'intersection-union';
defparam.crf_restrict = 0;

if ~exist('param'), param = struct ; end
param = vl_override(defparam, param) ;

required = {'prefix', 'db_path', 'db_type', 'db_tag'};
if isfield(param, 'db_type') && ...
  (strcmp(param.db_type, 'graz02') || strcmp(param.db_type, 'graz02odds'))
  required{end+1} = 'seg_prefix';
  required{end+1} = 'obj_prefix';
end
for i = 1:length(required)
  if ~isfield(param, required{i}) || length(getfield(param, required{i}))==0
    error(sprintf('param.%s is required', required{i}));
  end
end

fprintf('*** PARAMETERS ***\n') ;
disp(param) ;

clear global wrd ;
clear wrd ;
global wrd ;

wrd.prefix       = param.prefix  ;
wrd.enable_split = 0 ; 
wrd.bless_all    = 0 ;

% --------------------------------------------------------------------
% construct database
% --------------------------------------------------------------------

ex.db            = block_db ;
ex.db.tag        = ['db@' param.db_tag] ;
ex.db.db_type    = param.db_type ; 
ex.db.db_prefix  = param.db_path ;
ex.db.seg_prefix = param.seg_prefix; 
ex.db.obj_prefix = param.obj_prefix; 

if param.p09test
  ex.db.train = 'trainval';
  ex.db.test  = 'test';
  ex.db.tag = [ex.db.tag 'test'];
end

ex.db = block_db(ex.db) ;

% ------------------------------------------------------------------
% select category/training/testing
% ------------------------------------------------------------------

ex.dbpart           = block_dbpart ;
ex.dbpart.tag       = ['dbpart@' bkver(ex.db) '_' param.fg_cat] ;
ex.dbpart.fg_cat    = param.fg_cat ;
ex.dbpart.db_prefix = ex.db.db_prefix ;
ex.dbpart.db_type   = ex.db.db_type ;

ex.dbpart = bkplug(ex.dbpart, 'db', ex.db.tag) ;
ex.dbpart = block_dbpart(ex.dbpart) ;

% ------------------------------------------------------------------
% segmentations
% ------------------------------------------------------------------
ex.qseg = block_quickseg ;
ex.qseg.tag   = ['qseg@' bkver(ex.db) param.qseg_tag] ;
ex.qseg.sigma = param.qseg_sigma;
ex.qseg.ratio = param.qseg_ratio; 
ex.qseg.tau   = param.qseg_tau;
ex.qseg.split = 16;

ex.qseg = bkplug(ex.qseg, 'db', ex.db.tag) ;
ex.qseg = block_quickseg(ex.qseg) ;

ex.qstat = block_quickstat;
ex.qstat.tag = ['qstat@' bkver(ex.qseg)];

ex.qstat = bkplug(ex.qstat, 'qseg', ex.qseg.tag);
ex.qstat = bkplug(ex.qstat, 'db',   ex.db.tag);
ex.qstat = block_quickstat(ex.qstat);

% ------------------------------------------------------------------
% feature extraction
% ------------------------------------------------------------------
% Extract features at multiple scales

ex.feat            = block_feat ;
ex.feat.detector   = 'dsift' ;
ex.feat.descriptor = 'dsift' ;
ex.feat.tag        = ['feat@' bkver(ex.db) param.feat_tag];
ex.feat.dsift_step  = 1;
ex.feat.dsift_size  = param.feat_dsift_size;
ex.feat.split      = 16 ;

ex.feat = bkplug(ex.feat, 'db', ex.db.tag) ;
ex.feat = block_feat(ex.feat) ;

% ------------------------------------------------------------------
% train dictionary
% ------------------------------------------------------------------

ex.dict             = block_dictionary ;
ex.dict.tag         = ['dict@' bkver(ex.dbpart) bkver(ex.feat) param.dict_tag] ;
ex.dict.dictionary  = param.dict_dictionary ;
ex.dict.nfeats      = 2*50000 ;
ex.dict.rand_seed   = 1 ;
ex.dict.ntrials     = 1 ;
ex.dict.split       = 0 ;

ex.dict.dictionary   = 'ikm' ;
ex.dict.ikm_nwords   = param.dict_size;
ex.dict.ikm_at_once  = 1 ; % otherwise we need to deal with .cat

ex.dict = bkplug(ex.dict, 'db',   ex.dbpart.tag) ;
ex.dict = bkplug(ex.dict, 'feat', ex.feat.tag) ;
ex.dict = block_dictionary(ex.dict) ;  

% ----------------------------------------------------------------
% select a dictionary
% ----------------------------------------------------------------
ex.dictsel           = block_dict_sel ;
ex.dictsel.tag       = ['dictsel@' bkver(ex.dict.tag)] ;
ex.dictsel.selection = 1 ;

ex.dictsel = bkplug(ex.dictsel, 'dict', ex.dict.tag);
ex.dictsel = block_dict_sel(ex.dictsel) ;

% Extract training histograms from segmentation and features

ex.hist_qseg      = block_hist_qseg() ;
if strcmp(param.qseg_tag, 'def')
ex.hist_qseg.tag  = ['histq@' bkver(ex.dictsel.tag) '_m_'];
else
ex.hist_qseg.tag  = ['histq@' bkver(ex.dictsel.tag) '_' param.qseg_tag '_m_'];
end
ex.hist_qseg.min_sigma = 0 ;
ex.hist_qseg.ref_size  = [];

ex.hist_qseg.split = 16;

ex.hist_qseg = bkplug(ex.hist_qseg, 'db',   ex.dbpart.tag) ;
ex.hist_qseg = bkplug(ex.hist_qseg, 'feat', ex.feat.tag) ;
ex.hist_qseg = bkplug(ex.hist_qseg, 'dict', ex.dictsel.tag) ;
ex.hist_qseg = bkplug(ex.hist_qseg, 'qseg', ex.qseg.tag) ;

ex.hist_qseg = block_hist_qseg(ex.hist_qseg);

% ----------------------------------------------------------
% Select the training histograms
% ----------------------------------------------------------

ex.trainsel           = block_train_sel ;
ex.trainsel.hists_per_im = param.hists_per_image;
ex.trainsel.hists_per_cat = param.hists_per_cat;

if length(param.hists_per_cat) > 0
  ex.trainsel.tag       = ['trainsel@' bkver(ex.hist_qseg.tag) ...
    num2str(param.neighbors) '_' num2str(ex.trainsel.hists_per_cat)] ;
else
  ex.trainsel.tag       = ['trainsel@' bkver(ex.hist_qseg.tag) ...
    num2str(param.neighbors) '_' num2str(ex.trainsel.hists_per_im)] ;
end
ex.trainsel.seg_neighbors = param.neighbors;

ex.trainsel = bkplug(ex.trainsel, 'db', ex.dbpart.tag);
ex.trainsel = bkplug(ex.trainsel, 'hist', ex.hist_qseg.tag);
ex.trainsel = block_train_sel(ex.trainsel) ;

switch param.classifier
case 'svm'
  % --------------------------------------------------------------------
  % compute kernel
  % --------------------------------------------------------------------
  [train_ids labels] = bkfetch(ex.trainsel.tag, 'train_ids');
  % train
  ex.ktr             = block_ker() ;
  ex.ktr.row_seg_ids = train_ids ;
  ex.ktr.col_seg_ids = train_ids ;
  ex.ktr.seg_neighbors = param.neighbors;
  ex.ktr.kernel      = 'dchi2' ;
  ex.ktr.tag         = ['ktr@' bkver(ex.trainsel.tag)] ;

  ex.ktr.normalize = 'l1' ;
  ex.ktr.use_segs = 1 ;
  ex.ktr.split = 16 ;
  ex.ktr = bkplug(ex.ktr, 'hist',     ex.hist_qseg.tag) ;
  ex.ktr = bkplug(ex.ktr, 'trainsel', ex.trainsel.tag) ;

  ex.ktr = block_ker(ex.ktr) ;

  % --------------------------------------------------------------------
  % train SVM
  % --------------------------------------------------------------------
  ex.classifier = block_classify_svm;
  ex.classifier.seg_neighbors = param.neighbors;
  ex.classifier.tag = ['cl@' bkver(ex.ktr) '_svm'];
  if length(param.fg_cat) == 0
    ex.classifier.probability = 1 ;
  end

  db = bkfetch(ex.dbpart.tag, 'db');
  bg_ind = find(db.class_ids == 0);
  bg_cat = db.cat_ids(1);
  if length(bg_ind) > 0
    bg_cat = db.cat_ids(bg_ind);
  end
  ex.classifier.bg_cat = bg_cat;
  ex.classifier.svm_cross  = 10 ;
  ex.classifier.svm_gamma  = [] ;
  ex.classifier.svm_balance = 0 ; % Does svm_balance work?
  ex.classifier.svm_rbf    = 1 ; %param.ker_metric ;
  ex.classifier.debug      = 1 ;

  ex.classifier = bkplug(ex.classifier, 'kernel',   ex.ktr.tag);
  ex.classifier = bkplug(ex.classifier, 'hist',     ex.hist_qseg.tag);
  ex.classifier = bkplug(ex.classifier, 'trainsel', ex.trainsel.tag);
  ex.classifier = block_classify_svm(ex.classifier);

otherwise
  error(sprintf('Unknown classifier type: %s', param.classifier));
end

% Test Images

ex.testseg          = block_test_segloc;
ex.testseg.seg_neighbors = param.neighbors;
ex.testseg.tag      = ['testseg@' bkver(ex.classifier)];
ex.testseg.split = 16;
ex.testseg.testall = param.testall;

ex.testseg = bkplug(ex.testseg, 'db',     ex.dbpart.tag);
ex.testseg = bkplug(ex.testseg, 'qseg',   ex.qseg.tag);
ex.testseg = bkplug(ex.testseg, 'hist',   ex.hist_qseg.tag);
ex.testseg = bkplug(ex.testseg, 'classifier', ex.classifier);

ex.testseg = block_test_segloc(ex.testseg);

if param.crf == 1
  ex.traincrf = block_train_crf;
  ex.traincrf.luv = 1;
  ex.traincrf.max_images = 100; % Use at max 100 training images
  ex.traincrf.method = param.crf_trainmethod;
  ex.traincrf.goal   = param.crf_traingoal;
  ex.traincrf.tag = ['traincrf@' bkver(ex.testseg.tag) '_' ex.traincrf.method];
  if ex.traincrf.max_images ~= 100
    ex.traincrf.tag = sprintf('%s%d', ex.traincrf.tag, ex.traincrf.max_images);
  end

  ex.traincrf = bkplug(ex.traincrf, 'db',     ex.dbpart.tag);
  ex.traincrf = bkplug(ex.traincrf, 'segloc', ex.testseg.tag);
  ex.traincrf = bkplug(ex.traincrf, 'histq',  ex.hist_qseg.tag);
  ex.traincrf = bkplug(ex.traincrf, 'qseg',   ex.qseg.tag);
  ex.traincrf = block_train_crf(ex.traincrf);

  ex.testcrf = block_test_segcrf;
  ex.testcrf.restrict = param.crf_restrict;
  ex.testcrf.tag = ['testcrf@' bkver(ex.traincrf.tag)];
  if ex.testcrf.restrict
    ex.testcrf.tag = [ex.testcrf.tag '_restrict'];
  end

  ex.testcrf.split = 16;
  ex.testcrf.bg_cat = bg_cat;

  ex.testcrf = bkplug(ex.testcrf, 'db', ex.dbpart.tag);
  ex.testcrf = bkplug(ex.testcrf, 'segloc', ex.testseg.tag);
  ex.testcrf = bkplug(ex.testcrf, 'qseg', ex.qseg.tag);
  ex.testcrf = bkplug(ex.testcrf, 'traincrf', ex.traincrf.tag);
  ex.testcrf = block_test_segcrf(ex.testcrf);
  ex.testseg_old = ex.testseg;
  ex.testseg = ex.testcrf;
end

if strcmp(param.db_type, 'pascal07') || ...
  (strcmp(param.db_type, 'pascal09') && ~param.p09test)
  ex.vismulti = block_vismulti;
  ex.vismulti.tag = ['vismulti@' bkver(ex.testseg.tag)];

  ex.vismulti = bkplug(ex.vismulti, 'db', ex.dbpart.tag);
  ex.vismulti = bkplug(ex.vismulti, 'prediction', ex.testseg.tag);
  ex.vismulti = block_vismulti(ex.vismulti);

% graz
elseif strcmp(param.db_type, 'graz02odds') || strcmp(param.db_type, 'graz02')

  ex.visloc  = block_visloc;
  %ex.visloc.tag  = ['visloc@' bkver(ex.testseg.tag) '_goodsegs'] ;
  %ex.visloc.seg_prefix = '~/data/graz02/ig02segs/';
  ex.visloc.tag  = ['visloc@' bkver(ex.testseg.tag)] ;
  ex.visloc.seg_prefix = '~/data/graz02/segmentations/';
  ex.visloc.seg_ext    = 'png';
  ex.visloc.cat_ids    = {{'bike', 1}, {'cars', 2}, {'person', 3}, {'none', 0}};
  ex.visloc.fg_cat     = param.fg_cat;
  ex.visloc.split  = 0;

  ex.visloc = bkplug(ex.visloc, 'db',         ex.dbpart.tag) ;
  ex.visloc = bkplug(ex.visloc, 'prediction', ex.testseg.tag) ;
  ex.visloc = block_visloc(ex.visloc) ;

else
  fprintf('quickrec: Skipping ground truth visualization\n');
end

if strcmp(param.db_type, 'pascal09')
  ex.visvoc = block_visvoc;
  ex.visvoc.tag = ['visvoc@' bkver(ex.testseg.tag)];

  if param.p09test
    ex.visvoc.challenge = 'test';
  else
    ex.visvoc.challenge = 'val';
  end

  ex.visvoc = bkplug(ex.visvoc, 'db', ex.dbpart.tag);
  ex.visvoc = bkplug(ex.visvoc, 'prediction', ex.testseg.tag);
  ex.visvoc = block_visvoc(ex.visvoc);
end

