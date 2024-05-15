function bk = block_test_bruteloc(bk, varargin)
% BLOCK_TEST_BRUTELOC Test brute-force localization
%   Brute-force localization, as proposed in Fulkerson et. al 2008.
%
%   BK = BLOCK_TEST_BRUTELOC() Initializes the block with the default
%   options.
%
%   BK = BLOCK_TEST_BRUTELOC(BK) Executes the block with options and
%   inputs BK.
%
%   Required inputs:
%
%   db::
%     The database.
%
%   feat::
%     Features extracted on the database
%
%   hist::
%     Histograms extracted on the database.
%
%   svm::
%     A trained SVM.
%
%   kernel::
%     The kernel the SVM was trained with.
%
%   Options:
%
%   bk.rand_seed::
%     Default [] does not change the random seeds.
%
%   bk.scaleby::
%     Downsample the image by this factor before classification.
%     Default 4.
%
%   bk.windowsize::
%     Half-width of the window to use in hisogram computation.
%     Expressed in terms of the non-downsampled image.  Default 80.
%
%   bk.classifier::
%     The type of classifier to use. Valid options: 'svm' or 'nn'.
%     Default 'svm'.
%
%   bk.seg_ids::
%     The segment ids to classify. Default [] uses all testing images
%     in the database.
%
%   Fetchable attributes:
%
%   test::
%     The classification result. Returns [class confidence] for
%     required input: seg_id.


% AUTORIGHTS
% Copyright (c) 2009 Brian Fulkerson and Andrea Vedaldi
% Blocks is distributed under the terms of the modified BSD license.
% The full license may be found in LICENSE.

global wrd ;

if nargin == 0
  bk           = bkinit('test_bruteloc', 'db', 'feat', 'hist', 'svm', 'kernel') ;
  bk.fetch     = @fetch__ ;
  bk.rand_seed = [] ;
  bk.scaleby   = 4 ;
  bk.windowsize = 80 ;
  bk.classifier = 'svm';
  bk.seg_ids    = [];
  return ;
end

% --------------------------------------------------------------------
%                                                    Check/load inputs
% --------------------------------------------------------------------

[bk, dirty] = bkbegin(bk) ;
if ~ dirty, return ; end

db     = bkfetch(bk.db.tag, 'db') ;
dbcfg  = bkfetch(bk.db.tag) ;
bkfeat = bkfetch(bk.feat.tag) ;
bkhist = bkfetch(bk.hist.tag) ;
svmcfg = bkfetch(bk.svm.tag) ;
svm    = bkfetch(bk.svm.tag, 'svm') ;

kercfg = bkfetch(bk.kernel.tag) ;

[ker_func, norm_func] = kernel_function(kercfg.kernel, kercfg.normalize);

% --------------------------------------------------------------------
%                                                       Do computation
% --------------------------------------------------------------------

%keys = 1:length(db.segs) ;
keys = find([db.segs.flag]==db.TEST) ;
[reduce, mapkeys] = bksplit(bk, keys, varargin{:}) ;
ensuredir(fullfile(wrd.prefix, bk.tag, 'data')) ;

% load training data for kernel computation
if length(bk.seg_ids) > 0
    sel_train = bk.seg_ids;
else
    sel_train = find([db.segs.flag]==db.TRAIN) ;
end

train_seg_ids = [db.segs(sel_train).seg] ;
use_svm = 0;
if ~use_svm && strcmp(bk.classifier, 'svm')
    train_seg_ids = train_seg_ids(svm.libsvm_cl.SVs);
end

h = bkfetch(bkhist, 'histogram', train_seg_ids(1));
nclusters = length(h);

trainingdata = zeros(nclusters, length(train_seg_ids));
for i = 1:length(train_seg_ids)
    trainingdata(:, i) = bkfetch(bkhist, 'histogram', train_seg_ids(i)); 
    trainingdata(:, i) = trainingdata(:, i) / norm_func(trainingdata(:, i));
end

% another option is to add all the histograms then divide by the total per
% class
% Probably most kosher to partition data for this
classweights = zeros(nclusters,1);
classes       = [db.segs(sel_train).cat] ;
for i = 1:length(train_seg_ids)
    if classes(i) == svm.labels(1) % fg
        classweights = classweights + trainingdata(:,i);
    else
        classweights = classweights - trainingdata(:,i);
    end
end


for t=1:length(mapkeys)

  if ~ isempty(bk.rand_seed)
    setrandseeds(bk.rand_seed + mapkeys(t)-1) ;
  end
    
  seg_id = db.segs(mapkeys(t)).seg ;
  fprintf('block_test_bruteloc: seg_id: %d\n', seg_id); 
 
  % fetch the features and their locations
  f = bkfetch(bkfeat, 'frames', seg_id) ;
  [w sel] = bkfetch(bkhist, 'words', seg_id); 

  f = f(:,sel);

  I = bkfetch(dbcfg, 'image', seg_id);
  imsize = size(I);
  imsize = imsize(1:2);
  
  t0 = cputime;

  % make integral image signature
  intsig      = integralsig(w, f, nclusters, imsize, bk.scaleby);

  windowsize = round(bk.windowsize/bk.scaleby);
  % classify image
  class       = zeros(size(intsig,1), size(intsig,2));
  confidence  = zeros(size(intsig,1), size(intsig,2));
  %sigs = zeros(nclusters, size(intsig,1)*size(intsig,2));
  j = 1;
  for col = 1:size(intsig,2)
    for row = 1:size(intsig,1)

        h = integralhist(intsig, row, col, windowsize);
        if sum(h) == 0
            class(j) = svm.labels(1); confidence(j) = 0;
            j = j + 1;
            continue;
        end
        h = h / norm_func(h);

        %sigs(:,j) = h;

        switch bk.classifier
        case 'nn'
            class(j) = pred; confidence(j) = dec;

        case 'info'
            confidence(j) = sum(h .* classweights);
            if confidence(j) > 0
                class(j) = svm.labels(1);
            else
                class(j) = svm.labels(2);
            end

        case 'svm'
            K = ker_func(h, trainingdata);

            if ~use_svm
                if svm.rbf
                    K = exp(-svm.gamma * K);
                end
                dec = K*svm.libsvm_cl.sv_coef - svm.libsvm_cl.rho;
                if dec > 0
                    pred = svm.labels(1);
                else
                    pred = svm.labels(2);
                end
            else
                [pred, dec]  = svmkerneltest(svm, K) ;
            end
            class(j) = pred; confidence(j) = dec;
        otherwise
            error('block_test_bruteloc: Unknown classifier type: %s', ...
            bk.classifier);
        end
        j = j + 1;
    end
  end
  confidence = abs(confidence);
  classifytime = cputime - t0;

  % save results
  n = fullfile('data', sprintf('%05d', seg_id)) ;
  save(fullfile(wrd.prefix, bk.tag, [n '.test']), 'class', 'confidence', '-MAT');
  save(fullfile(wrd.prefix, bk.tag, [n '.time']), 'classifytime', '-MAT');

  fprintf('block_test_bruteloc: %3.0f%% completed\n', ...
          t / length(mapkeys) * 100) ;    
end

if reduce
  bk = bkend(bk) ;
end

%%%%%%%%%%%%
function  inthist = integralhist(intim, row, col, windowsize)
%%%%%%%%%%%%

r = min(size(intim,1), max(1, ([row row] + [-1 1]*windowsize)));
c = min(size(intim,2), max(1, ([col col] + [-1 1]*windowsize)));

inthist = getintegralsig(intim, r(1),c(1),r(2),c(2));

%%%%%%%%%%%%
function  intsig = integralsig(w, f, nclusters, sz, scaleby)
%%%%%%%%%%%%

rows = int16(sz(1)/scaleby);
cols = int16(sz(2)/scaleby);

signature = zeros(rows, cols, nclusters);

for i = 1:length(w)
    row = int16(floor((f(2,i)-1)/scaleby)+1);
    col = int16(floor((f(1,i)-1)/scaleby)+1);
    signature(row,col,w(i)) =  signature(row,col,w(i)) + 1;
end

intsig = vl_imintegral(signature);

%% per dictionary element with vl_binsum?
%for i = 1:nclusters
%    sel = find(w==i);
%    ind = sub2ind(imsize, f(2,sel), f(1,sel));
%    intsig(:,:,i) = vl_binsum(intsig(:,:,i), ind, ones(size(ind)));
%end

% --------------------------------------------------------------------
function varargout = fetch__(bk, what, varargin)
% --------------------------------------------------------------------
global wrd ;

switch lower(what)
  
  case 'test'
    i = varargin{1} ;
    path = fullfile(wrd.prefix, bk.tag, 'data', sprintf('%05d.test', i)) ;
    data = load(path, '-MAT') ;
    varargout{1} = data.class ;
    varargout{2} = data.confidence ;

  otherwise
    error(sprintf('Unknown ''%s''.', what)) ;
end


