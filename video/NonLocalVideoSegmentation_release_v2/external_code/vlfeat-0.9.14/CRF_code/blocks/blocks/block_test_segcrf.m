function bk = block_test_segcrf(bk, varargin)
% BLOCK_TEST_SEGCRF Classify test segments with a CRF
%   Superpixel CRF, as proposed in Fulkerson et. al 2009.
%
%   BK = BLOCK_TEST_SEGCRF() Initializes the block with the default
%   options.
%
%   BK = BLOCK_TEST_SEGCRF(BK) Executes the block with options and
%   inputs BK.
%
%   Required inputs:
%
%   db::
%     The database.
%
%   qseg::
%     The quick shift segmentations.
%
%   segloc::
%     The unary potentials, in the form output by BLOCK_TEST_SEGLOC()
%
%   traincrf::
%     Parameters for the crf, from BLOCK_TRAIN_CRF()
%
%   Options:
%
%   bk.rand_seed::
%     Set the random seeds before proceeding. Default of [] does not
%     change the seeds.
%
%   bk.restrict::
%     Restrict the possible solutions of the CRF to include only those
%     which have co-occurred in the training data. Default 0. 
%
%   Fetchable attributes:
%
%   test::
%     Classification result (images). Returns [class confidence] for
%     required input: seg_id.
%
%   segtest::
%     Classification result (superpixels). Returns [class confidence]
%     for required input: seg_id. class is a vector Nx1 where N is the
%     number of superpixels. confidence is a matrix Nx1 cell array,
%     where each cell is a Cx1 vector expressing the confidence in
%     each of C classes.

% AUTORIGHTS
% Copyright (c) 2009 Brian Fulkerson and Andrea Vedaldi
% Blocks is distributed under the terms of the modified BSD license.
% The full license may be found in LICENSE.

global wrd ;

if nargin == 0
  bk           = bkinit('test_segcrf', 'db', 'segloc', 'qseg', 'traincrf') ;
  bk.fetch     = @fetch__ ;
  bk.rand_seed = [] ;
  bk.restrict  = 0;
  return ;
end

% --------------------------------------------------------------------
%                                                    Check/load inputs
% --------------------------------------------------------------------

[bk, dirty] = bkbegin(bk) ;
if ~ dirty, return ; end

db       = bkfetch(bk.db.tag, 'db') ;
params   = bkfetch(bk.traincrf.tag, 'params');

% --------------------------------------------------------------------
%                                                       Do computation
% --------------------------------------------------------------------
if bk.restrict
  tsegs = find([db.segs.flag]==db.TRAIN);
  nclasses = length(db.cat_ids);
  cooccur = zeros(nclasses, nclasses);
  for im = 1:length(tsegs)
    labels = unique(db.segs(tsegs(im)).obj_ids);
    for i = 1:length(labels)-1
      for j = i+1:length(labels)
        cooccur(labels(i),labels(j)) = cooccur(labels(i),labels(j)) + 1;
      end
    end
  end
  cooccur = cooccur + cooccur' - diag(diag(cooccur));
  context_cats = {};
  z = 1;
  for i = 1:nclasses
    row = unique([i find(cooccur(i,:) > 0)]);
    if length(row) < nclasses
      context_cats{z} = row;
      z = z + 1;
    end
  end
end

%if bk.testall
%  keys = 1:length(db.segs) ;
%else
  keys = find([db.segs.flag]==db.TEST) ;
%end
[reduce, mapkeys] = bksplit(bk, keys, varargin{:}) ;
ensuredir(fullfile(wrd.prefix, bk.tag, 'data')) ;

icat_map = zeros(max(db.cat_ids),1);
for i=1:length(db.cat_ids)
  icat_map(db.cat_ids(i)) = i;
end

for t=1:length(mapkeys)

  if ~ isempty(bk.rand_seed)
    setrandseeds(bk.rand_seed + mapkeys(t)-1) ;
  end
    
  seg_id = db.segs(mapkeys(t)).seg ;
  fprintf('block_test_segcrf: seg_id: %d\n', seg_id); 

  I    = bkfetch(bk.qseg.tag, 'segimage', seg_id);
  % fetch segment histograms
  segs = bkfetch(bk.qseg.tag, 'segs', seg_id);
  map = bkfetch(bk.qseg.tag, 'segmap', seg_id);
  boundary = boundarylen(map, length(segs));
  [seglabel segdec] = bkfetch(bk.segloc.tag, 'segtest', seg_id);
  segprob = cat(1, segdec{:})';
  seglabel = icat_map(seglabel) - 1;
  if size(segprob,1) == 1 % have confidence, convert to prob (graz)
    segprob = abs(segprob);
    segprob(segprob>1) = 1;
    segprob = [segprob/2 + .5; segprob/2 + .5];
    %segprob = [segprob; segprob];
    ind = find(seglabel == 0);
    segprob(2,ind) = 1-segprob(2,ind); 
    ind = find(seglabel == 1);
    segprob(1,ind) = 1-segprob(1,ind);
  end
  unary = -log(segprob);
  
  % include counts
  %scount = repmat([segs.count], size(unary,1), 1);
  %unary = unary.*scount;

  % need to possibly renumber labels
  t0 = cputime;
  [labels E Ebefore] = crfprocess(segs, seglabel, unary, params, boundary);
  if bk.restrict
    context_labels = zeros(size(labels,1), length(context_cats));
    context_E = zeros(length(context_cats),1);
    for zz = 1:length(context_cats)
      % TODO: Assumes last cat is the bg
      labelmap = ones(size(unary,1), 1)*length(context_cats{zz}); 
      labelmap(context_cats{zz}) = 1:length(context_cats{zz});
      conlabel = labelmap(seglabel+1) - 1;
      conunary = unary(context_cats{zz},:);
      conparams = params;
      if isfield(conparams, 'pairwise_w')
        for p = 1:length(conparams.pairwise_w)
          conparams.pairwise_w{p} = conparams.pairwise_w{p}(context_cats{zz},:);
          conparams.pairwise_w{p} = conparams.pairwise_w{p}(:,context_cats{zz});
        end
      end
      [CL CE] = crfprocess(segs, conlabel, conunary, conparams, boundary);
      context_labels(:,zz) = context_cats{zz}(CL+1);
      context_E(zz) = CE;
    end
    [E ind] = min(context_E);
    labels = context_labels(:,ind) - 1;
  end

  if length(segdec{1}) == 1 % use new confidence
    ind = find(seglabel ~= labels); % changed pixels i
    for i = 1:length(labels)
      if seglabel(i) == labels(i)
        % offset labels which stayed the same by a 0.5 confidence
        segdec{i}(1) = abs(segdec{i}(1)) + 0.5;
      else
        % the prob will always be <= 0.5
        segdec{i}(1) = segprob(labels(i)+1,i);
      end
    end
    fprintf('block_test_segcrf: flipped %d/%d labels\n', length(ind), length(labels)); 
  end
  labels = labels + 1;
  newlabels = db.cat_ids(labels);

  nsegs = length(segs);
  pred  = zeros(nsegs,1);
  dec   = {};

  segtime = cputime - t0;

  class = ones(size(I,1), size(I,2))*bk.bg_cat;
  confidence = zeros(size(I,1), size(I,2),length(segdec{1}));
  for j = 1:size(confidence, 3)
    c2 = zeros(size(class));
    for i = 1:nsegs
      if j == 1, class(segs(i).ind) = newlabels(i); end
      c2(segs(i).ind) = segdec{i}(j);
    end
    confidence(:,:,j) = c2;
  end
  confidence = abs(confidence);

  
  classifytime = cputime - t0;

  % save results
  n = fullfile('data', sprintf('%05d', seg_id)) ;
  save(fullfile(wrd.prefix, bk.tag, [n '.test']), 'class', 'confidence', '-MAT');
  save(fullfile(wrd.prefix, bk.tag, [n '.segtest']), 'newlabels', 'segdec', '-MAT');
  save(fullfile(wrd.prefix, bk.tag, [n '.time']), 'classifytime', 'segtime', '-MAT');

  fprintf('block_test_segcrf: %3.0f%% completed\n', ...
          t / length(mapkeys) * 100) ;   
end

if reduce
  bk = bkend(bk) ;
end

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

  case 'segtest'
    i = varargin{1} ;
    path = fullfile(wrd.prefix, bk.tag, 'data', sprintf('%05d.segtest', i)) ;
    data = load(path, '-MAT') ;
    varargout{1} = data.newlabels;
    varargout{2} = data.segdec ;

  otherwise
    error(sprintf('Unknown ''%s''.', what)) ;
end


