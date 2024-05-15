function bk = block_train_crf(bk, varargin)
% BLOCK_TRAIN_CRF Train a superpixel CRF
%   Train a superpixel CRF from validation images. If validation
%   images are not available, uses training images.
%
%   BK = BLOCK_TRAIN_CRF() Initializes the block with the default
%   options.
%
%   BK = BLOCK_TRAIN_CRF(BK) Executes the block with options and
%   inputs BK.
%
%   Required inputs:
%
%   db::
%     The database.
%
%   segloc::
%     The unary potentials, in the form output by BLOCK_TEST_SEGLOC()
%
%   histq::
%     Superpixel histograms.
%
%   qseg::
%     Quick shift superpixels.
%
%   Options:
%   
%   bk.method::
%     The training method. Valid methods are: static, gridsearch.
%     Default 'gridsearch'.
%
%   bk.goal::
%     The training goal to optimize. Valid goals are: meanacc,
%     intersection-union. Default 'intersection-union'.
%
%   bk.luv::
%     Should the color difference be in the LUV space? Default 1.
%   
%   bk.max_images::
%     If there are more than max_images in the validation or training
%     data, select max_images and use those. Default 1000.
%
%   Fetchable attributes:
%
%   params::
%     The learned parameters of the CRF.
%
%   paramspace::
%     A structure containing the parameters tried at each iteration.

% AUTORIGHTS
% Copyright (c) 2009 Brian Fulkerson and Andrea Vedaldi
% Blocks is distributed under the terms of the modified BSD license.
% The full license may be found in LICENSE.

global wrd ;

if nargin == 0
  bk = bkinit('train_crf', 'db', 'segloc', 'histq', 'qseg') ;
  bk.fetch = @fetch__ ;

  bk.method = 'gridsearch';
  bk.goal   = 'intersection-union';
  bk.luv = 1;
  bk.max_images = 1000;

  return ;
end

% --------------------------------------------------------------------
%                                                    Check/load inputs
% --------------------------------------------------------------------

[bk, dirty] = bkbegin(bk) ;
if ~ dirty, return ; end

db      = bkfetch(bk.db.tag,     'db') ;

icat_map = zeros(max(db.cat_ids), 1);
for i = 1:length(db.cat_ids)
  icat_map(db.cat_ids(i)) = i;
end

tload = cputime;

segclasses = {};
unaries = {};
pairwise = {};
gtlabels = {};
counts = {};
ncats = length(db.cat_ids);
labelcost = ones(ncats) - eye(ncats);
images = find([db.segs.flag]==db.VALIDATION);
if length(images) == 0
  fprintf('block_train_crf: WARNING, no validation images found\n');
  fprintf('block_train_crf:   using training images\n');
  images = find([db.segs.flag]==db.TRAIN);
end

if bk.max_images < length(images)
  images = images(randperm(length(images)));
  images = images(1:bk.max_images);
end
fprintf('block_train_crf: Considering %d images\n', length(images));

for im = 1:length(images)
  if strcmp(bk.method, 'static'), continue; end

  sid = db.segs(images(im)).seg;
  [H labels] = bkfetch(bk.histq.tag, 'seghistograms', sid);
  segs = bkfetch(bk.qseg.tag, 'segs', sid);
  map = bkfetch(bk.qseg.tag, 'segmap', sid);
  [pred dec] = bkfetch(bk.segloc.tag, 'segtest', sid);
 
  gtlabels{im} = labels;
  nonzero = find(labels); 
  gtlabels{im}(nonzero) = icat_map(labels(nonzero));
  counts{im} = [segs.count];
  
  segclass = icat_map(pred) - 1;
  segprob = cat(1,dec{:})';
  segprob = abs(segprob);

  if size(segprob, 1) == 1 % Graz
    segprob(segprob>1) = 1;
    segprob  = [segprob/2 + .5; segprob/2 + .5];
    %segprob  = [segprob; segprob];
    ind = find(segclass==0);
    segprob(2,ind) = 1 - segprob(2,ind);
    ind = find(segclass==1);
    segprob(1,ind) = 1 - segprob(1,ind);
  end

  unaries{im}{1} = -log(segprob+eps);

  %%%%%% PAIRWISE %%%%%%%
  % build graph
  colors = [segs.color];
  if bk.luv == 1
    C = squeeze(vl_xyz2luv(vl_rgb2xyz(reshape(colors', [size(colors,2), 1, size(colors,1)]))))';
    colors = C;
  end

  nsegs = length(segs);
  
  D = vl_alldist2(colors,colors);
  Ds = sum(D(:));
  Ds = Ds / ((nsegs*nsegs - nsegs));
  beta = 1/(2*Ds);
  clear D;

  pairwise = zeros(nsegs, nsegs);
  for i = 1:nsegs
    c = segclass(i);
    for n = 1:length(segs(i).adj)
      j = segs(i).adj(n);
      pairwise(i,j) = 1/(1+norm(colors(:,i) - colors(:,j)));
      %pairwise(i,j) = exp(-beta*norm(colors(:,i) - colors(:,j)));
    end
  end
  segclasses{im} = segclass;
  pairwises{im}{1} = sparse(pairwise);
  pairwises{im}{2} = sparse(boundarylen(map, length(segs)));
  clear pairwise;


  fprintf('block_train_crf: Loaded data %.0f%%\r', 100*im/length(images));
end
fprintf('block_train_crf: Data loaded %.0f seconds\n', cputime - tload);

% --------------------------------------------------------------------
%                                                      Do optimization
% --------------------------------------------------------------------

t0 = cputime;
switch bk.method
case 'static'
  params = struct();
  params.luv = bk.luv;
  %params.l_edge = 1.5264;
  params.l_edge = bk.l_edge;
  save(fullfile(wrd.prefix, bk.tag, 'params.mat'), 'params', '-MAT') ;

case 'gridsearch'
  switch bk.goal
  case 'intersection-union'
    intunion = 1;
  case 'meanacc'
    intunion = 0;
  otherwise
    error('Unrecognized bk.goal');
  end
  fprintf('block_train_crf: Training goal %s\n', bk.goal);

  iter = 1;
  for edge = linspace(0.01, 5, 20);
    %for offset = logspace(log10(0.0001), log10(10), 10);
        paramspace(iter).l_edge = edge;
        %paramspace(iter).l_offset = offset;
        paramspace(iter).luv     = bk.luv;
        iter = iter + 1;
    %end
  end 
  
  ncats = length(db.cat_ids);

  % Calculate meanacc of the classifier
  conf = zeros(ncats);
  for im = 1:length(images)
    gtl = gtlabels{im};
    c = counts{im};
    labels = segclasses{im} + 1;
    confim = zeros(ncats);
    for i = 1:length(db.cat_ids)
      ind = find(gtl == icat_map(db.cat_ids(i)));
      confim(:,i) = vl_binsum(confim(:,i), c(ind), labels(ind));
    end
    conf = conf + confim;
  end
  conf = conf ./ repmat(sum(conf + eps), ncats, 1);
  cl_conf = conf;
  cl_meanacc = 100*mean(diag(conf));
  fprintf('block_train_crf: Classifier mean accuracy: %.2f%% for %d cats\n', ...
    cl_meanacc, ncats);

  % Explore the grid
  edge = linspace(0.01, 20, 10);
  paramspace = struct();
  iter = 1;
  while true
    acc = zeros(length(edge),1);
    for z = 1:length(edge)
      param = struct();
      param.l_edge = edge(z);

      [meanacc E] = evalparam(segclasses, unaries, pairwises, labelcost, ...
                              gtlabels, counts, ncats, images, db, icat_map, ...
                              param, intunion);

      acc(z) = meanacc;
    end
    [maxacc best] = max(acc);
    pick = max(find(acc == maxacc));
    step = edge(2) - edge(1);
    bestedge = edge(pick);

    paramspace(iter).acc = acc;
    paramspace(iter).edge = edge;
    paramspace(iter).bestedge = bestedge;

    edge = linspace(max(0.01, edge(pick) - 2*step), edge(pick) + 2*step, 11);

    fprintf('block_train_crf: Iter %d %.2f%% at %f min %f max %f (%.0fs)\n', iter, ...
      maxacc, bestedge, edge(1), edge(end), cputime - t0); 
    
    if max(acc) - min(acc) < 0.2 % .2 percent
      break
    end

    iter = iter + 1;
    if iter > 10
      break
    end
  end

  params = struct();
  params.luv = bk.luv;
  params.l_edge = paramspace(end).bestedge;

  fprintf('block_train_crf: Finished gridsearch in %.0fs\n', cputime - t0);
  
  disp(params)
  save(fullfile(wrd.prefix, bk.tag, 'paramspace.mat'), 'paramspace', '-MAT') ;
  save(fullfile(wrd.prefix, bk.tag, 'params.mat'), 'params', 'cl_meanacc', 'cl_conf', '-MAT') ;

otherwise
  error('Unknown optimization type');
end

bk = bkend(bk) ;

% --------------------------------------------------------------------
function varargout = fetch__(bk, what, varargin)
% --------------------------------------------------------------------

global wrd ;

switch lower(what)
    
  case 'params'
    path = fullfile(wrd.prefix, bk.tag, 'params.mat') ;    
    data = load(path, '-MAT') ;
    varargout{1} = data.params ;
 
  case 'paramspace'
    path = fullfile(wrd.prefix, bk.tag, 'paramspace.mat') ;    
    data = load(path, '-MAT') ;
    varargout{1} = data.paramspace ;
  otherwise
    error(sprintf('Unknown ''%s''.', what)) ;
end

%%%%%%%%%%%%%%%%%%% gridsearch support function
function [meanacc E] = evalparam(segclasses, unaries, pairwises, labelcost, ...
                                 gtlabels, counts, ncats, images, db, ...
                                 icat_map, params, intunion)
% evaluate a parameter
conf = zeros(ncats);
E = 0;
for im = 1:length(images)
  %pairwise = spfun(@(x) paramspace(iter).l_edge*x + paramspace(iter).l_offset, ...
  %                 pairwises{im});
  pairwise = params.l_edge*pairwises{im}{1}.*pairwises{im}{2};
  [labels Eim Eimbefore] = crfcore(segclasses{im}, unaries{im}{1}, ...
    pairwise, labelcost);

    % Evaluate parameter set
    labels = labels + 1;
    gtl = gtlabels{im};
    c = counts{im};
    confim = zeros(ncats);
    for i = 1:length(db.cat_ids)
      ind = find(gtl == icat_map(db.cat_ids(i)));
      confim(:,i) = vl_binsum(confim(:,i), c(ind), labels(ind));
    end
    conf = conf + confim;
    E = E + Eim;
end

% optimize correct pixels
% paramspace(iter).meanacc = sum(diag(conf));

% optimize mean accuracy
if intunion
 intu = zeros(ncats, 1);
  for j = 1:ncats
    gtj = sum(conf(j,:));
    resj = sum(conf(:,j));
    gtjresj = sum(conf(j,j));
    intu(j) = 100*gtjresj/(gtj+resj-gtjresj);
  end
  meanacc = mean(intu);
else
  conf = conf ./ repmat(sum(conf + eps), ncats, 1);
  meanacc = 100*mean(diag(conf));
end
