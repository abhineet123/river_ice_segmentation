function bk = block_train_sel(bk)
% BLOCK_TRAIN_SEL Select training superpixels
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
%   hist::
%     Segment histograms.
%
%   Options:
%
%   bk.hists_per_im::
%     The number of histograms to select from each training image. If
%     hists_per_cat is set, this has no effect.  Default 50
%
%   bk.hists_per_cat::
%     The number of histograms to select per category. Default [] uses
%     hist_per_im.
%
%   bk.seg_neighbors::
%     The size of the superpixel neighborhood. Default 0.
%
%   bk.rand_seed::
%     Set the random seed before proceeding. Default [] does not
%     change the random seeds.
%
%   Fetchable attributes:
%
%   train_ids::
%     Returns [train_ids labels]. train_ids is a Nx2 vector, where the
%     first column denotes the seg_id and the second column denotes
%     the superpixel.

% AUTORIGHTS
% Copyright (c) 2009 Brian Fulkerson and Andrea Vedaldi
% Blocks is distributed under the terms of the modified BSD license.
% The full license may be found in LICENSE.

global wrd;

if nargin == 0
  bk = bkinit('train_sel', 'db', 'hist');
  bk.fetch        = @fetch__;
  bk.hists_per_im = 50;
  bk.hists_per_cat = [];
  bk.seg_neighbors = 0;
  bk.rand_seed    = [];
  return;
end

[bk, dirty] = bkbegin(bk);
if ~ dirty, return ; end

db    = bkfetch(bk.db.tag, 'db') ;

if ~ isempty(bk.rand_seed)
  setrandseeds(bk.rand_seed) ;
end

catsegs = {};
seltr = find([db.segs.flag] == db.TRAIN) ;
for c = 1:length(db.cat_ids)
  catseg = [];
  for i = 1:length(seltr)
    if length(find(db.segs(seltr(i)).obj_ids == db.cat_ids(c))) > 0
      catseg = [catseg db.segs(seltr(i)).seg];
    end
  end
  catsegs{c} = catseg;
end
nsegs = zeros(length(catsegs), 1);
for i = 1:length(nsegs), nsegs(i) = length(catsegs{i}); end

hists_per_im = zeros(length(catsegs), 1);
if length(bk.hists_per_cat) == 0
  hists_per_im = ones(length(catsegs),1)*bk.hists_per_im;
else
  for i = 1:length(nsegs) 
    hists_per_im(i) = floor(bk.hists_per_cat / nsegs(i));
  end
end

train_ids = [];
labels = [];
total = length([catsegs{:}]);
zz = 1;
for c = 1:length(catsegs)
  ngot = 0;
  for i = 1:length(catsegs{c})
    [hists histlabels] = bkfetch(bk.hist.tag, 'seghistograms', ...
                                 catsegs{c}(i), bk.seg_neighbors);
    counts = sum(hists, 2);
    ind = find(counts > 1 & histlabels == db.cat_ids(c));
    ind = ind(randperm(length(ind)));

    % if we are behind, catch up
    if length(bk.hists_per_cat) == 0
      nexpected = ngot;
    else
      nexpected = round((i-1)*bk.hists_per_cat/nsegs(c));
    end

    for j = 1:min(hists_per_im(c) + nexpected - ngot, length(ind))
      train_ids = [train_ids; catsegs{c}(i) ind(j)];
      labels = [labels; db.cat_ids(c)];
      ngot = ngot + 1;
    end

    fprintf('block_train_sel: %3.0f%% done\r', zz*100 / total);
    zz = zz + 1;
  end
end

save(fullfile(wrd.prefix, bk.tag, 'train_ids.mat'), 'train_ids', 'labels', '-MAT');

bk = bkend(bk);

end

% --------------------------------------------------------------------
function varargout = fetch__(bk, what, varargin)
% --------------------------------------------------------------------
global wrd ;

switch lower(what)
  case 'train_ids'
    data = load(fullfile(wrd.prefix, bk.tag, 'train_ids.mat'));
    varargout{1} = data.train_ids;
    varargout{2} = data.labels;
  otherwise
    error(sprintf('Unknown ''%s''.', what)) ;
end
end

