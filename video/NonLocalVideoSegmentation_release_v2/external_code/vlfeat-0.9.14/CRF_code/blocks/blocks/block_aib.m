function bk = block_aib(bk)
% BLOCK_AIB Use AIB to compress training histograms
%   This block uses agglomerative information bottleneck, as proposed
%   in Slonim et. al 2000. See also BLOCK_AIBDICT()
%
%   BK = BLOCK_AIB() Initializes the block with the default options.
%
%   BK = BLOCK_AIB(BK) Executes the block with options and inputs BK.
%
%   Required Inputs:
%   
%   db::
%     A database partitioned into testing and training data.
%
%   hist::
%     Histograms to be compressed.
%   
%   Options:
%
%   bk.normalize_hists::
%     Should histograms be normalized before they are accumulated for
%     AIB? Default 1.
%  
%   bk.seg_ids::
%     The segment ids to use for AIB. Default is all training
%     examples.
% 
%   bk.rand_seed::
%     Set the random seed. Default is [], which does not change the
%     seed.
%
%   Fetchable attributes:
%
%   Pcx::
%     The joint class / dictionary probability matrix passed to AIB
%
%   tree:
%     The full AIB merge tree.

% AUTORIGHTS
% Copyright (c) 2009 Brian Fulkerson and Andrea Vedaldi
% Blocks is distributed under the terms of the modified BSD license.
% The full license may be found in LICENSE.

global wrd ;

if nargin == 0
  bk = bkinit('aib', 'hist', 'db') ;
  bk.normalize_hists = 1;
  bk.fetch      = @fetch__ ;
  bk.rand_seed = [];
  bk.seg_ids   = []; % [] means use all of the training data
  return ;
end

% --------------------------------------------------------------------
%                                                              Do work
% --------------------------------------------------------------------

[bk, dirty] = bkbegin(bk) ;
if ~ dirty, return ; end

bkdb   = bkfetch(bk.db.tag) ;
bkhist = bkfetch(bk.hist.tag) ;
db     = bkfetch(bkdb, 'db') ;

if length(bk.seg_ids) > 0
  sel_train = bk.seg_ids;
else
  sel_train = find([db.segs.flag]==db.TRAIN) ;
end
train_seg_ids = [db.segs(sel_train).seg] ;

hist     = bkfetch(bkhist, 'histogram', train_seg_ids(1)) ;
nclasses = length(db.cat_ids);

P = zeros(length(hist), nclasses);

for t=1:length(train_seg_ids);

    segid = train_seg_ids(t);
    class = find(db.segs(sel_train(t)).cat == db.cat_ids);
    hist = bkfetch(bkhist, 'histogram', segid);
    if bk.normalize_hists
        hist = double(hist)/sum(hist+eps);
    end
    P(:,class) = P(:,class) + hist;
end

P = P';

if ~ isempty(bk.rand_seed)
  setrandseeds(bk.rand_seed + mapkeys(t)-1) ;
end

parents = vl_aib(P);

save(fullfile(wrd.prefix, bk.tag, 'data.mat'), 'P', 'parents', '-MAT') ;
bk = bkend(bk) ;

% --------------------------------------------------------------------
function varargout = fetch__(bk, what, varargin)
% --------------------------------------------------------------------
global wrd ;

switch lower(what)
  case 'Pcx'
    data = load(fullfile(wrd.prefix, bk.tag, 'data.mat')) ;
    varargout{1} = data.P;
  case 'tree'
    data = load(fullfile(wrd.prefix, bk.tag, 'data.mat')) ;
    varargout{1} = data.parents;
  otherwise
    error(sprintf('Unknown ''%s''.', what)) ;
end
