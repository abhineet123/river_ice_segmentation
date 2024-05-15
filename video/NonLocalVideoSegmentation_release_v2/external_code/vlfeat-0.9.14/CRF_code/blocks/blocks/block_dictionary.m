function bk = block_dictionary(bk, varargin)
% BLOCK_DICTIONARY Construct a dictionary
%   This block learns a dictionary from a database and a set of
%   features.
%
%   BK = BLOCK_DICTIONARY() Initializes the block with the default
%   options.
%
%   BK = BLOCK_DICTIONARY(BK) Executes the block with options and
%   inputs BK.
%
%   Required Inputs:
%
%   db::
%     The partitioned database.
%
%   feat::
%     Features extracted on the partitioned database.
%
%   Options:
%   
%   bk.dictionary::
%     The type of dictionary to create. Supported types: 
%     ikm: Integer k-means
%     hikm: Hierarchical Integer k-means
%
%   bk.nfeats::
%     The maximum number of features to sample for training. Default
%     1000 features.
%
%   bk.rand_seed::
%     Set the random seed before proceeding. Default [] does not
%     change the random seeds.
%
%   bk.ntrials::
%     The number of trials to run.
%
%   bk.split::
%     How many processes to use. Default 0.
%
%   bk.seg_ids::
%     The segment ids to use for training. The default of [] will
%     select all of the data marked as training in the database.
%
%   IKM options:
%   
%   bk.ikm_nwords::
%     Number of visual words generated for each category. If
%     IKM_AT_ONCE is activated, this parameter is instead the total
%     number of visual words.
%
%   bk.ikm_at_once::
%     Train a single dictionary, instead of one for each category.
%
%   Hierarchical IKM options:
%
%   bk.hikm_k::
%     The branching factor of the HIKM tree.
%
%   bk.hikm_nleaves::
%     The number of leaf nodes in the HIKM tree.
%
%   bk.hikm_only_leaves::
%     Push works as if only the leaves of the tree existed.
%
%   Block functions:
%
%   push::
%     [WORDS,HIST,SEL] = PUSH(DICT, DATA) pushes the data through the
%     dictionary. sel indexes which data items correspond to which
%     words.

% AUTORIGHTS
% Copyright (c) 2009 Brian Fulkerson and Andrea Vedaldi
% Blocks is distributed under the terms of the modified BSD license.
% The full license may be found in LICENSE.

global wrd ;

if nargin == 0
  bk = bkinit('dictionary', 'db', 'feat') ;
  bk.fetch        = @fetch__ ;
  bk.push         = [] ;

  bk.dictionary   = 'ikm' ;
  bk.nfeats       = 1e3 ;  
  bk.rand_seed    = [] ;
  bk.ntrials      = 1 ;
  bk.split        = 0 ;

  bk.seg_ids      = [];

  bk.ikm_nwords       = 100 ;
  bk.ikm_at_once      = 0 ;
  bk.hikm_K           = 10 ;
  bk.hikm_nleaves     = 100 ;
  bk.hikm_only_leaves = 0 ;
  return ;
end

% --------------------------------------------------------------------
%                                                      Virutal methods
% --------------------------------------------------------------------

% ....................................................................
function [w,h,sel] = ikm_push__(dict, d)
% ....................................................................
w = vl_ikmeanspush(d, dict) ;
if nargout > 1
  h = vl_ikmeanshist(size(dict,2), w) ;
end
sel = 1:length(w);
end

% ....................................................................
function [w,h,sel] = hikm_push__(dict, d)
% ....................................................................
w = vl_hikmeanspush(dict, d) ;
ndescriptors = size(d, 2);

if bk.hikm_only_leaves
  % convert PATH to leaves to leaf ids
  tmp = zeros(1,size(w,2)) ;
  for d=1:dict.depth
    tmp = tmp * dict.K ;
    tmp = tmp + double(w(d,:)) - 1 ;        
  end
  w = tmp + 1 ;  
  if nargout > 1
    h = vl_ikmeanshist(dict.K^dict.depth, w) ;
  end
  sel = 1:ndescriptors;
else
  wtmp = w;
  nodes = zeros(dict.depth+1, size(w,2));
  nodes(1,:) = ones(1, size(nodes, 2)); % Root node
  for d = 1:dict.depth
    if d > 1
      wtmp(1:d-1,:) = wtmp(1:d-1,:)*dict.K;
      nodes(1+d, :) = sum(wtmp(1:d,:)) + 1; % +1 for the root node
    else
      nodes(1+d, :) = wtmp(1, :) + 1;
    end
  end

  if nargout > 1
    h = vl_hikmeanshist(dict, w) ;
  end
  if nargout > 2
    sel = repmat(1:ndescriptors', dict.depth+1, 1);
    sel = sel(:);
  end
  w = nodes(:); % flatten each node of the tree into a list
end
end

% Nested function used as methods needs to be defined here because
% during init the bk stuff will not be initialized.
switch lower(bk.dictionary) 
  case 'ikm'
    bk.push = @ikm_push__ ;
    
  case 'hikm'
    bk.push = @hikm_push__ ;
    
  otherwise
    error('Uknown dictionary type.') ;
end

% --------------------------------------------------------------------
%                                                    Check/load inputs
% --------------------------------------------------------------------

[bk, dirty] = bkbegin(bk) ;
if ~ dirty, return ; end

db_cfg   = bkfetch(bk.db.tag) ;
db       = bkfetch(bk.db.tag, 'db') ;

% --------------------------------------------------------------------
%                                                       Do computation
% --------------------------------------------------------------------

keys = 1:bk.ntrials ;
[reduce, mapkeys] = bksplit(bk, keys, varargin{:}) ;

for t=1:length(mapkeys)
  
  if ~ isempty(bk.rand_seed)
    setrandseeds(bk.rand_seed + mapkeys(t)-1) ;
  end
  
  dict = [] ;
   
  if strcmp(lower(bk.dictionary), 'ikm') & ~ bk.ikm_at_once

    %
    % Learn one dict. per category
    %
    
    for c=1:length(db.cat_ids)
      if length(bk.seg_ids > 0)
        sel_train = intersect(find([db.segs.cat] == db.cat_ids(c)), ...
                              bk.seg_ids);
      else
        sel_train = ...
          find([db.segs.cat]  == db.cat_ids(c) & ...
               [db.segs.flag] == db.TRAIN) ;
      end
      fprintf('block_dictionary: processing category:  ''%s''\n',...
              db.cat_names{c}) ;
      descr = collect_features(bk, db, sel_train) ;    
      dict  = [dict learn(bk, descr)] ;
    end
    
  else        
    
    %
    % Learn one dict. for all categories
    %

    if length(bk.seg_ids) > 0
      sel_train = bk.seg_ids;
    else
      sel_train = [];
      if isfield(db.segs(1), 'obj_ids')
        for i = 1:length(db.segs)
          if db.segs(i).flag == db.TRAIN && ...
             length(intersect(db.cat_ids, db.segs(i).obj_ids)) > 0
            sel_train = [sel_train i];
          end
        end
      else
        for c = 1:length(db.cat_ids)
          sel_train = [sel_train ...
                       find([db.segs.cat]==db.cat_ids(c) & ...
                            [db.segs.flag]==db.TRAIN)] ;
        end
      end
    end
    fprintf('block_dictionary: processing all cats at once\n') ;
    descr = collect_features(bk, db, sel_train) ;    
    dict  = learn(bk, descr) ;
    
  end
  
  % ------------------------------------------------------------------
  %                                                          Save back
  % ------------------------------------------------------------------  
  save(fullfile(wrd.prefix, bk.tag, ...
                sprintf('dict-%02d.mat',mapkeys(t))), 'dict') ;
  fprintf('block_dictionary: dictionary saved.\n') ;
  
end % next dictiontary to learn

if reduce
  bk = bkend(bk) ;
end

end %end function

% --------------------------------------------------------------------
function descr = collect_features(bk, db, sel_train)
% --------------------------------------------------------------------

feat_cfg  = bkfetch(bk.feat.tag) ;
ntrain    = length(sel_train) ;
d         = bkfetch(feat_cfg, 'descriptors', db.segs(sel_train(1)).seg) ;    
fdims     = size(d,1);
descr     = zeros(fdims, bk.nfeats, 'uint8') ;

info = whos('descr') ;
fprintf('block_dictionary: === Collecting Features === \n') ; 
fprintf('block_dictionary: num training features: %d\n',   bk.nfeats) ;
fprintf('block_dictionary: num training images:   %d\n',      ntrain) ;
fprintf('block_dictionary: buffer size:           %.3g MB\n', info.bytes/1024^2) ;

%
% Scan sel_train to count available features
%

nfeats = zeros(1,ntrain) ;
for j = 1:ntrain  
  seg_id    = db.segs(sel_train(j)).seg ;    
  d         = bkfetch(feat_cfg, 'descriptors', seg_id) ;    
  nfeats(j) = size(d, 2) ;
  
  fprintf('block_dictionary: scanned %.1f %%\r', 100*j/ntrain) ;
end 
fprintf('\n') ;

% total number of features in sel_train
av_nfeats = sum(nfeats) ;

%
% Randomly extract nfeats from av_nfeats available features
%

which_feats = randperm(av_nfeats) ;
which_feats = sort(which_feats(1:min(bk.nfeats,length(which_feats)))) ;

if length(which_feats) < bk.nfeats
  fprintf('block_dictionary: found only %.3f K features\n', ...
          length(which_feats) / 1000) ;
  fprintf('block_dictionary: using %.2%% of the requested features.\n', ...
          length(which_feats) / bk.nfeats * 100) ;
  descr = zeros(fdims, length(which_feats), 'uint8') ;
end

% which_feat indexes n enumeration of all the features of the
% category (across multiple images). We scan the images in the
% dataset retrieving the selected features.

bg = 1 ;
curr_seg = 1 ;
while 1
  
  % get name of feature descriptors file
  seg_id = db.segs(sel_train(curr_seg)).seg ;    
  d      = bkfetch(feat_cfg, 'descriptors', seg_id) ;
  
  % last feature in the current image
  last_feat = nfeats(curr_seg) ;    
  
  % features sampled from this image
  sel_feats = which_feats(which_feats <= last_feat) ;
  sel_nfeats = length(sel_feats) ;
  
  % add them to descr buffer
  descr(:, bg:bg+sel_nfeats - 1) = d(:, sel_feats) ;
  bg = bg + sel_nfeats ;  
  
  fprintf('block_dictionary: loaded %.1f %%\r',...
          100 * (bg - 1) / bk.nfeats) ;
  
  % stop?
  if length(which_feats) == sel_nfeats,
    break ;
  end
  
  % go on
  which_feats = which_feats(sel_nfeats+1:end) - last_feat ;    
  curr_seg = curr_seg + 1 ;
end
fprintf('\n') ;

end

% --------------------------------------------------------------------
function dict = learn(bk, descr)
% --------------------------------------------------------------------

switch lower(bk.dictionary)
  case 'ikm'    
    fprintf('block_dictionary: === Running IKM ===\n') ;
    fprintf('block_dictionary: num words (K): %d\n', bk.ikm_nwords) ;

    dict = vl_ikmeans(descr, bk.ikm_nwords, 'verbose', 'method', 'elkan') ;
    fprintf('block_dictionary: IKM done\n') ;
    
  case 'hikm'
    fprintf('block_dictionary: === Running HIKM ===\n') ;
    fprintf('block_dictionary: num leaves:    %d\n', bk.hikm_nleaves) ;
    fprintf('block_dictionary: branching (K): %d\n', bk.hikm_K) ;
    fprintf('block_dictionary: only_leaves:   %d\n', bk.hikm_only_leaves);
    
    dict = vl_hikmeans(descr, bk.hikm_K, bk.hikm_nleaves, 'verbose', ...
                    'method', 'elkan') ;
    fprintf('block_dictionary: HIKM done\n') ;
    
  otherwise
    error('Unkown dictionary type') ;
end

end

% --------------------------------------------------------------------
function varargout = fetch__(bk, what, varargin)
% --------------------------------------------------------------------

global wrd ;

switch lower(what)
  
  case 'type'
    varargout{1} = bk.dictionary ;
    
  case {'dict', 'dictionary'}
    if length(varargin) == 0
      n = 1 ;
    else
      n = varargin{1} ;
    end
    path = fullfile(wrd.prefix, bk.tag, sprintf('dict-%02d.mat', n)) ;
    data = load(path, '-MAT') ;
    varargout{1} = data.dict ;
    
  otherwise
    error(sprintf('Unknown ''%s''.', what)) ;
end

end
