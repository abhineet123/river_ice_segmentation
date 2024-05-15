function bk = block_hist(bk, varargin)
% BLOCK_HIST Construct histograms for all database images.
%   This block extracts histograms from images in a database.
%
%   BK = BLOCK_HIST() Initializes the block with the default options.
%
%   BK = BLOCK_HIST(BK) Executes the block with options and inputs BK.
%
%   Required Inputs:
%
%   db::
%     The database of images.
%
%   feat::
%     Features extracted from the database.
%
%   dict::
%     A dictionary to use on the features.
%
%   Options:
%
%   bk.min_sigma::
%     Discard features with scale less than min_sigma. Default 0.
%
%   bk.max_num::
%     Keep at most max_num features, selected randomly. Default +inf.
%
%   bk.ref_size::
%     ref_size used in feature extraction. Required to accurately
%     measure min_sigma. Default [] does not modify the scale of the
%     features.
%   
%   bk.rand_seed::
%     Set the random seeds before proceeding. Default [] does not
%     modify the random seeds.
%
%   Fetchable attributes:
%   
%   db::
%     The database used.
%
%   dictionary::
%     The dictionary used.
%
%   histogram::
%     The histogram of a particular image. Required argument: seg_id.
%
%   words::
%     The quantized words which form the histogram. Required argument:
%     seg_id.

% AUTORIGHTS
% Copyright (c) 2009 Brian Fulkerson and Andrea Vedaldi
% Blocks is distributed under the terms of the modified BSD license.
% The full license may be found in LICENSE.

global wrd ;

if nargin == 0
  bk           = bkinit('hist', 'db', 'feat', 'dict') ;
  bk.fetch     = @fetch__ ;
  bk.min_sigma = 0 ;
  bk.max_num   = +inf ;
  bk.ref_size  = [] ;
  bk.rand_seed = [] ;
  return ;
end

% --------------------------------------------------------------------
%                                                    Check/load inputs
% --------------------------------------------------------------------

[bk, dirty] = bkbegin(bk) ;
if ~ dirty, return ; end

bkdb   = bkfetch(bk.db.tag) ;
bkfeat = bkfetch(bk.feat.tag) ;
bkdict = bkfetch(bk.dict.tag) ;

db     = bkfetch(bkdb, 'db') ;
dict   = bkfetch(bkdict, 'dictionary') ;

% --------------------------------------------------------------------
%                                                       Do computation
% --------------------------------------------------------------------

fprintf('block_hist: ref_size:  %s\n', tostr(bk.ref_size)) ;
fprintf('block_hist: min_sigma: %s\n', tostr(bk.min_sigma)) ;
fprintf('block_hist: max_num:   %s\n', tostr(bk.max_num)) ;

if isfield(bk, 'fg_cat')
  db_fg_id = 0;
  for i=1:length(db.cat_names)
    if strcmp(db.cat_names{i}, bk.fg_cat)
        db_fg_id = db.cat_ids(i);
    end
  end
end

keys = 1:length(db.segs) ;
[reduce, mapkeys] = bksplit(bk, keys, varargin{:}) ;
ensuredir(fullfile(wrd.prefix, bk.tag, 'data')) ;

for t=1:length(mapkeys)

  if ~ isempty(bk.rand_seed)
    setrandseeds(bk.rand_seed + mapkeys(t)-1) ;
  end
    
  seg_id = db.segs(mapkeys(t)).seg ;
  d = bkfetch(bkfeat, 'descriptors', seg_id) ;

  %
  % Filter features
  %
  
  sel = 1:size(d,2) ;
  
  f = bkfetch(bkfeat, 'frames', seg_id) ;    
  if bk.min_sigma > 0 
    if ~isempty(bk.ref_size)
      info = bkfetch(bkdb, 'imageinfo', seg_id) ;
      rho = bk.ref_size / max(info.Width,info.Height) ;
    else
      rho = 1 ;
    end    
    keep = rho * f(3,:) > bk.min_sigma ;
    f = f(:,keep) ;
    sel = sel(keep) ;
  end

  % Only filter training images belonging to the fg category
  % TODO: Do this like vis, using the category of the image
  if isfield(bk, 'seg_prefix') && db.segs(mapkeys(t)).flag == db.TRAIN && ...
      db.segs(mapkeys(t)).cat == db_fg_id
    S = getseg(bk.seg_prefix, bk.seg_ext, db.segs(mapkeys(t)).path, bk.fg_id);
    ind = sub2ind(size(S), round(f(2,:)), round(f(1,:)));
    keep = find(S(ind)) ;
    sel = sel(keep) ;
  end
  
  if size(sel,2) > bk.max_num
    N = numel(sel) ;
    keep = logical(zeros(1,N)) ;
    perm = randperm(N) ;
    keep(perm(1:bk.max_num)) = 1 ;
    sel = sel(keep) ;
  end
  
  d = d(:,sel) ;

  %
  % Project features
  %
  
  [w, h, dsel] = bkdict.push(dict, d) ;
  sel = sel(dsel);
  
  %
  % Save back
  %
  
  n = fullfile('data', sprintf('%05d', seg_id)) ;
  save(fullfile(wrd.prefix, bk.tag, [n '.w']), 'w', 'sel', '-MAT') ;
  save(fullfile(wrd.prefix, bk.tag, [n '.h']), 'h', '-MAT') ;  
  fprintf('block_hist: %3.0f%% completed\n', ...
          t / length(mapkeys) * 100) ;    
end

if reduce
  bk = bkend(bk) ;
end

function seg = getseg(gt_prefix, gt_ext, imname, fg_cat)

[pathstr,name,ext] = fileparts(imname);
gt_name = fullfile(gt_prefix, pathstr, [name '.' gt_ext]);

seg = imread(gt_name);
seg = (seg == fg_cat);

% --------------------------------------------------------------------
function varargout = fetch__(bk, what, varargin)
% --------------------------------------------------------------------
global wrd ;

switch lower(what)
  
  case 'db'
    varargout{1} = bkfetch(bk.db.tag, 'db') ;
    
  case 'dictionary'
    varargout{1} = bkfetch(bk.dict.tag, 'dict') ;
    
  case 'histogram'
    i = varargin{1} ;
    path = fullfile(wrd.prefix, bk.tag, 'data', sprintf('%05d.h', i)) ;    
    data = load(path, '-MAT') ;
    varargout{1} = data.h ;
    
  case 'words'
    i = varargin{1} ;
    path = fullfile(wrd.prefix, bk.tag, 'data', sprintf('%05d.w', i)) ;    
    data = load(path, '-MAT') ;
    varargout{1} = data.w ;
    if nargout > 1
      varargout{2} = data.sel ;
    end
    
  otherwise
    error(sprintf('Unknown ''%s''.', what)) ;
end


