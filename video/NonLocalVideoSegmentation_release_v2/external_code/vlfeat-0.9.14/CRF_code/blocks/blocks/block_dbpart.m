function bk = block_dbpart(bk)
% BLOCK_DBPART  Partition a database into training and testing images
%   BK = BLOCK_DBPART() Initializes the block with the default
%   options.
%
%   BK = BLOCK_DBPART(BK) Executes the block with options and inputs
%   BK.
%
%   Required inputs:
%   
%   db::
%     The database block.
%
%   Options:
%  
%   bk.db_type::
%     The type of database. Supported types are:
%     Graz02:      Graz02 (see also BLOCK_DBPART())
%     Graz02odds:  Graz02 with odd numbered images for training
%     Caltech4:    Caltech-4
%     Caltech101:  Caltech-101
%     Pascal05:    Pascal Challenge 2005
%     Pascal07:    Pascal Challenge 2007
%     Pascal09:    Pascal Challenge 2009
%     Scenes:      Fei-fei and Lazebnik scenes database.
%     Path:        A generic folder of images, randomly partitioned.
%
%   bk.rand_seed::
%     Set the random seeds before executing. Default [], does not
%     change the seeds.
%
%   bk.fg_cat::
%     The foreground category name. Required parameter for db types:
%     graz02, graz02odds, and pascal05.
%
%   bk.ntrain::
%     The number of training images. Required parameter for db_types:
%     caltech4, scenes, caltech101, path.
%
%   bk.cat_filt::
%     A regular expression that specifies which categories to keep.
%     Optional parameter for db_types: caltech4, scenes, caltech101,
%     path.

% AUTORIGHTS
% Copyright (c) 2009 Brian Fulkerson and Andrea Vedaldi
% Blocks is distributed under the terms of the modified BSD license.
% The full license may be found in LICENSE.

global wrd ;

if nargin == 0
  bk = bkinit('dbpart', 'db') ;
  bk.fetch     = @fetch__ ;
  bk.rand_seed = [] ;
  bk.db_type   = '';
  return ;
end

% --------------------------------------------------------------------
%                                                    Check/load inputs
% --------------------------------------------------------------------

[bk, dirty] = bkbegin(bk) ;
if ~ dirty, return ; end

db_cfg = bkfetch(bk.db.tag) ;
db     = bkfetch(bk.db.tag, 'db') ;

% --------------------------------------------------------------------
%                                                              Do work
% --------------------------------------------------------------------

if ~ isempty(bk.rand_seed)
  setrandseeds(bk.rand_seed) ;
end

switch lower(bk.db_type)

  % ------------------------------------------------------------------
  case 'graz02'
    
    % Ref. [Opelt et al.], [Ling et al.]
    fg = find_cat(db, bk.fg_cat) ;
    bg = find_cat(db, 'none') ;
    
    selp = find([db.segs.cat] == fg) ;
    seln = find([db.segs.cat] == bg) ;
      
    selp = selp(1:150+75) ;
    seln = seln(1:150+75) ;
    
    selp_test = selp(3:3:end) ;
    seln_test = seln(3:3:end) ;
    
    selp_train = setdiff(selp, selp_test) ;
    seln_train = setdiff(seln, seln_test) ;
    
    db.segs = db.segs([selp_train seln_train selp_test seln_test]) ;
    for i=1:300,   db.segs(i).flag = db.TRAIN ; end
    for i=301:450, db.segs(i).flag = db.TEST ; end
  
  case 'graz02odds'

    % Ref. [Marzalek Schmid CVPR07, Opelt Pinz]
    fg = find_cat(db, bk.fg_cat) ;
    bg = find_cat(db, 'none') ;
    
    selp = find([db.segs.cat] == fg) ;
    seln = find([db.segs.cat] == bg) ;
      
    selp = selp(1:300) ;
    seln = seln(1:300) ;
    
    selp_test = selp(2:2:end) ;
    seln_test = seln(2:2:end) ;
    
    selp_train = setdiff(selp, selp_test) ;
    seln_train = setdiff(seln, seln_test) ;
    
    db.segs = db.segs([selp_train seln_train selp_test seln_test]) ;
    ntrain = length(selp_train) + length(seln_train) ;
    ntest  = length(selp_test)  + length(seln_test) ;
    for i=1:ntrain,              db.segs(i).flag = db.TRAIN ; end
    for i=ntrain+1:ntrain+ntest, db.segs(i).flag = db.TEST ; end

  % ------------------------------------------------------------------
  case 'pascal05'
    
    fg = find_cat(db, bk.fg_cat) ;
    bg = find_cat(db, ['no-' bk.fg_cat]) ;
    
    selp_train = find([db.segs.cat] == fg & [db.segs.flag] == db.TRAIN) ;
    seln_train = find([db.segs.cat] == bg & [db.segs.flag] == db.TRAIN) ;
    selp_test  = find([db.segs.cat] == fg & [db.segs.flag] == db.TEST) ;
    seln_test  = find([db.segs.cat] == bg & [db.segs.flag] == db.TEST) ;
    
    db.segs = db.segs([selp_train seln_train selp_test seln_test]) ;
 
  case {'pascal07', 'pascal09'}
    nvalidation = 100;
    trainsegs = find([db.segs.flag] == db.TRAIN);
    perm = randperm(length(trainsegs));
    valsegs = trainsegs(perm(1:nvalidation));
    for zz = 1:length(valsegs)
      db.segs(zz).flag = db.VALIDATION;
    end
    
  % ------------------------------------------------------------------
  case {'caltech4', 'scenes', 'caltech101', 'path'}    
    [db.segs.flag] = deal(db.TEST) ;
        
    for c=db.cat_ids
      sel = find([db.segs.cat] == c) ;
      N = length(sel) ;      
      db.segs(sel) = db.segs(sel(randperm(N))) ;
      for i=sel(1:bk.ntrain), db.segs(i).flag = db.TRAIN ; end
    end
    
    if isfield(bk, 'cat_filt')
      keep = [] ;
      s = regexp(db.cat_names, bk.cat_filt) ;
      for c=1:length(db.cat_ids)
        if ~isempty(s{c})
          keep = [keep db.cat_ids(c)] ;
        end
      end
      sel_keep = ismember([db.segs.cat], keep) ;
      db.segs = db.segs(sel_keep) ;
    end
    
end

db.seg_ids   = [db.segs.seg] ;
db.cat_ids   = unique([db.segs.obj_ids]) ;
db.cat_names = db.cat_names(db.cat_ids) ;
if isfield(db, 'class_ids')
  db.class_ids = db.class_ids(db.cat_ids) ;
end

save(fullfile(wrd.prefix, bk.tag, 'db.mat'), '-STRUCT', 'db') ;

bk = bkend(bk) ;

% --------------------------------------------------------------------
function varargout = fetch__(bk, what, varargin)
% --------------------------------------------------------------------

global wrd ;

switch lower(what)
  
  case 'db'
    path = fullfile(wrd.prefix, bk.tag, 'db.mat') ;
    varargout{1} = bkcacheload(bk, 'db', path) ;
    
  otherwise 
    varargout{:} = bkfetch(bk.db.tag, what, varargin{:}) ;
end
