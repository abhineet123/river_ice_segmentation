function bk = block_db(bk)
% BLOCK_DB  Construct a database
%   This block constructs a database from a folder structure.
%
%   BK = BLOCK_DB() Initializes the block with the default options.
%
%   BK = BLOCK_DB(BK) Executes the block with options and inputs BK.
%
%   Options:
%
%   bk.db_prefix::
%     A prefix to where the database resides
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
%     Path:        A generic database consisting of a folder of
%                  images. Each folder is a category. 
%
%   bk.verb::
%     Be verbose. Default 0.
%
%   Fetchable attributes:
%   
%   db::
%     The database structure.
%
%   image::
%     Image data, requires parameter: segid.
%
%   imageinfo::
%     Image information, required parameter: segid.
%
%   mask::
%     Image mask [if provided], required parameter: segid. 

% AUTORIGHTS
% Copyright (c) 2009 Brian Fulkerson and Andrea Vedaldi
% Blocks is distributed under the terms of the modified BSD license.
% The full license may be found in LICENSE.

global wrd ;

if nargin == 0
  bk           = bkinit('db') ;  
  bk.fetch     = @fetch__ ;
  bk.verb      = 0 ;
  bk.db_prefix = '' ;
  bk.db_type   = '' ;
  return
end

[bk, dirty] = bkbegin(bk) ;
if ~ dirty, return ; end

% --------------------------------------------------------------------
%                                                        Scan database
% --------------------------------------------------------------------
    
switch lower(bk.db_type)
  case {'graz02',     ...
        'graz02odds'}
    db = dbfrompath(bk.db_prefix, 'verbose', bk.verb) ;
    bg = find_cat(db, 'none');
    for i = 1:length(db.segs), db.segs(i).obj_ids = [bg db.segs(i).cat]; end 

    if isfield(bk, 'seg_prefix')
      for i = 1:length(db.segs)
        [pathstr,name,ext] = fileparts(db.segs(i).path);
        segname = fullfile(bk.seg_prefix, pathstr, [name '.png']);
        db.segs(i).classseg = segname;
      end
    end
    if isfield(bk, 'obj_prefix')
      for i = 1:length(db.segs)
        [pathstr,name,ext] = fileparts(db.segs(i).path);
        objname = fullfile(bk.obj_prefix, pathstr, [name '.png']);
        db.segs(i).objseg = objname;
      end
    end

    db.class_ids = [1 2 0 3]; % class_ids in seg_prefix
  
  case {'caltech4',   ...
        'scenes',     ...
        'caltech101', ...
        'path'            }
    db = dbfrompath(bk.db_prefix, 'verbose', bk.verb) ;
   
  case 'pascal05'
    db = dbfrompascal05 (bk.db_prefix, 'verbose', bk.verb) ;
    
  case 'pascal07'
    db = dbfrompascal07 (bk.db_prefix, 'verbose', bk.verb) ;
    db.segs = db.aspects;
    for i = 1:length(db.segs)
      db.segs(i).seg = i;
    end
    db.class_ids = [1:20' 0]; % class_ids in seg_prefix
    
  case 'pascal09'
    if isfield(bk, 'test') && isfield(bk, 'train')
      fprintf('calling\n');
      db = dbfrompascal09 (bk.db_prefix, 'verbose', bk.verb, ...
           'test', bk.test, 'train', bk.train) ;
    else
      db = dbfrompascal09 (bk.db_prefix, 'verbose', bk.verb) ;
    end

    db.segs = db.aspects;
    for i = 1:length(db.segs)
      db.segs(i).seg = i;
    end
    db.class_ids = [1:20' 0]; % class_ids in seg_prefix
    
  otherwise
    error(sprintf('Unknown db type %s ',bk.db_type)) ;
end

db.seg_ids = 1:length(db.segs) ;
db.cat_ids = 1:length(db.cat_names) ;

% --------------------------------------------------------------------
%                                                            Save back
% --------------------------------------------------------------------

save(fullfile(wrd.prefix, bk.tag, 'db'), '-STRUCT', 'db') ;
bk = bkend(bk) ;

% --------------------------------------------------------------------
function varargout = fetch__(bk, what, varargin)
% --------------------------------------------------------------------

global wrd ;

switch lower(what)
  
  case 'db'
    path = fullfile(wrd.prefix, bk.tag, 'db.mat') ;
    varargout{1} = bkcacheload(bk, 'db', path) ;
    
  case 'image'
    db = bkfetch(bk, 'db') ;
    path = fullfile(db.images_path, db.segs(varargin{1}).path) ;
    varargout{1} = imread(path) ;
    
  case 'imageinfo'
    db = bkfetch(bk, 'db') ;
    path = fullfile(db.images_path, db.segs(varargin{1}).path) ;
    varargout{1} = imfinfo(path) ;
    
  case 'mask'
    db = bkfetch(bk, 'db') ;
    if ~isfield(db.segs(varargin{1}), 'mask')
      error('Mask does not exist for %d', varargin{1});
    end
    varargout{1} = imread(db.segs(varargin{1}).mask) ;
    
  otherwise
    error(sprintf('Unknown ''%s''.', what)) ;
end
