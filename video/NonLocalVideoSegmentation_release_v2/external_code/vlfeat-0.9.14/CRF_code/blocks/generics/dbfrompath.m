function db = dbfrompath(path, varargin)
% DBFROMPATH  Initialize a database of images from a directory
%  DB = DBFROMPATH(PATH) builds a database from the images in a file
%  hierarchy. The directory PATH has one sub-directory for each visual
%  category and, within each of those, a list of images from the
%  corresponding category.
%
%  Many simple databases such as Caltech-4, Caltech-101 and Graz come
%  in this format.
%
%  A few options may be used to control the process:
%
%  Shuffle [0]
%    Shuffle images within each category.
%
%  Verbose [0]
%    Set verbosity level.
%
%  FileFilter [.png|.jpg|.gif|.bmp|.pgm|.pbm|.ppm]
%    Set a regexp to filter out unwanted image file. Use Verbose=1
%    to see the default filter.
%
%  DirFilter [all directories]
%    Set a regexp to filter otu unwanted directories. Use Verbose=1
%    to see the default filter.

% AUTORIGHTS
% Copyright (c) 2009 Brian Fulkerson and Andrea Vedaldi
% Blocks is distributed under the terms of the modified BSD license.
% The full license may be found in LICENSE.

do_shuffle = 0 ;
verbose    = 0 ;
file_filt  = '.*\.(png|PNG|jpg|JPG|jpeg|JPEG|gif|GIF|bmp|BMP|pgm|PGM|pbm|PBM|ppm|PPM)$' ;
dir_filt   = '.*' ;

for i=1:2:length(varargin)
  opt = varargin{i} ;
  arg = varargin{i+1} ;
  switch lower(opt)    
    case 'shuffle'
      do_shuffle = arg ;
    case 'verbose'
      verbose = arg ;    
    case 'filefilter'
      file_filt = arg ;
    case 'dirfilter'
      dir_filt = arg ;
    otherwise
      error(sprintf('Unknown option ''%s''.',opt)) ;
  end
end

% scan corpus
k         = 1 ; % object path
c         = 1 ; % category index

if verbose
  fprintf('dbfrompath: shuffle images   : %d\n',    do_shuffle) ;
  fprintf('dbfrompath: directory filter : ''%s''\n', dir_filt) ;
  fprintf('dbfrompath: file filter      : ''%s''\n', file_filt) ;
end

% pre initialize structure to have the member oreder we like
db.segs     = [] ;
db.depth       = 1 ;
db.obj_names   = {} ;
db.cat_names   = {} ;
db.images_path = path ;
db.TRAIN       = 0 ;
db.TEST        = 1 ;
db.VALIDATION  = 2 ;

dir_list = dir(path) ;
dir_list = {dir_list([dir_list.isdir]).name} ;
for dn = dir_list
  dir_name = dn{1} ;
  if(strcmp(dir_name,'.')  || ...
     strcmp(dir_name,'..') || ...
     isempty(regexp(dir_name,dir_filt)))
    continue ;  
  end
  
  db.cat_names{c} = dir_name ;
  
  file_list = dir(fullfile(path, dir_name)) ;
  file_list = {file_list(~[file_list.isdir]).name} ;
  
  if do_shuffle
    perm = randperm(length(file_list)) ;
    file_list = file_list(perm) ;
  end
  
  for fn = file_list
    file_name = fn{1} ;
    if(isempty(regexp(file_name, file_filt))), continue ; end 
        
    db.segs(k).seg        = k ;    
    db.segs(k).obj        = k ;    
    db.segs(k).cat        = c ;
    db.segs(k).path       = fullfile(dir_name,file_name) ;

    db.obj_names{k}       = db.segs(k).path ;

    k = k + 1 ;

    if verbose > 1
      fprintf('dbfrompath: added ''%s''\r',file_name) ;
    end
  end % next file
  c = c + 1 ;
end % next directory

if verbose
  fprintf('\ndbfrompath: path: ''%s''\n', path) ;
  fprintf('dbfrompath: done.\n') ;
  fprintf('dbfrompath: %d segs.\n',       length(db.segs)) ;
  fprintf('dbfrompath: %d objects.\n',    length(db.obj_names)) ;
  fprintf('dbfrompath: %d categories.\n', length(db.cat_names)) ;
end

db.aspects = db.segs
