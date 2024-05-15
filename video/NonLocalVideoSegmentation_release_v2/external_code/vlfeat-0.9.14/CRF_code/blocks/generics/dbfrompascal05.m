function db = dbfrompascal05 (vocroot, varargin)
% DBFROMPASCAL05  Construct DB from PASCAL VOC 2005 data

% AUTORIGHTS
% Copyright (c) 2009 Brian Fulkerson and Andrea Vedaldi
% Blocks is distributed under the terms of the modified BSD license.
% The full license may be found in LICENSE.

verbose    = 0 ;
cat_filt   = '.*' ;

for i=1:2:length(varargin)
  opt = varargin{i} ;
  arg = varargin{i+1} ;
  switch lower(opt)    
    case 'verbose'
      verbose = arg ;    
    case 'catfilter'
      cat_filt = arg ;
    otherwise
      error(sprintf('Unknown option ''%s''.',opt)) ;
  end
end

% --------------------------------------------------------------------
%                                                               Do job
% --------------------------------------------------------------------

verbose   = 1 ;
lastdir   = cd ;
%vocroot   = '~/Data/pascal-05' ;
voccode   = fullfile(vocroot,'PASCAL') ;

which_train = 'train+val' ;
which_test  = 'test2' ;

if ~ exist(voccode, 'dir')
  error(sprintf('Could not find PASCAL VOC 2005 code in ''%s''', voccode)) ;
end

if verbose
  fprintf('dbfrompascal05: which train : ''%s''\n', which_train) ;
  fprintf('dbfrompascal05: which test  : ''%s''\n', which_test) ;
	fprintf('dbfrompascal05: cat filter  : ''%s''\n', cat_filt) ;
end

addpath(voccode) ;
cd(vocroot)

db.TRAIN      = 0 ;
db.TEST       = 1 ;
db.VALIDATION = 2 ;
db.depth      = 1 ;

try
  % initialize VOC 
  VOCinit ;
 
  % get categories
  cat_names = {PASopts.VOCclass.label} ;
  ncats = length(cat_names) ;
	
  % scan all
	db.cat_names = {} ;
  a = 1 ;
	c = 1 ;
		
  for cat_name = cat_names
		cat_name = cat_name{1} ;	
		if(isempty(regexp(cat_name, cat_filt))), continue ; end 
		
		db.cat_names{c}   = cat_name ;
		db.cat_names{c+1} = ['no-' cat_name] ;
		
    if verbose
      fprintf('dbfrompascal05: loading ''%s'' (''%s'')\n', ...
              cat_name, which_train) ;
    end
        
    imgset = VOCreadimgset(PASopts, cat_name, which_train) ;
    
    present = [imgset.recs.present] ;    
    for s = 1:length(imgset.recs) 
      db.segs(a).path = fullfile('VOCdata', imgset.recs(s).imgname) ;
      db.segs(a).seg  = a ;
      db.segs(a).obj  = a ;
      db.segs(a).cat  = c + 1 - imgset.recs(s).present ;
      db.segs(a).flag = db.TRAIN ;

      db.obj_names{a} = fullfile('VOCdata', imgset.recs(s).imgname) ;
      a = a + 1 ;
    end
    
    if verbose
      fprintf('dbfrompascal05: loading ''%s'' (''%s'')\n', ...
              cat_name, which_test) ;
    end
		    
    imgset = VOCreadimgset(PASopts, cat_name,  which_test) ;
    
    present = [imgset.recs.present] ;    
    for s = 1:length(imgset.recs)       
      db.segs(a).path = fullfile('VOCdata', imgset.recs(s).imgname) ;
      db.segs(a).seg  = a ;
      db.segs(a).obj  = a ;
      db.segs(a).cat  = c + 1 - imgset.recs(s).present ;
      db.segs(a).flag = db.TEST ;      
      
      db.obj_names{a} = fullfile('VOCdata', imgset.recs(s).imgname) ;
      a = a + 1 ;
    end		
		
		c = c + 2 ;
  end  
  
catchbk
  rmpath(voccode) ;
  cd(lastdir) ;
  error(lasterr) ;
end

cd(lastdir) ;
rmpath(voccode) ;

if verbose
  fprintf('dbfrompascal05: done.\n') ;
  fprintf('dbfrompascal05: %d segmentss.\n',  length(db.segs)) ;
  fprintf('dbfrompascal05: %d objects.\n',    length(db.obj_names)) ;
  fprintf('dbfrompascal05: %d categories.\n', length(db.cat_names)) ;
end
