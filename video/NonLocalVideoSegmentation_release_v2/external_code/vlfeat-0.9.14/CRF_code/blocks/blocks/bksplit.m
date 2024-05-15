function [reduce, mapkeys] = bksplit(bk, keys, varargin)
% BKSPLIT  Splits keys according to the split property of the block
%  [REDUCE, MAPKEYS] = BKSPLIT(BK, KEYS) Splits KEYS into N chunks,
%  where N is specified by BK.split. Runs N processes via SSH, which
%  continue at this line of the file with REDUCE = 0 and MAPKEYS
%  equal to the split set of keys. Upon finishing, REDUCE will be set
%  to 1, and MAPKEYS will be empty (since work has been completed). 
%
%  BKSPLIT(...,'MAP',MAPKEYS) is called internally, to signify we have
%  already mapped the keys and perform some action on them.

% AUTORIGHTS
% Copyright (c) 2009 Brian Fulkerson and Andrea Vedaldi
% Blocks is distributed under the terms of the modified BSD license.
% The full license may be found in LICENSE.

global wrd ;

mapkeys = [] ;
mapmode = 0 ;

for k=1:2:length(varargin)
  opt=lower(varargin{k}) ;
  arg=varargin{k+1} ;
  switch opt
    case 'map'
      mapkeys = arg ;
      mapmode = 1 ;
    otherwise
      error(sprintf('Unknown option ''%s''.', opt)) ;
  end
end

% --------------------------------------------------------------------
% If MAP is an option, then the block has been executed in MAP mode.
% --------------------------------------------------------------------

if mapmode
  reduce = 0 ;
  return ;
end

% --------------------------------------------------------------------
% The block was not called in MAP mode. Now decide wether
% the computation should be split or not.
% --------------------------------------------------------------------

if isfield(bk, 'split') & bk.split & wrd.enable_split
  
  % divide keys evenly
  groups = linspace(1, length(keys)+1,bk.split+1) ;
  groups = round(groups) ;
  
  % number of jobs
  num_jobs = length(groups) - 1 ;
  
  % freeze state
  ensuredir(fullfile(wrd.prefix, bk.tag, 'split')) ;
  state_file = fullfile(wrd.prefix, bk.tag, 'split', 'state.mat') ;  
  save(state_file, 'wrd', 'bk', 'keys', 'groups', '-MAT') ;
 
  % prepare scriptlet
  scriptlet{1} = sprintf('load(''%s'') ;', state_file) ;
  scriptlet{2} = sprintf('global wrd ;', state_file) ;
  scriptlet{3} = sprintf('b=groups(cluster_job_id);') ;
  scriptlet{4} = sprintf('e=groups(cluster_job_id+1)-1;') ;  
  scriptlet{5} = sprintf('block_%s(bk, ''map'', keys(b:e)) ;', bk.type) ;

  scriptlet = [scriptlet{:}] ;
  
  % prepare system command
  syscall = sprintf( ...
    './clusterrun.py -s "%s" -j %d -l "%s"', ...
    scriptlet, num_jobs, fullfile(wrd.prefix, bk.tag, 'split', 'log')) ;
  
  fprintf('bksplit: executing ''%s''.\n', syscall) ;
    
  % run system command
  tic
  [st, rs] = unix(syscall, '-echo') ;
  fprintf('bksplit: completed in %.2f sec (%.2f min)\n', ...
          toc, toc/60) ;
  
  if (st)
    error('Parallel run failed!') ;
  end
  
  % after parallel execution, nothing is left to map
  mapkeys = [] ;
else

  % no parallel, no fun
  mapkeys = keys ;
end

% --------------------------------------------------------------------
% Except when run in MAP mode, a block has to REDUCE too.
% --------------------------------------------------------------------

reduce = 1 ;
