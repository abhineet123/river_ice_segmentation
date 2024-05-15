function reports = fetchreports(pattern)
% FETCHREPORTS Fetch reports from visualization blocks
%   REPORTS = FETCHREPORTS(PATTERN) fetches the reports specified by
%   pattern and returns a structure with their contents.

% AUTORIGHTS
% Copyright (c) 2009 Brian Fulkerson and Andrea Vedaldi
% Blocks is distributed under the terms of the modified BSD license.
% The full license may be found in LICENSE.

global wrd ;

reports = {} ;

r = 1 ;
files = dir(wrd.prefix) ;
files = {files([files.isdir]).name} ;
for file=files
  file=file{1} ;
  if(strcmp(file,'.')  || ...
     strcmp(file,'..') || ...
     isempty(regexp(file,pattern)))
    continue ;  
  end
    
  reports{r} = load(fullfile(wrd.prefix, file, 'report.mat')) ;
  reports{r}.file = file;
  
  %fprintf('fetchreports: loaded file ''%s'' (eer: %.2f)%%.\n', ...
  %        file, reports{r}.eer*100) ;
  
  r = r + 1 ;  
end
