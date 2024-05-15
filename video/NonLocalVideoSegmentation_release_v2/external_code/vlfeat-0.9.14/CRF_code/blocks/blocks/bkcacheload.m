function x = bkcacheload(bk, prop, path)
% BKCACHELOAD Loads data from a global cache
%   VAL = BKCACHELOAD(BK, PROP, PATH) Loads the property PROP from the
%   block BK. If the value in the cache is current, it is loaded from
%   memory. Otherwise, it is loaded from PATH and put into the cache.

% AUTORIGHTS
% Copyright (c) 2009 Brian Fulkerson and Andrea Vedaldi
% Blocks is distributed under the terms of the modified BSD license.
% The full license may be found in LICENSE.

global wrd ;

tag = bk.tag ;
tag(tag=='@')='_' ; 

% check for wrd.cahe
if isfield(wrd,'cache')

  % check for wrd.cache.(tag)
  if isfield(wrd.cache, tag)
    
    % check for wrd.cache.(tag).(prop)
    if isfield(wrd.cache.(tag), prop)
      
      % check if up-to-date
      if bk.timestamp <= wrd.cache.(tag).(prop).timestamp
        
        % ok, fetch from cache
        x = wrd.cache.(tag).(prop).value ;
        return ;
      end
    end
  end
end

% Otherwise, load it from disk
x = load(path, '-MAT') ;
wrd.cache.(tag).(prop).timestamp = bk.timestamp ;
wrd.cache.(tag).(prop).value     = x ;
