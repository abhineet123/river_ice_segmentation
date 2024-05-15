function str = tostr(x)
% TOSTR Convert an object to a string representation

% AUTORIGHTS
% Copyright (c) 2009 Brian Fulkerson and Andrea Vedaldi
% Blocks is distributed under the terms of the modified BSD license.
% The full license may be found in LICENSE.

if isa(x,'char')
  str = x ;  
  return ;
end

if iscell(x)
  str = '{';
  for i = 1:length(x)
    str = [tostr(x{i}) ','];
  end
  str(end) = '}';
end

if isnumeric(x)
  if numel(x) == 0
    str = '[]' ;
    return ;
  end
  
  if numel(x) == 1
    str = num2str(x) ;
    return ;
  end
  
  str = num2str(size(x, 1)) ;
  for d=2:ndims(x)
    str = [str 'x' num2str(size(x, d))] ;
  end
  
  str = [str ': [' num2str(x(1))] ;
  for i=2:min(numel(x), 4)
    str = [str ' ' num2str(x(i))] ;
  end
  if numel(x) > 4
    str = [str ' ...]'] ;
  else
    str = [str ']'] ;
  end
end
