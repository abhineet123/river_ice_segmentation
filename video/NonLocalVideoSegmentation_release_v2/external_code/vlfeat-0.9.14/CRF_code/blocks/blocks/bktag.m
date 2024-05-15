function tag = bktag(bk)
% BKTAG  Extract tag from a block or a tag
%   TAG = BKTAG(TAG) simply returns the tag, unmodified
%
%   TAG = BKTAG(BK) returns the tag for a block.
%
%   See also: BKVER()

% AUTORIGHTS
% Copyright (c) 2009 Brian Fulkerson and Andrea Vedaldi
% Blocks is distributed under the terms of the modified BSD license.
% The full license may be found in LICENSE.


if isstr(bk),
  tag = bk ;
  return ;
end

tag = bk.tag ;
