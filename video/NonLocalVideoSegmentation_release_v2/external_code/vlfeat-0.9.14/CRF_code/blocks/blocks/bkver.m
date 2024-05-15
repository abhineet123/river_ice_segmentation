function version = bkver(bk)
% BKVER  Extracts block version
%   VERSION = BKVER(BK) extracts the version from the block BK. The
%   version is the portion of the block TAG after the '@' symbol.
%   BKVER(TAG) does the same thing, but operates directly on the tag
%   TAG.
%
%   See also BKTAG().

% AUTORIGHTS
% Copyright (c) 2009 Brian Fulkerson and Andrea Vedaldi
% Blocks is distributed under the terms of the modified BSD license.
% The full license may be found in LICENSE.

tag     = bktag(bk) ;
t       = regexp(tag, '^\w*@(.*)$', 'tokens') ;
version = t{1}{1} ;
