function path = blocks_root
% BLOCKS_ROOT  Get the blocks package root directory
%  PATH = BLOCKS_ROOT() returns the root directory of the Blocks
%  package.

% AUTORIGHTS
% Copyright (c) 2009 Brian Fulkerson and Andrea Vedaldi
% Blocks is distributed under the terms of the modified BSD license.
% The full license may be found in LICENSE.

[a,b,c] = fileparts(which('blocks_root')) ;
path = a ;


