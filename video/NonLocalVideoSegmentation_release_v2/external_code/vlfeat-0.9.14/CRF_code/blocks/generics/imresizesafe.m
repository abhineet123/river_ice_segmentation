function I = imresizesafe(varargin)
% IMRESIZESAFE A safe version of IMRESIZE
%   Uses IMRESIZE_OLD if it exists, otherwise defaults to IMRESIZE

% AUTORIGHTS
% Copyright (c) 2009 Brian Fulkerson and Andrea Vedaldi
% Blocks is distributed under the terms of the modified BSD license.
% The full license may be found in LICENSE.

if exist('imresize_old')
    I = imresize_old(varargin{:});
else
    I = imresize(varargin{:});
end
