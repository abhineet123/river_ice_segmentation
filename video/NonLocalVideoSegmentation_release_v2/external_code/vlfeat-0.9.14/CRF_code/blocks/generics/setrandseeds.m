function setrandseeds(seed)
% SETRANDSEEDS  Set all random seeds at once
%  SETRANDSEED(SEED) seeds with SEED all random number
%  generators. These are:
%   
%  - RANDN STATE
%  - RAND STATE
%  - RAND TWISTER
%  - TWISTER (VLFeat random number generator)
%
%  It also make RAND TWISTER the current generator for RAND.

% AUTORIGHTS
% Copyright (c) 2009 Brian Fulkerson and Andrea Vedaldi
% Blocks is distributed under the terms of the modified BSD license.
% The full license may be found in LICENSE.

randn('state',   seed) ;
rand('state',    seed) ;
rand('twister',  seed) ;
vl_twister('state', seed) ;
