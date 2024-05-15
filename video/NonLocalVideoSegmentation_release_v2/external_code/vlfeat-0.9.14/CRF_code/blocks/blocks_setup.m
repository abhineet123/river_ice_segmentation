function path = blocks_setup
% BLOCKS_SETUP  Add Blocks toolbox path to MATLAB path
%  PATH = BLOCKS_SETUP() adds Blocks to MATLAB path.

% AUTORIGHTS
% Copyright (c) 2009 Brian Fulkerson and Andrea Vedaldi
% Blocks is distributed under the terms of the modified BSD license.
% The full license may be found in LICENSE.

if exist('setup')==2
  setup
end

root=blocks_root ;

addpath(fullfile(root,'mex'));
addpath(fullfile(root,'generics')) ;
addpath(fullfile(root,'blocks')) ;
addpath(fullfile(root,'experiments')) ;

% Check for existence of VLFeat
if ~exist('vl_sift')
  error(['Blocks requires VLFeat, which does not appear to be available. ' ...
  'Have you initialized VLFeat using vl_setup?']);
elseif ~(exist('vl_sift') == 3) % 3 == mex file, 2 == m file
  error(['Blocks requires VLFeat, but vl_sift is not a mex file. ' ...
  'Did you download the binary version of VLFeat?']);
end

% Check for existence of GCMex
if ~(exist('GCMex')==3)
  error(['GCMex does not exist, or is not compiled. ' ...
  'Make sure you have downloaded GCMex from:\n' ...
  'http://vision.ucla.edu/~brian/gcmex.html' ...
  ])
end

% Check for existence of libsvm*
if ~(exist('svmtrain')==3)
  error(['Blocks requires libsvm, which does not appear to be available. ' ...
  'Have you added the compiled libsvm-mat mex files to your path?']);
end

% Check for existence of IPLD

fprintf('** Welcome to the Blocks Toolbox **\n') ;
