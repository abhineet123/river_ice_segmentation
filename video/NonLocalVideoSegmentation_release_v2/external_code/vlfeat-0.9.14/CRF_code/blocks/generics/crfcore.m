function [labels Eafter E] = crfcore(segclass, unary, pairwise, labelcost);
% CRFCORE A simple wrapper around GCMex
%   [LABELS Eafter E] = CRFCORE(SEGCLASS, UNARY, PAIRWISE, LABELCOST)
%   is a simple wrapper around GCMEX(). It converts UNARY and
%   LABELCOST to single if they are not already.

% AUTORIGHTS
% Copyright (c) 2009 Brian Fulkerson and Andrea Vedaldi
% Blocks is distributed under the terms of the modified BSD license.
% The full license may be found in LICENSE.

[labels E Eafter] = GCMex(segclass, single(unary), pairwise, single(labelcost),1);
