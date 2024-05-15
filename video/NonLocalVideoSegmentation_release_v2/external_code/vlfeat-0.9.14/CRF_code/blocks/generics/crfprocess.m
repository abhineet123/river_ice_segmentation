function [labels Eafter E] = crfprocess(segs, segclass, unary, params, boundary)
% CRFPROCESS Processes a CRF on superpixels
%   [LABELS EAFTER E] = CRFPROCESS(SEGS, SEGCLASS, UNARY, PARAMS, BOUNDARY)
%   Sets up and processes a CRF on the structure SEGS, with initial
%   labels SEGCLASS, unary potentials UNARY, parameters PARAMS, and
%   shared boundary lengths BOUNDARY. The energy which is minimized is
%   the same as is suggested in Fulkerson et. al 2009.
%
%   PARAMS is a structure which has l_edge, the tradeoff between the
%   unary and pairwise potentials. 
%
%   BOUNDARY is a sparse matrix which contains the boundary lengths
%   between each adjacent pair of superpixels.
%

% AUTORIGHTS
% Copyright (c) 2009 Brian Fulkerson and Andrea Vedaldi
% Blocks is distributed under the terms of the modified BSD license.
% The full license may be found in LICENSE.

% params has
% luv 0 or 1
% l_edge
% l_offset
if ~isfield(params, 'l_edge')
  params.l_edge = 1;
end

% and possibly
% unary_w
% pairwise_w
% weights

colors = [segs.color];
if params.luv == 1
  C = squeeze(vl_xyz2luv(vl_rgb2xyz(reshape(colors', [size(colors,2), 1, size(colors,1)]))))';
  colors = C;
end

nsegs = length(segs);
% Faster?
D = vl_alldist2(colors,colors);
Ds = sum(D(:));
Ds = Ds / ((nsegs*nsegs - nsegs));
beta = 1/(2*Ds);

pairwise = sparse(length(segs), length(segs));
for i = 1:length(segs)
  for n = 1:length(segs(i).adj)
    j = segs(i).adj(n);
    if nargin > 4
      %pairwise(i,j) = params.l_edge*exp(-beta*norm(colors(:,i) - colors(:,j))) ...
      %                + params.l_offset*boundary(i,j);
      pairwise(i,j) = params.l_edge*boundary(i,j)/(1+norm(colors(:,i) -colors(:,j)));
    else
      pairwise(i,j) = params.l_edge*exp(-beta*norm(colors(:,i) - colors(:,j))) ...
                      + params.l_offset;
    end
  end
end
% make sparse
pairwise = sparse(pairwise);

% unary scale is implicit: unary + 0-1(l_edge*pairwise + l_offset)
nclasses = size(unary,1);

if isfield(params, 'unary_w')
  unary = unary .* repmat(params.unary_w{1}, 1, size(unary,2));
end

if isfield(params, 'pairwise_w')
  labelcost = params.pairwise_w{1};
else
  % This corresponds to a 0-1 indicator function on labels being equal on the
  % pairwise term
  labelcost = ones(nclasses)- eye(nclasses);
end

[labels Eafter E] = crfcore(segclass, unary, pairwise, labelcost);
