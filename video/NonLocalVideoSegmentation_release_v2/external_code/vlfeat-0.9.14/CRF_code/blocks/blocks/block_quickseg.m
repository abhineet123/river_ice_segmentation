function bk = block_quickseg(bk, varargin)
% BLOCK_QUICKSEG Quick shift segmentation
%   This block computes the quick shift segmentation of all database
%   images.
%
%   BK = BLOCK_QUICKSEG() Initializes the block with the default
%   options.
%
%   BK = BLOCK_QUICKSEG(BK) Executes the block with options and inputs
%   BK.
%
%   Required Inputs:
%   
%   db::
%     The database of images to extract quick shift superpixels on.
%
%   Options:
%
%   bk.ratio::
%     The ratio between spatial consistency and color consistency. See
%     VL_QUICKSEG(), parameter ratio.
%
%   bk.sigma::
%     The standard deviation of the parzen window density estimator.
%     See VL_QUICKSEG(), parameter kernelsize.
%
%   bk.tau::
%     The maximum distance between nodes in the quick shift tree. See
%     VL_QUICKSEG(), parameter maxdist.
%
%   bk.ref_size::
%     Resize to ref_size before performing the segmentation.
%     Segmentation maps are resized to match the original image upon
%     completion. 
%
%   Fetchable attributes:
%
%   segs::
%     A structure representing the segmentation. Required parameter:
%     seg_id. For each superpixel s in the requested image, the format
%     is:
%     segs(s).ind     An linear index to all pixels which form the
%                     superpixel.
%     segs(s).count   The size of the superpixel in pixels.
%     segs(s).color   The mean color of the superpixel.
%     segs(s).adj     The neighbors of this superpixel.
%
%   segimage::
%     The color segmentation image produced by VL_QUICKSEG(). Required
%     parameter: seg_id.
%
%   segmap::
%     The label map produced by VL_QUICKSEG. Required parameter
%     seg_id. If two outputs are provided, the second output is the
%     labels. If three outputs are provided, the third output is the
%     size of each superpixel.
%
%   labels::
%     A vector of the labels in label map. Required parameter: seg_id
%
%   time::
%     The amount of time taken for a particular image. Required
%     parameter: seg_id.

% AUTORIGHTS
% Copyright (c) 2009 Brian Fulkerson and Andrea Vedaldi
% Blocks is distributed under the terms of the modified BSD license.
% The full license may be found in LICENSE.

global wrd ;

if nargin == 0
  bk             = bkinit('quickseg', 'db') ;
  bk.fetch       = @fetch__ ;
  bk.sigma       = 2 ;
  bk.ratio       = 0.5 ;
  bk.tau         = 11 ;
  bk.ref_size    = [] ;
  return ;
end

% --------------------------------------------------------------------
%                                                    Check/load inputs
% --------------------------------------------------------------------

[bk, dirty] = bkbegin(bk) ;
if ~ dirty, return ; end

db = bkfetch(bk.db.tag, 'db') ;

% --------------------------------------------------------------------
%                                                       Do computation
% --------------------------------------------------------------------

fprintf('block_quickseg: sigma   : %d\n', bk.sigma) ;
fprintf('block_quickseg: ratio   : %f\n', bk.ratio) ;
fprintf('block_quickseg: tau     : %d\n', bk.tau) ;
if ~isempty(bk.ref_size)
fprintf('block_quickseg: ref_size: %d\n', bk.ref_size) ;
end

% --------------------------------------------------------------------
%                                                       Do computation
% --------------------------------------------------------------------

keys = 1:length(db.segs) ;
[reduce, mapkeys] = bksplit(bk, keys, varargin{:}) ;
ensuredir(fullfile(wrd.prefix, bk.tag, 'data')) ;
  
for t=1:length(mapkeys)

  seg_id = db.segs(mapkeys(t)).seg ;
  
  I = bkfetch(bk.db.tag, 'image', seg_id) ;
 
  % resize image to reference size
  [M,N,k] = size(I) ;
  I_ = I; 
  rho = 1;
  if ~isempty(bk.ref_size)
    rho = bk.ref_size / max(M,N) ;
    I_ = imresizesafe(I,rho) ;
  end
  
  tic; 
  [Iseg map] = vl_quickseg(I_, bk.ratio, bk.sigma, bk.tau); 
  elapsed = toc;
  labels = unique(map);

  tic
  % build adjacency matrix and counts
  segs = struct('ind', [], 'count', [], 'color', [], 'adj', []);
  map1 = circshift(map, [1 0]);
  map1(1,:) = map(1,:);
  map2 = circshift(map, [-1 0]);
  map2(end,:) = map(end,:);
  map3 = circshift(map, [0 1]);
  map3(:,1) = map(:,1);
  map4 = circshift(map, [0 -1]);
  map4(:,end) = map(:,end);

  for i=1:length(labels)
    ind = find(map(:) == labels(i));
    segs(i).ind = ind;
    segs(i).count = length(ind);
    [row col] = ind2sub(size(map), ind(1));
    segs(i).color = squeeze(Iseg(row, col, :));
    adj = [map1(ind) map2(ind) map3(ind) map4(ind)];
    adj = unique(adj(:));
    adj = setdiff(adj, labels(i));
    segs(i).adj = adj;
  end

  if ~isempty(bk.ref_size)
    Iseg = imresizesafe(Iseg, 1/rho, 'nearest');
    map  = imresizesafe(map, 1/rho, 'nearest');
  end
  
  % ------------------------------------------------------------------
  %                                                               Save
  % ------------------------------------------------------------------
  n = fullfile('data', sprintf('%05d', seg_id)) ;
  save(fullfile(wrd.prefix, bk.tag, [n '.mat']), ...
       'Iseg', 'map', 'labels', 'elapsed', 'segs', '-MAT') ;
  fprintf('block_quickseg: %3.0f%% completed\n', t / length(mapkeys) * 100) ;
end

if reduce
  bk = bkend(bk) ;
end

% --------------------------------------------------------------------
function varargout = fetch__(bk, what, varargin)
% --------------------------------------------------------------------

global wrd ;

i = varargin{1} ;
path = fullfile(wrd.prefix, bk.tag, 'data', sprintf('%05d.mat', i)) ;
data = load(path, '-MAT') ;

switch lower(what)

  case 'segs'
    varargout{1} = data.segs; 

  case 'segimage'
    varargout{1} = data.Iseg;

  case 'segmap' 
    varargout{1} = data.map;
    varargout{2} = data.labels;
    if nargout > 2
      counts = zeros(length(data.labels), 1);
      varargout{3} = vl_binsum(counts, ones(length(data.map(:)),1), data.map(:));
    end

  case 'labels'
    varargout{1} = data.labels;

  case 'time'
    varargout{1} = data.elapsed;
    
  otherwise
    error(sprintf('Unknown ''%s''.', what)) ;
end


