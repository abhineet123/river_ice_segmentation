function bk = block_feat(bk, varargin)
% BLOCK_FEAT  Extract features from images in a database.
%   This block extracts features from images in a database.
%
%   BK = BLOCK_DICTIONARY() Initializes the block with the default
%   options.
%
%   BK = BLOCK_DICTIONARY(BK) Executes the block with options and
%   inputs BK.
%
%   Required Inputs:
%
%   db::
%     The database to extract features from. All segments will be
%     processed.
%
%   Options:
%
%   bk.rand_seed::
%     Set the random seed. Default is [], which does not change the
%     random seed.
%
%   bk.detector::
%     The type of feature detector to use. Default 'sift'. Valid types
%     are:
%     sift:         Standard SIFT detector (DoG) from VLFeat.
%     ipld:         Harris multiscale + DoG (IPLD implementation).
%     iplddog:      The DoG from the IPLD implementation.
%     dsift:        Dense SIFT. Only compatible with dsift descriptor.
%     dsift-color:  Same as dsift. Only compatible with dsift-color
%                   descriptor.
%
%   bk.descriptor::
%     The type of feature descriptor to use. Default 'sift'. Valid
%     types are:
%     sift:         Standard SIFT descriptor from VLFeat.
%     siftnosmooth: Standard SIFT descriptor which ommits smoothing.
%     ipld:         Standard SIFT descriptor (IPLD implementation).
%     dsift:        Dense SIFT descriptor.
%     dsift-color:  Dense color SIFT descriptor
%
%   bk.ref_size::
%     Resize images to have their longest side equal to this number
%     before processing. Default of [] means leave the images
%     unmodified.
%
%   bk.min_sigma::
%     Throw away detected features whose frame size is less than
%     min_sigma. Frame size is relative to the ref_size of the image.
%   
%   bk.max_num:: 
%     A limit on how many features are extracted. Default +inf.
%
%   bk.rescale::
%     Descriptors are computed on regions whose radius is rescale times
%     the scale sigma of the frame. For the standard sift descriptor,
%     for instance, rescale is two times the magnification paramter of
%     the descriptor. Default 6.
%
%   DSIFT and DSIFT-COLOR Options:
%
%   bk.dsift_size:
%     The size of a spatial bin in dsift. For example, 3 will create a
%     descriptor which is 12x12 pixels. This option is required when
%     using this descriptor.
%
%   bk.dsift_step:
%     The step size in pixels between descriptors. 1 produces a
%     descriptor for every pixel of the image. This option is required
%     when using this descriptor.
%
%   DSIFT-COLOR Options:
%
%   bk.dsift_minnorm:
%     Discard descriptors whose norm is less than dsift_minnorm in
%     both the red and green channels.

% AUTORIGHTS
% Copyright (c) 2009 Brian Fulkerson and Andrea Vedaldi
% Blocks is distributed under the terms of the modified BSD license.
% The full license may be found in LICENSE.

global wrd ;

if nargin == 0
  bk             = bkinit('feat', 'db') ;
  bk.fetch       = @fetch__ ;
  bk.rand_seed   = [] ;
  bk.detector    = 'sift' ;
  bk.descriptor  = 'siftnosmooth' ;
  bk.ref_size    = [] ;
  bk.min_sigma   = 0 ;
  bk.max_num     = +inf ; 
  bk.rescale     = 6 ;
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

fprintf('block_feat: detector  : %s\n', bk.detector) ;
fprintf('block_feat: descriptor: %s\n', bk.descriptor) ;
fprintf('block_feat: ref_size  : %s\n', num2str(bk.ref_size)) ;
fprintf('block_feat: min_sigma : %f\n', bk.min_sigma) ;
fprintf('block_feat: max_num   : %d\n', bk.max_num) ;
fprintf('block_feat: rescale   : %f\n', bk.rescale) ;

if strcmp(bk.detector, 'dsift') && ~strcmp(bk.detector, bk.descriptor)
    error('Detector and descriptor should be the same for dsift');
end
% --------------------------------------------------------------------
%                                                       Do computation
% --------------------------------------------------------------------

keys = 1:length(db.segs) ;
[reduce, mapkeys] = bksplit(bk, keys, varargin{:}) ;
ensuredir(fullfile(wrd.prefix, bk.tag, 'data')) ;
  
for t=1:length(mapkeys)

  if ~ isempty(bk.rand_seed)
    setrandseeds(bk.rand_seed + mapkeys(t)-1) ;
  end

  seg_id = db.segs(mapkeys(t)).seg ;
  
  % ------------------------------------------------------------------
  %                                                         Preprocess
  % ------------------------------------------------------------------

  % read image and make it grayscale
  Iorig = bkfetch(bk.db.tag, 'image', seg_id) ;
  I = im2single(Iorig);
  if size(I,3) > 1
    I = rgb2gray(I) ; 
  end
    
  % resize image to reference size
  [M,N,k] = size(I) ;
  if ~isempty(bk.ref_size)
    rho = bk.ref_size / max(M,N) ;
  else
    rho = 1 ;
  end
  I_ = imresize(I,round(rho * [M, N])) ;
  Icolor = imresize(Iorig,round(rho * [M, N])) ;
  Icolor = im2single(Icolor);
  
  % ------------------------------------------------------------------
  %                                                           Detector
  % ------------------------------------------------------------------
  switch bk.detector
    
    case {'baseline', 'ipld'}
      f_dog = ipldfeat(I_, 'detector', 'dog',    'verbosity', 0) ;
      f_har = ipldfeat(I_, 'detector', 'harris', 'verbosity', 0) ;
      f = [f_dog f_har] ;
            
    case 'iplddog'
      f = ipldfeat(I_, 'detector', 'dog','verbosity', 0) ;
      
    case 'sift'
      f = vl_sift(I_) ;    
      
    case  {'dft', 'dsift', 'dsift-color'}
      f = zeros(4,1); % dsift calculates descriptors and features at the same time

    otherwise
      error('Unknown detector type.') ;
  end

  % ------------------------------------------------------------------
  %                                                    Frame selection
  % ------------------------------------------------------------------
  
  % remove frames if too small
  sel = find(f(3,:) > bk.min_sigma) ;
  f = f(:,sel) ;
  
  % remove frames if too many
  if size(f,2) > bk.max_num
    N = size(f,2) ;
    keep = logical(zeros(1,N)) ;
    perm = randperm(N) ;
    keep(perm(1:bk.max_num)) = 1 ;
    f = f(:,keep) ;
  end
  
  % remove frames if too close to the boundary
  [M,N] = size(I_) ;
  R = f(3,:) * bk.rescale ;
  
  keep = f(1,:) - R >= 1 & ...
         f(1,:) + R <= N & ...
         f(2,:) - R >= 1 & ...
         f(2,:) + R <= M ;
  
  f = f(:,keep) ;
  
  if 0
    figure(1) ; clf ;
    imagesc(I_) ;  
    colormap gray ;  
    hold on ;
    vl_plotframe(f) ;
    drawnow ;
  end

  % ------------------------------------------------------------------
  %                                                         Descriptor
  % ------------------------------------------------------------------
    
  switch bk.descriptor
    case 'siftnosmooth'
      [f,d] = siftnosmooth(double(I_), f, 'magnif', bk.rescale/2) ;
      
    case 'sift'
      [f,d] = vl_sift(I_, 'frames', f, 'magnif', bk.rescale/2) ;
      
    case 'ipld'
      [f,d] = ipldfeat(I_, 'frames', f(1:3,:), 'magnif', bk.rescale/2) ;

    case 'dsift'
      dsift_size = bk.dsift_size;
      [f,d] = vl_dsift(I_, 'size', dsift_size, 'step', bk.dsift_step, 'fast');
      sigma = dsift_size*4/6;
      f = [f; sigma*ones(1, size(f,2)); pi/2*ones(1,size(f,2))];
 
      % remove frames if too close to the boundary
      [M,N] = size(I_) ;
      R = f(3,:) * 6 / 2 ; % 6 * sigma is the domain of the descriptor

      keep = f(1,:) - R >= 1 & ...
             f(1,:) + R <= N & ...
             f(2,:) - R >= 1 & ...
             f(2,:) + R <= M ;
      
      f = f(:,keep) ;
      d = d(:,keep) ; 

    case 'dsift-color'
      if size(Icolor) ~= 3, error('dsift-color requires color images'); end

      RGB = (sum(Icolor,3) + eps);
      Irgb = Icolor ./ cat(3,RGB,RGB,RGB);
      
      % Need to combine, possibly with norms
      [fr,dr] = vl_dsift(Irgb(:,:,1), 'size', bk.dsift_size, 'step', ...
        bk.dsift_step, 'fast', 'norm');
      [fg,dg] = vl_dsift(Irgb(:,:,2), 'size', bk.dsift_size, 'step', ...
        bk.dsift_step, 'fast', 'norm');

      % with norms
      %rind = find(fr(3,:) < fg(3,:));
      %gind = find(fr(3,:) >= fg(3,:));
      %dr(:,rind) = uint8(double(dr(:,rind)).*repmat(fr(3,rind)./fg(3,rind), [size(dr,1) 1]));
      %dg(:,gind) = uint8(double(dg(:,gind)).*repmat(fg(3,gind)./fr(3,gind), [size(dg,1) 1]));
      % without norms
      d = [dr; dg];
      keep = union(find(fr(3,:) > bk.dsift_minnorm), find(fg(3,:) > bk.dsift_minnorm));
      f = fr(1:2,:);
      f = f(:,keep);
      d = d(:,keep);

      sigma = bk.dsift_size*4/6;
      f = [f; sigma*ones(1, size(f,2)); pi/2*ones(1,size(f,2))];

      % remove frames if too close to the boundary
      [M,N] = size(I_) ;
      R = f(3,:) * bk.rescale ;

      keep = f(1,:) - R >= 1 & ...
             f(1,:) + R <= N & ...
             f(2,:) - R >= 1 & ...
             f(2,:) + R <= M ;

      f = f(:,keep) ;
      d = d(:,keep) ; 

    otherwise
      error('Unknown descriptor type') ;
  end
  
  % ------------------------------------------------------------------
  %                                                       Post process
  % ------------------------------------------------------------------
  f(1:2,:) = (f(1:2,:) - 1) / rho + 1 ;
  f(3,:)   = f(3,:) / rho ;
  
  % ------------------------------------------------------------------
  %                                                               Save
  % ------------------------------------------------------------------
  n = fullfile('data', sprintf('%05d', seg_id)) ;
  save(fullfile(wrd.prefix, bk.tag, [n '.f']), 'f', '-MAT') ;
  save(fullfile(wrd.prefix, bk.tag, [n '.d']), 'd', '-MAT') ;  
  fprintf('block_feat: %3.0f%% completed\n', ...
          t / length(mapkeys) * 100) ;  
end

if reduce
  bk = bkend(bk) ;
end

% --------------------------------------------------------------------
function varargout = fetch__(bk, what, varargin)
% --------------------------------------------------------------------

global wrd ;

switch lower(what)
  
  case 'descriptors'
    i = varargin{1} ;
    path = fullfile(wrd.prefix, bk.tag, 'data', sprintf('%05d.d', i)) ;    
    data = load(path, '-MAT') ;
    varargout{1} = data.d ;
    
  case 'frames'
    i = varargin{1} ;
    path = fullfile(wrd.prefix, bk.tag, 'data', sprintf('%05d.f', i)) ;    
    data = load(path, '-MAT') ;
    varargout{1} = data.f ;
    
  otherwise
    error(sprintf('Unknown ''%s''.', what)) ;
end

