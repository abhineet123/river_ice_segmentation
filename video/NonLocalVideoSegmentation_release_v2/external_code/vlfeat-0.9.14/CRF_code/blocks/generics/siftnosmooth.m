function [f,d]=siftnosmooth(I, f, varargin)
% SIFTNOSMOOTH Calculate a SIFT descriptor without smoothing
%   [F,D] = SIFTNOSMOOTH(I, F)
%
% Options:
%
% MAGNIF::
%   SIFT descriptor magnification factor (see SIFTDESCRIPTOR()).

% AUTORIGHTS
% Copyright (c) 2009 Brian Fulkerson and Andrea Vedaldi
% Blocks is distributed under the terms of the modified BSD license.
% The full license may be found in LICENSE.

magnif=3 ;

for k=1:2:length(varargin)
  opt=varargin{k} ;
  arg=varargin{k+1} ;
  
  switch lower(opt)    
    case 'magnif'
      magnif=arg ;
    otherwise
      error(sprintf('Uknown option ''%s''', opt)) ;
  end
end

r = .5 ;
N = size(f,2) ;
d = zeros(128,N,'uint8') ;

I = im2double(I) ;
if size(I,3) > 1 ;
  I = rgb2gary(I) ;
end

fo      = floor(log2(f(3,:)/1.6)) ;
octaves = unique(sort(fo)) ;

for o = octaves
  sel = find(fo == o) ;

  % presmooth and downsample image at IPLD level
  I_ = vl_imsmooth(I, r) ;%2^(o-1)) ;  
  I_ = imresize(I_,2^(-o)) ;
  
  % calculate gradienet
  [Ix,Iy] = vl_grad(I_) ;
  mod     = sqrt(Ix.^2 + Iy.^2) ;
  ang     = atan2(Iy,Ix) ;
  
  % downsample keypoints
  f_          = f(:,sel) ;
  f_([1;2],:) = (f_([1;2],:) - 1) / 2^o + 1 + 1;
  f_(3,:)     = f_(3,:) / 2^o ; 
  
  % run descriptor
  grd = shiftdim(cat(3,mod,ang),2) ;
  grd = single(grd) ;
  d(:,sel) = vl_siftdescriptor(grd, f_, 'magnif', magnif) ;
end



