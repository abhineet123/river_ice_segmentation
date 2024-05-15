function [ker_func, norm_func] = kernel_function(kernel, normalize)
% KERNEL_FUNCTION Returns function pointers for various kernels
%   [KFUNC, NFUNC] = KERNEL_FUNCTION(KERNEL, NORMALIZE) returns the
%   kernel KFUNC and normalization function NFUNC for two string
%   inputs.
%    
%   Possible values for normalize:: {'none', 'l1', 'l2', 'linf'}
%   Possible values for kernel:: {'l1', 'chi2', 'hell', 'dl2', 'dl1',
%     'dchi2', 'dhell'}

% AUTORIGHTS
% Copyright (c) 2009 Brian Fulkerson and Andrea Vedaldi
% Blocks is distributed under the terms of the modified BSD license.
% The full license may be found in LICENSE.

if nargin < 2
    normalize = 'none'
end

switch normalize
  case 'none'
    norm_func = @(x) 1 ;
   
  case 'l1'
    norm_func = @(x) (sum(x)+eps) ;
    
  case 'l2'
    norm_func = @(x) (sqrt(sum(x.^2))+eps) ;
    
  case 'linf'
    norm_func = @(x) (max(abs(x))+eps) ;
    
  otherwise
    error(sprintf('Unknown norm ''%s''.', normalize)) ;
end

switch kernel
  case {[], 'l2'}
    ker_func = @(x,y) vl_alldist2(x,y,'kl2') ;
    
  case 'l1'
    ker_func = @(x,y) vl_alldist2(x,y,'kl1') ;
    
  case 'chi2'
    ker_func = @(x,y) vl_alldist2(x,y,'kchi2') ;
    
  case 'hell'
    ker_func = @(x,y) vl_alldist2(x,y,'khell') ;
    
  case 'dl2'
    ker_func = @(x,y) vl_alldist2(x,y,'l2') ;
    
  case 'dl1'
    ker_func = @(x,y) vl_alldist2(x,y,'l1') ;
  
  case 'dchi2'
    ker_func = @(x,y) vl_alldist2(x,y,'chi2') ;

  case 'dhell'
    ker_func = @(x,y) vl_alldist2(x,y,'hell') ;
    
  otherwise
    error(sprintf('Uknown kernel type ''%s''.', kernel)) ;
end

