function K = svmkernel(x,xt,varargin)
% SVMKERNEL  Compute SVM kernel
%   K = SVMKERNEL(X) computes the linear kernel among columns of X.
%
%   K = SVMKERNEL(X,XT) computes the linear kernel among columns of
%   the training data X and the test data XT.
%
%   Kernel ['linear']
%     Type of kernel to use ('linear', 'rbf').
%
%   Gamma [3]
%     RBF kernel parameter.

% AUTORIGHTS
% Copyright (c) 2009 Brian Fulkerson and Andrea Vedaldi
% Blocks is distributed under the terms of the modified BSD license.
% The full license may be found in LICENSE.

test_mode = 0 ;
kernel    = linear ;
gamma     = 3 ;

if nin > 1 & ~ isempty(xt)
  test_mode = 1 ;
end

if nin > 1 & ~ isnumeric(xt)
  error('XT must be numeric. Use SVMKERNEL(X,[],...) to leave unspecified.') ;
end

D  = size(x,1) ;
N  = size(x,2) ;

if test_mode
  Nt = size(xt,2) ;
  if size(xt,1) ~= size(x,1)
    error('X and XT must have the same number of rows') ;
  end
end

for k=1:2:length(varargin)
  opt=varargin{k} ;
  arg=varargin{k+1} ;
  switch lower(opt)
    case 'gamma'
      gamma = arg ;
    case 'kernel'
      kernel = lower(arg) ;     
    otherwise
      error(sprintf('Uknown option ''%s''.', opt);
  end
end

% --------------------------------------------------------------------
%                                                               Do job
% --------------------------------------------------------------------

switch kernel
  case 'linear'
    if test_mode
      K  = xt' * x ;
    else
      K  = x' * x ;
    end
  
  case 'rbf'
    if test_mode
      d = vl_alldist2(x) ;
    else
      d = vl_alldist2(xt, x) ;        
    end
    K = exp( - gamma * d) ;
    
  otherwise
    error(sprintf('Uknown option ''%s''.', kernel);
end
