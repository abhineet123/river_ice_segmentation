function bk = block_test_svm(bk, varargin)
% BLOCK_TEST_SVM Classify whole images with an SVM
%   Classify whole images with an SVM.
%
%   BK = BLOCK_TEST_SVM() Initializes the block with the default
%   options.
%
%   BK = BLOCK_TEST_SVM(BK) Executes the block with options and inputs
%   BK.
%
%   Required inputs:
%
%   kernel::
%     The kernel computed between the training and test images.
%
%   svm::
%     The trained SVM.
%   
%   Fetchable attributes:
%
%   prediction::
%     Returns [PRED, PRED_SEG_IDS, DEC, DEC_LABELS] where PRED are the
%     predictions of the segment labels, PRED_SEG_IDS are the ID of
%     the correspodning segments, DEC is the matrix of the decision
%     functions and DEC_LABELS the corresponding labels. See
%     SVMKERNELTEST()).

% AUTORIGHTS
% Copyright (c) 2009 Brian Fulkerson and Andrea Vedaldi
% Blocks is distributed under the terms of the modified BSD license.
% The full license may be found in LICENSE.

global wrd ;

if nargin == 0
  bk = bkinit('test_svm', 'svm', 'kernel') ;
  bk.fetch = @fetch__ ;
  return ;
end

% --------------------------------------------------------------------
%                                                    Check/load inputs
% --------------------------------------------------------------------

[bk, dirty] = bkbegin(bk) ;
if ~ dirty, return ; end

svm = bkfetch(bk.svm.tag, 'svm') ;
[K,r,c] = bkfetch(bk.kernel.tag, 'kernel') ;

% --------------------------------------------------------------------
%                                                       Do computation
% --------------------------------------------------------------------

[pred, dec]  = svmkerneltest(svm, K) ;
pred_seg_ids = r ;
dec_labels   = svm.labels ;

save(fullfile(wrd.prefix, bk.tag, 'prediction.mat'), ...
     'pred',         ...
     'pred_seg_ids', ...
     'dec',          ...
     'dec_labels',   ...
     '-MAT') ;

bk = bkend(bk) ;

% --------------------------------------------------------------------
function varargout = fetch__(bk, what, varargin)
% --------------------------------------------------------------------

global wrd ;

switch lower(what)
    
  case 'prediction'
    path = fullfile(wrd.prefix, bk.tag, 'prediction.mat') ;
    data = load(path, '-MAT') ;
    varargout{1} = data.pred ;
    varargout{2} = data.pred_seg_ids ;
    varargout{3} = data.dec ;
    varargout{4} = data.dec_labels ;
  
  otherwise
    error(sprintf('Unknown ''%s''.', what)) ;
end
