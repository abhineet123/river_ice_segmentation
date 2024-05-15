function bk = block_train_svm(bk, varargin)
% BLOCK_TRAIN_SVM Train an SVM
%   Trains an SVM from a provided training kernel.
%
%   BK = BLOCK_TRAIN_SVM() Initializes the block with the default
%   options.
%
%   BK = BLOCK_TRAIN_SVM(BK) Executes the block with options and
%   inputs BK.
%
%   Required inputs:
%
%   db::
%     The database.
%
%   kernel::
%     The training kernel.
%
%   Options:
%
%   bk.svm_type::
%     The type of SVM to learn with libSVM. Default 'C'.
%
%   bk.svm_C::
%     The value of C to use. Default 1.
%
%   bk.svm_nu::
%     The value of nu to use. Default 0.5.
% 
%   bk.svm_balance::
%     Balance the svm? Default 0.
%
%   bk.svm_cross::
%     Perform N-fold cross validation. Default 10.
% 
%   bk.svm_rbf::
%     Use an rbf kernel? Default 1.
%
%   bk.svm_gamma::
%     Gamma for the rbf kernel. Default [] means automatically
%     determine a good gamma.
%
%   bk.debug::
%     Run the SVM in debug mode? Default 0.
%
%   bk.verb::
%     Be verbose? Default 1.
%
%   Fetchable attributes:
%
%   svm::
%     The trained svm.

% AUTORIGHTS
% Copyright (c) 2009 Brian Fulkerson and Andrea Vedaldi
% Blocks is distributed under the terms of the modified BSD license.
% The full license may be found in LICENSE.

global wrd ;

if nargin == 0
  bk = bkinit('train_svm', 'db', 'kernel') ;
  bk.fetch = @fetch__ ;

  bk.svm_type      = 'C' ;
  bk.svm_C         = 1 ;
  bk.svm_nu        = 0.5 ;
  bk.svm_balance   = 0 ;
  bk.svm_cross     = 10 ;
  bk.debug         = 0 ;
  bk.verb          = 1 ; 
  bk.seg_ids       = [];
  
  return ;
end

% --------------------------------------------------------------------
%                                                    Check/load inputs
% --------------------------------------------------------------------

[bk, dirty] = bkbegin(bk) ;
if ~ dirty, return ; end

db      = bkfetch(bk.db.tag,     'db') ;
[K,r,c] = bkfetch(bk.kernel.tag, 'kernel') ;

% --------------------------------------------------------------------
%                                                       Do computation
% --------------------------------------------------------------------

if length(bk.seg_ids) > 0
    sel_train = bk.seg_ids;
else
    sel_train = find([db.segs.flag]==db.TRAIN) ;
end

y         = [db.segs(sel_train).cat] ;

train_seg_ids = [db.segs(sel_train).seg] ;
%if ~isequal(r(:,1), train_seg_ids) | ~isequal(c(:,1), train_seg_ids)
%  error('Kernel and database are inconsistent!')
%end
fprintf('block_train_svm: Starting training\n');
svm = svmkernellearn(...
  K,                 y ,             ...
  'type',            bk.svm_type,    ...
  'C',               bk.svm_C,       ...
  'nu',              bk.svm_nu,      ...
  'balance',         bk.svm_balance, ...
  'crossvalidation', bk.svm_cross,   ...
  'rbf',             bk.svm_rbf,     ...
  'gamma',           bk.svm_gamma,   ...
  'debug',           bk.debug,       ...
  'verbosity',       bk.verb         ) ;

save(fullfile(wrd.prefix, bk.tag, 'svm.mat'), 'svm', '-MAT') ;
bk = bkend(bk) ;

% --------------------------------------------------------------------
function varargout = fetch__(bk, what, varargin)
% --------------------------------------------------------------------

global wrd ;

switch lower(what)
    
  case 'svm'
    path = fullfile(wrd.prefix, bk.tag, 'svm.mat') ;    
    data = load(path, '-MAT') ;
    varargout{1} = data.svm ;
  
  otherwise
    error(sprintf('Unknown ''%s''.', what)) ;
end


