function bk = block_classify_svm(bk)
% BLOCK_CLASSIFY_SVM Classify segments with an SVM 
%   This block learns an SVM using an input kernel and uses it to
%   classify segment histograms.
%
%   BK = BLOCK_CLASSIFY_SVM() Initializes the block with the default
%   options.
%
%   BK = BLOCK_CLASSIFY_SVM(BK) Executes the block with options and
%   inputs BK.
%
%   Required Inputs:
%
%   kernel::
%     A pre-computed kernel block
%
%   hist::
%     Segment histograms
%
%   Options:
%
%   bk.seg_neighbors::
%     The number of neighbors to include in the histogram. Default 0.
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
%   bk.probability::
%     Compute probability output? Default 0.
%
%   bk.bg_cat::
%     Category to assign the segment to if it has an empty histogram.
%     Default 0.
%
%   Fetchable attributes:
%
%   type::
%     The type of classifier used 'svm'
%
%   cl::
%     A structure representing the classifier.
%   
%   Block Functions:
%
%   function [class confidence] = classify(cl, h)::
%     Classify histogram h with cl using the SVM.

% AUTORIGHTS
% Copyright (c) 2009 Brian Fulkerson and Andrea Vedaldi
% Blocks is distributed under the terms of the modified BSD license.
% The full license may be found in LICENSE.

global wrd ;

if nargin == 0
  bk = bkinit('classify_svm', 'kernel', 'hist') ;
  bk.fetch      = @fetch__ ;
  bk.seg_neighbors  = 0 ;

  bk.svm_type      = 'C' ;
  bk.svm_C         = 1 ;
  bk.svm_nu        = 0.5 ;
  bk.svm_balance   = 0 ;
  bk.svm_cross     = 10 ;
  bk.svm_rbf       = 1 ;
  bk.svm_gamma     = [] ;
  bk.debug         = 0 ;
  bk.verb          = 1 ;
  bk.probability   = 0 ;
  bk.bg_cat        = 0 ;
  
  return ;
end

% --------------------------------------------------------------------
%                                                      Virutal methods
% --------------------------------------------------------------------

bk.classify = @classify__;

% --------------------------------------------------------------------
%                                                              Do work
% --------------------------------------------------------------------

[bk, dirty] = bkbegin(bk) ;
if ~ dirty, return ; end

% Load svm
kercfg = bkfetch(bk.kernel.tag);
[cl.ker_func cl.norm_func] = kernel_function(kercfg.kernel, kercfg.normalize);

K = bkfetch(bk.kernel.tag, 'kernel');
[train_ids labels] = bkfetch(bk.trainsel.tag, 'train_ids');

labels = labels';
cl.svm = svmkernellearn(...
  K,                 labels,         ...
  'type',            bk.svm_type,    ...
  'C',               bk.svm_C,       ...
  'nu',              bk.svm_nu,      ...
  'balance',         bk.svm_balance, ...
  'crossvalidation', bk.svm_cross,   ...
  'rbf',             bk.svm_rbf,     ...
  'gamma',           bk.svm_gamma,   ...
  'probability',     bk.probability, ...
  'debug',           bk.debug,       ...
  'verbosity',       bk.verb         ) ;

cl.svs = cl.svm.libsvm_cl.SVs; 
cl.train_segs = bkfetch(bk.trainsel.tag, 'train_ids');
cl.bg_cat = bk.bg_cat;
cl.nclasses = length(cl.svm.labels);
cl.probability = bk.probability;

% Build training data
hists = bkfetch(bk.hist.tag, 'seghistograms', cl.train_segs(1,1));
nclusters = size(hists,2);
sv_segs = cl.train_segs(cl.svs, :);

cl.trainingdata = zeros(nclusters, size(sv_segs, 1));
for i = 1:length(sv_segs)
  hists = bkfetch(bk.hist.tag, 'seghistograms', sv_segs(i, 1), ...
                  bk.seg_neighbors);
  cl.trainingdata(:, i) = hists(sv_segs(i, 2), :)';
  cl.trainingdata(:, i) = cl.trainingdata(:, i) / cl.norm_func(cl.trainingdata(:, i));
end

save(fullfile(wrd.prefix, bk.tag, 'cl.mat'), 'cl', '-MAT');
  
bk = bkend(bk) ;
end

function [class confidence] = classify__(cl, h)
  if sum(h) == 0
    confidence = 0;
    if cl.probability
      confidence = zeros(1, cl.nclasses);
    end
    class = cl.bg_cat;
    return
  end
  % Normalize histogram
  h = h / cl.norm_func(h);

  % Compute kernel
  K = zeros(1, length(cl.train_segs));
  K(cl.svs) = cl.ker_func(h, cl.trainingdata);

  % Classify
  [class confidence]  = svmkerneltest(cl.svm, K, 'probability', cl.probability) ;

end

% --------------------------------------------------------------------
function varargout = fetch__(bk, what, varargin)
% --------------------------------------------------------------------
global wrd ;

switch lower(what)
  case 'type'
    varargout{1} = 'svm' ;
  case 'cl'
    path = fullfile(wrd.prefix, bk.tag, 'cl.mat');
    data = load(path, '-MAT');
    varargout{1} = data.cl;
  otherwise
    error(sprintf('Unknown ''%s''.', what)) ;
  end
end
