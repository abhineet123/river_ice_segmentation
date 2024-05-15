function bk = block_ker(bk, varargin)
% BLOCK_KER Construct a kernel
%   This block constructs an SVM kernel on two sets of segment ids or
%   segment ids and superpixel ids.
%
%   BK = BLOCK_KER() Initializes the block with the default options.
%
%   BK = BLOCK_KER(BK) Executes the block with options and inputs BK.
%
%   Required Inputs:
%
%   hist::
%     Histograms or superpixel histograms.
%
%   Options:
%
%   bk.kernel::
%     The type of kernel to use. The following are supported:
%
%     L2     - k(x,y) = sum x .* y
%     L1     - k(p,q) = sum min(p,q)
%     CHI2   - k(p,q) = sum  2 (p.*q) / (p+q)
%     HELL   - k(p,q) = sum  sqrt(p.*q) / 4
%
%     DL2    - k(x,y) = sum (x-y).^2           
%     DL1    - k(p,q) = sum |p-q|              
%     DCHI2  - k(p,q) = sum (p-q).^2 / (p+q)      
%     DHELL  - k(p,q) = sum (p.^.5 - q.^.5).^2 / 4
%
%    Here p,q denote non-negative vectors, usually l1 normalized
%    (histograms). Notice that DL2, DL1, DCHI2 and DHELL are not
%    kernels, but the corresponding metrics (this is useful to
%    construct RBF kernels). See also KERNEL_FUNCTION() and
%    VL_ALLDIST2(). This parameter is required and there is no
%    default.
%
%   bk.normalize::
%    Normalize the data by the specified norm (L1, L2, ...) before
%    computing the kernel. See also KERNEL_FUNCTION(). This parameter
%    is required and there is no default.
%
%   bk.row_seg_ids::
%     The seg_ids of the rows of the kernel matrix. If use_segs and
%     seg_neighbors are set, row_seg_ids is a Nx2 matrx, where the
%     first column denotes the seg_id and the second column denotes
%     the superpixel.
%
%   bk.col_seg_ids::
%     The seg_ids of the columns of the kernel matrix.  If use_segs and
%     seg_neighbors are set, col_seg_ids is a Nx2 matrx, where the
%     first column denotes the seg_id and the second column denotes
%     the superpixel.
%
%   bk.use_segs::
%     Use superpixel neighborhoods.
%
%   bk.seg_neighbors::
%     The size of the neighborhood to use. If this parameter is
%     ommited, even if bk.use_segs is true, segs will not be used.
%
%   bk.split::
%     Split the job into a number of subtasks. 0 disables splitting.
%
%  Fetchable attributes:
%
%  KERNEL::
%    Returns [K, ROW_SEG_IDS, COL_SEG_IDS].

% AUTORIGHTS
% Copyright (c) 2009 Brian Fulkerson and Andrea Vedaldi
% Blocks is distributed under the terms of the modified BSD license.
% The full license may be found in LICENSE.

global wrd ;

if nargin == 0
  bk = bkinit('ker', 'hist') ;
  bk.row_seg_ids = [] ;
  bk.col_seg_ids = [] ;

  bk.kernel      = [] ; % 'chi2' ;
  bk.normalize   = [] ;

  bk.use_segs    = 0 ;
  
  bk.fetch = @fetch__ ;
  return ;
end

% --------------------------------------------------------------------
%                                                    Check/load inputs
% --------------------------------------------------------------------

[bk, dirty] = bkbegin(bk) ;
if ~ dirty, return ; end

ensuredir(fullfile(wrd.prefix, bk.tag, 'split')) ;

nz = bk.normalize; 
if isempty(nz), nz = 'none' ; end

[ker_func, norm_func] = kernel_function(bk.kernel, nz);

% --------------------------------------------------------------------
%                                                       Do computation
% --------------------------------------------------------------------

% rows, cols: segment ids labeling rows and columns of K.
% M,N       : number of rows and columns of K.
% rowst     : which elements of rows are in cols and where.
% colst     : which elements of cols are in rows and where.

rows = bk.row_seg_ids ;
cols = bk.col_seg_ids ;

M    = length(rows) ;
N    = length(cols) ;

%[drop, rowst] = ismember(rows, cols) ;
%[drop, colst] = ismember(cols, rows) ;

% In order to avoid loading into memory too many signatures
% at once, we split the computation in blocks of about 100
% comparisons per time.

[rr, cr] = split(M, N, 100) ;
keys = 1:size(rr,2) ;
[reduce, mapkeys] = bksplit(bk, keys, varargin{:}) ;

% Process each block of comparisons.

for k=mapkeys
  
  clear hr  hc ;
  
  m  = rr(2,k) - rr(1,k) + 1 ;
  n  = cr(2,k) - cr(1,k) + 1 ;
  i_ = rr(1,k):rr(2,k) ;
  j_ = cr(1,k):cr(2,k) ;
  
  K = zeros(m,n) ;
  
  for i = 1:m
    if bk.use_segs && isfield(bk, 'seg_neighbors')
      hists = bkfetch(bk.hist.tag, 'seghistograms', ...
                      bk.row_seg_ids(i_(i), 1), bk.seg_neighbors);
      hr{i} = hists(bk.row_seg_ids(i_(i), 2), :)';
    else
      hr{i} = bkfetch(bk.hist.tag, 'histogram', bk.row_seg_ids(i_(i))) ;
    end
    hr{i} = hr{i} / norm_func(hr{i}) ;
  end
  
  for j = 1:n
    if bk.use_segs && isfield(bk, 'seg_neighbors')
      hists = bkfetch(bk.hist.tag, 'seghistograms', ...
                      bk.col_seg_ids(j_(j), 1), bk.seg_neighbors);
      hc{j} = hists(bk.col_seg_ids(j_(j), 2), :)';
    else
      hc{j} = bkfetch(bk.hist.tag, 'histogram', bk.col_seg_ids(j_(j))) ;
    end
    hc{j} = hc{j} / norm_func(hc{j}) ;
  end

  for j = 1:n
    for i = 1:m      
      % compute only if this is not a duplicate
      %if rowst(i_(i)) & colst(j_(j)) & rows(i_(i)) < cols(j_(j))
      %  continue ;
      %end
      
      % compare two signatures
      K(i,j) = ker_func(hr{i}, hc{j}) ;
    end
  end
    
  save(fullfile(wrd.prefix, bk.tag, 'split', ...
                sprintf('K_%05d.mat',k)), 'K', '-MAT') ;
end

% --------------------------------------------------------------------
%                                         Collect results and save back
% --------------------------------------------------------------------

if reduce
  K = zeros(M,N) ;
  
  % assemble kernel blocks
  for k=1:size(rr,2)
    data = load(fullfile(wrd.prefix, bk.tag, 'split', ...
                         sprintf('K_%05d.mat',k)), '-MAT') ;
    r = rr(1,k):rr(2,k) ;
    c = cr(1,k):cr(2,k) ;
    K(r,c) = data.K ;
  end
  
  % fill holes left by duplicates
  %for j=1:N
  %  for i=1:M
  %    if rowst(i) & colst(j) & rows(i) < cols(j)
  %      K(i,j) = K(colst(j),rowst(i)) ;
  %    end
  %  end
  %end
  
  row_seg_ids = bk.row_seg_ids ;
  col_seg_ids = bk.col_seg_ids ;
  
  save(fullfile(wrd.prefix, bk.tag, 'K.mat'), ...
       'K', 'row_seg_ids', 'col_seg_ids') ;
  
  bk = bkend(bk) ;
end

% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



% --------------------------------------------------------------------
function varargout = fetch__(bk, what, varargin)
% --------------------------------------------------------------------

global wrd ;

switch lower(what)
  
  case 'kernel'
    path = fullfile(wrd.prefix, bk.tag, 'K.mat') ;
    data = load(path, '-MAT') ;
    varargout{1} = data.K ;
    varargout{2} = data.row_seg_ids ;
    varargout{3} = data.col_seg_ids ;
    
  otherwise
    error(sprintf('Unknown ''%s''.', what)) ;
end

% --------------------------------------------------------------------
function [rr, cr] = split(r, c, n)
% --------------------------------------------------------------------
% split a matrix r x c in about n square submatrices rr x cr

rg = floor(r / sqrt(n)) + 1 ;
cg = rg ;

r_ = ceil(r / rg) ;
c_ = ceil(c / cg) ;
n_ = r_ * c_ ;

k = 1 ;
for j = 0 : c_ - 1
  for i = 0 : r_ - 1
    rr(:,k) = i*rg + [1 ; rg] ;
    cr(:,k) = j*cg + [1 ; cg] ;
    k=k+1 ;
  end
end

rr = min(rr, r) ;
cr = min(cr, c) ;
