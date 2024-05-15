function bk = block_vis(bk, varargin)
% BLOCK_VIS Visualize whole image classification
%   Visualize whole image classification.
%
%   BK = BLOCK_VIS() Initializes the block with the default options.
%
%   BK = BLOCK_VIS(BK) Executes the block with options and inputs BK.
%
%   Required inputs:
%
%   db::
%     The database.
%
%   prediction::
%     The classifier's predictions.
%
%   Options:
%
%   bk.verb::
%     Verbosity level. Default 1.
%
%   Fetchable attributes:
%
%   report::
%     A structure containing a report on the results which has fields:
%     tp:   True positives
%     tn:   True negatives
%     eer:  The equal error point
%     ur:   Uniform prior error rate
%     urt:  Uniform prior error rate threshold
%     auc:  The area under the ROC curve.

% AUTORIGHTS
% Copyright (c) 2009 Brian Fulkerson and Andrea Vedaldi
% Blocks is distributed under the terms of the modified BSD license.
% The full license may be found in LICENSE.

global wrd ;

if nargin == 0
  bk = bkinit('vis', 'db', 'prediction') ;
  bk.fetch = @fetch__ ;
  bk.verb  = 1 ;
  return ;
end

% --------------------------------------------------------------------
%                                                    Check/load inputs
% --------------------------------------------------------------------

[bk, dirty] = bkbegin(bk) ;
if ~ dirty, return ; end

db = bkfetch(bk.db.tag, 'db') ;
[y_,ids,dec,lab] = bkfetch(bk.prediction.tag, 'prediction') ;

% --------------------------------------------------------------------
%                                                       Do computation
% --------------------------------------------------------------------

sel_test = find([db.segs.flag] == db.TEST) ;

test_seg_ids = [db.segs(sel_test).seg] ;
if ~isequal(ids, test_seg_ids)
  error(['Predictions and database are inconsistent!']) ;
end

N = length(sel_test) ;
y = zeros(1,N) ;
y([db.segs(sel_test).cat] == lab(1)) = +1 ;
y([db.segs(sel_test).cat] == lab(2)) = -1 ;

tp    = [] ;  % true positive
tn    = [] ;  % true negative
eer   = [] ;  % equal error rate
ur    = [] ;  % uniform prior error rate
urt   = [] ;  % uniform prior erorr rate threshold
auc   = [] ;  % area under ROC

[tp,tn,info] = vl_roc(y, dec') ;  

eer = info.eer ;
ur  = info.ur ;
urt = info.ut ;
auc = info.auc ;

if bk.verb
  figure(2) ; clf ; tp = vl_roc(y, dec') ; 
end

report.tp  = tp ;
report.tn  = tn ;
report.eer = eer ;
report.ur  = ur ;
report.urt = urt ;
report.auc = auc ;

fprintf('block_vis: *********************************\n');
fprintf('block_vis: eer : %.2f%%\n', report.eer * 100);
fprintf('block_vis: auc : %.2f%%\n', report.auc * 100);
fprintf('block_vis: *********************************\n');
save(fullfile(wrd.prefix, bk.tag, 'report.mat'), ...
     '-STRUCT', 'report', '-MAT') ;

bk = bkend(bk) ;

% --------------------------------------------------------------------
function varargout = fetch__(bk, what, varargin)
% --------------------------------------------------------------------

global wrd ;

switch lower(what)
  case 'report'
    report = load(fullfile(wrd.prefix, bk.tag, 'report.mat'));
    varargout{1} = report;

  otherwise
    error(sprintf('Unknown ''%s''.', what)) ;
end


