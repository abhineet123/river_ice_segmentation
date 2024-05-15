function bk = block_visvoc(bk, varargin)
% BLOCK_VISVOC Create VOC segmentation submission compatible tgz files.
%   Create VOC segmentation submission compatible tgz files.
%
%   BK = BLOCK_VISVOC() Initializes the block with the default
%   options.
%
%   BK = BLOCK_VISVOC(BK) Executes the block with options and inputs
%   BK.
%
%   Required input:
%
%   db::
%     The database.
%
%   prediction::
%     The output of the localization classifier.
%
%   Options:
%   
%   bk.challenge::
%     The challenge which we are participating in.
%
%   bk.writeonly
%     Should we also compute a report about the accuracy? Default 1.
%
%   Fetchable attributes:
%
%   report::
%     Only valid if bk.writeonly was 0. Uses the VOC code to prodce
%     accuracies and returns a structure with:
%     accuracies: The accuracies for each category
%     conf:       The confusion matrix
%     cat_names:  The category names

% AUTORIGHTS
% Copyright (c) 2009 Brian Fulkerson and Andrea Vedaldi
% Blocks is distributed under the terms of the modified BSD license.
% The full license may be found in LICENSE.

global wrd ;

if nargin == 0
  bk = bkinit('visvoc', 'db', 'prediction') ;
  bk.fetch = @fetch__ ;
  bk.verb  = 1 ;
  bk.writeonly = 1;
  return ;
end

% --------------------------------------------------------------------
%                                                    Check/load inputs
% --------------------------------------------------------------------
% Write test images

[bk, dirty] = bkbegin(bk) ;
if ~ dirty, return ; end

db = bkfetch(bk.db.tag, 'db') ;
testsel = find([db.segs.flag] == db.TEST);

restag = bkver(bk.tag);
bkpred = bkfetch(bk.prediction.tag);

rpath = sprintf('results/VOC2009/Segmentation/comp5_%s_cls', bk.challenge);
resdir = fullfile(wrd.prefix, bk.tag, rpath);
ensuredir(resdir) ;
pathstr = fullfile(resdir, '%s.png');

voccode = fullfile(db.VOCopts.datadir, 'VOCcode');
addpath(voccode);
map = VOClabelcolormap(256);
rmpath(voccode);

fprintf('block_visvoc: Writing class images\n');
switched = 0;
for i = 1:length(testsel)
  seg_id = testsel(i);
  seg = db.segs(testsel(i));
  [class confidence] = bkfetch(bkpred, 'test', seg_id);

  [path name ext] = fileparts(seg.path);
  destloc = sprintf(pathstr, name);
  class = db.class_ids(class);
  imwrite(uint8(class), map, destloc);
  fprintf('block_visvoc: %d/%d\r', i, length(testsel));
end
fprintf('block_visvoc: switched %d pixels\n', switched);
fprintf('block_visvoc: VOCevalseg\n');

prefix = fullfile(wrd.prefix, bk.tag);
f = fopen(fullfile(prefix, 'results', 'README.txt'), 'w');
fprintf(f, '%s\n', bk.tag);
fclose(f);
[status, result] = system(sprintf('tar --directory %s -czf %s/results.tgz results', prefix, prefix));
if status
  error(sprintf('block_visvoc: result code was %d (%s)', status, result));
end

if ~bk.writeonly
  voccode = fullfile(db.VOCopts.datadir, 'VOCcode');
  addpath(voccode);
  [accuracies, avgacc, conf] = VOCevalseg(db.VOCopts, restag);
  rmpath(voccode);

  report = struct();
  report.accuracies = [accuracies(2:end); accuracies(1)]; % move bg to the end
  report.avgacc = avgacc;
  report.conf = conf;
  report.cat_names = db.cat_names;

  fprintf('block_visvoc: *********************************\n');
  fprintf('Accuracy: ');
  fprintf('%.2f ', accuracies);
  fprintf('\nAverage: %.2f\n', avgacc);
  fprintf('block_visvoc: *********************************\n');
  save(fullfile(wrd.prefix, bk.tag, 'report.mat'), ...
      '-STRUCT', 'report', '-MAT') ;
end

bk = bkend(bk) ;

% --------------------------------------------------------------------
function varargout = fetch__(name, what, varargin)
% --------------------------------------------------------------------

global wrd ;

switch lower(what)
  case 'report'
    report = load(fullfile(wrd.prefix, bk.tag, 'report.mat'));
    varargout{1} = report;

  otherwise
    error(sprintf('Unknown ''%s''.', what)) ;
end


