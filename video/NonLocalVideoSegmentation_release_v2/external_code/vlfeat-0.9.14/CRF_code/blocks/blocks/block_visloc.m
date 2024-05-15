function bk = block_visloc(bk, varargin)
% BLOCK_VISLOC Visualize two class localization
%   Visualize two-class localization (such as Graz-02) and find the
%   precision=recall point.
%
%   BK = BLOCK_VISLOC() Initializes the block with the default
%   options.
%
%   BK = BLOCK_VISLOC(BK) Executes the block with options and inputs
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
%   bk.seg_prefix::
%     The prefix path to the ground truth segmentations (Required).
%
%   bk.seg_ext::
%     The extension of each ground truth segmentation (Required).
%
%   bk.cat_ids::
%     A mapping from category names in the database to values in the
%     ground truth segmentation images (Required).
%
%   bk.num_levels::
%     The number of thresholds at which to evaluate performance.
%     Default 40.
%
%   bk.levels::
%     The actual thresholds to evalutate performance. Default [] uses
%     num_levels to determine the thresholds automatically.
%
%   Fetchable attributes:
%
%   report::
%     A structure containing a report on the results which has fields:
%     seg_ids:    The training segment ids.
%     tp:         The raw true positives.
%     fp:         The raw false positives.
%     tn:         The raw true negatives.
%     fn:         The raw false negatives.
%     levels:     The levels chosen.
%     eer:        The precision = recall point.
%     precision:  Precision for each level.
%     recall:     Recall for each leve.
%
%   eer_level::
%     The precision = recall point.


global wrd ;

if nargin == 0
  bk = bkinit('visloc', 'db', 'prediction') ;
  bk.fetch = @fetch__ ;
  bk.verb  = 1 ;
  bk.seg_prefix = [];
  bk.seg_ext = [];
  bk.cat_ids = [];
  bk.num_levels = 40;
  bk.levels  = [];
  return ;
end

% --------------------------------------------------------------------
%                                                    Check/load inputs
% --------------------------------------------------------------------

[bk, dirty] = bkbegin(bk) ;
if ~ dirty, return ; end

if length(bk.seg_prefix) == 0, error('block_visloc: missing seg_prefix'); end;
if length(bk.seg_ext) == 0,    error('block_visloc: missing seg_ext'); end;
if length(bk.cat_ids) == 0,    error('block_visloc: missing cat_ids'); end;

db = bkfetch(bk.db.tag, 'db') ;

% --------------------------------------------------------------------
%                                                       Do computation
% --------------------------------------------------------------------

% Get the foreground id
for i = 1:length(db.cat_names)
    if strcmp(bk.fg_cat, db.cat_names{i})
        fg_id = db.cat_ids(i);
        break;
    end
end

%sel_test = find([db.segs.flag] == db.TEST) ;
% Use only foreground images
sel_test = find([db.segs.flag] == db.TEST & [db.segs.cat] == fg_id) ;

N = length(sel_test) ;

gt_ids = mapcats(db.cat_names, db.cat_ids, bk.cat_ids);

if length(bk.levels) == 0
    fprintf('block_visloc: scanning confidence levels\n');
    maxconf = -inf;
    for i = 1:N
        testi   = sel_test(i);
        seg_id  = db.segs(testi).seg;
        gt_name = gtname(bk.seg_prefix, bk.seg_ext, db.segs(testi).path);
        gt = imread(gt_name);
        [class confidence] = bkfetch(bk.prediction.tag, 'test', seg_id);

        maxconf = max(max(confidence(:)), maxconf);

        % resize image
        % adjust threshold through range -2 - 2
        % record tp fp tn fn
    end
    fprintf('block_visloc: maxconf %f\n', maxconf);
    levels = linspace(-maxconf, maxconf, bk.num_levels);
else
    levels = bk.levels;
end

tp = zeros(length(levels), N);
fp = zeros(size(tp));
tn = zeros(size(tp));
fn = zeros(size(tp));
seg_ids = zeros(N,1);

gt_fg_id = [];
for i = 1:length(bk.cat_ids)
    if strcmp(bk.fg_cat, bk.cat_ids{i}{1})
        gt_fg_id = bk.cat_ids{i}{2};
        break;
    end
end
if length(gt_fg_id) == 0
    error('block_visloc: %s not in cat_ids', bk.fg_cat); 
end


for i = 1:N
    testi   = sel_test(i);
    seg_id  = db.segs(testi).seg;
    seg_ids(i) = seg_id;
    gt_name = gtname(bk.seg_prefix, bk.seg_ext, db.segs(testi).path);
    gt = imread(gt_name);
    [class conf] = bkfetch(bk.prediction.tag, 'test', seg_id);

    % This part downsamples the ground truth because:
    %   can't interpolate confidence if it is absolute value
    %   can't interpolate class labels (nearest?)
    labels = unique(gt);
    gt = imresize(gt, size(conf), 'nearest');
    newlabels = unique(gt);
    if length(labels) ~= length(newlabels) || any(labels ~= newlabels)
        error('block_visloc: Resizing is broken for %s\n', gt_name);
    end

    gt = (gt == gt_fg_id); % everything not fg is treated as background

    % extend to multiclass?
    for l = 1:length(levels)
        level = levels(l);
        invert = level > 0;
        level = abs(level);
        if invert
            classification = (conf > level) & (class == fg_id);
        else
            classification = ((conf < level) & (class ~= fg_id)) | ...
                             (class == fg_id); 
        end
        tp(l,i) = length(find( classification &  gt));
        fp(l,i) = length(find( classification & ~gt));
        tn(l,i) = length(find(~classification & ~gt));
        fn(l,i) = length(find(~classification &  gt));
    end
    fprintf('block_visloc: %d/%d\r', i, N);

end

report.seg_ids = seg_ids;
report.tp = tp;
report.fp = fp;
report.tn = tn;
report.fn = fn;
report.levels = levels;
[report.eer report.precision report.recall] = prequal(tp, fp, tn, fn);

if bk.verb
    figure(20); clf;
    subplot(2,2,1);
    plot(report.levels, sum(report.tp, 2));
    hold on;
    plot([0 0], [0 max(sum(report.tp,2))], 'r');
    title('True Positive');

    subplot(2,2,2);
    plot(report.levels, sum(report.fn, 2));
    hold on;
    plot([0 0], [0 max(sum(report.fn,2))], 'r');
    title('False Negative');

    subplot(2,2,3);
    plot(report.levels, sum(report.fp, 2));
    hold on;
    plot([0 0], [0 max(sum(report.fp,2))], 'r');
    title('False Positive');

    subplot(2,2,4);
    plot(report.levels, sum(report.tn, 2));
    hold on;
    plot([0 0], [0 max(sum(report.tn,2))], 'r');
    title('True Negative');

    figure(21);
    clf
    [eer precision recall] = prequal(report.tp, report.fp, report.tn, report.fn);
    plot(recall, precision);
    hold on;
    plot([0 1], [0, 1], 'y-');
    plot(eer, eer, 'r+');
    title('Recall-Precision');
    xlabel('Recall'); ylabel('Precision');
    legend(sprintf('%.2f%% eer', eer*100));
    axis square
end

fprintf('block_visloc: *********************************\n');
eer = prequal(report.tp, report.fp, report.tn, report.fn);
fprintf('block_visloc: precision = recall : %.2f%%\n', eer*100);
fprintf('block_visloc: *********************************\n');
save(fullfile(wrd.prefix, bk.tag, 'report.mat'), ...
     '-STRUCT', 'report', '-MAT') ;

bk = bkend(bk) ;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function gt_name = gtname(gt_prefix, gt_ext, imname);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[pathstr,name,ext] = fileparts(imname);
gt_name = fullfile(gt_prefix, pathstr, [name '.' gt_ext]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function gt_ids = mapcats(cat_names, cat_ids, gt_cats)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

gt_ids = zeros(size(cat_ids));
for i = 1:length(cat_names)
    found = 0;
    for j = 1:length(gt_cats)
        if strcmpi(cat_names{i}, gt_cats{j}{1})
            gt_ids(i) = gt_cats{j}{2};
            found = 1;
            break;
        end
    end
    if ~found
        error('block_visloc: %s not found in ground truth', cat_names{i});
    end
end

% --------------------------------------------------------------------
function varargout = fetch__(bk, what, varargin)
% --------------------------------------------------------------------

global wrd ;

switch lower(what)
  case 'report'
    path = fullfile(wrd.prefix, bk.tag, 'report.mat');
    varargout{1} = load(path, '-MAT');
  case 'eer_level'
    report = bkfetch(bk, 'report');
    recall = report.recall;
    precision = report.precision;

    % linear interpolation to find the equal error point
    i1 = max(find(recall  >= precision));
    i2 = min(find(precision >= recall));
    if i1==i2
      eer_level = (report.levels(i1) + report.levels(i2))/2;
    else
      delta = (report.eer - precision(i1))/(precision(i2) - precision(i1));
      eer_level = report.levels(i1) + (report.levels(i2) - report.levels(i1))*delta;
    end
    varargout{1} = eer_level;

  otherwise
    error(sprintf('Unknown ''%s''.', what)) ;
end


