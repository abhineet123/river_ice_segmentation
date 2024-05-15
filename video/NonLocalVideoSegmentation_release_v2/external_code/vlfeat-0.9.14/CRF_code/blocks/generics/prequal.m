function [eer precision recall] = prequal(tp, fp, tn, fn)
% PREQUAL Find the equal point of the precision recall curve

% AUTORIGHTS
% Copyright (c) 2009 Brian Fulkerson and Andrea Vedaldi
% Blocks is distributed under the terms of the modified BSD license.
% The full license may be found in LICENSE.

% Plot PR curve
truepos   = sum(tp, 2);
trueneg   = sum(tn, 2);
falsepos  = sum(fp, 2);
falseneg  = sum(fn, 2);

precision = truepos ./ (truepos + falsepos + eps);
recall    = truepos ./ (truepos + falseneg + eps);

precision(find(recall==0)) = 1;

% linear interpolation to find the equal error point
i1 = max(find(recall  >= precision));
i2 = min(find(precision >= recall));
if i1==i2
    eer = max(recall(i1), precision(i2));
    eer = (recall(i1) + precision(i1))/2;
else
    dy = precision(i2) - precision(i1);
    dx = recall(i2) - recall(i1);
    b  = precision(i1) - (dy/dx) * recall(i1);
    eer = b/(1-dy/dx);
end
