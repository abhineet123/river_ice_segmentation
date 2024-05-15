function stats=vismulti(varargin)
% VISMULTI Visualize multiple category localization
%   STATS=VISMULTI(TAG)
%
%   [STATS1, STATS2, ...] = VISMULTI(TAG1, TAG2, ...)

% AUTORIGHTS
% Copyright (c) 2009 Brian Fulkerson and Andrea Vedaldi
% Blocks is distributed under the terms of the modified BSD license.
% The full license may be found in LICENSE.

if nargin < 1
  varargin{1} = '.*' ;
end

iu = 0;

stats = {};
j = 1 ;
for i=1:length(varargin)

  if strcmp(varargin{i}, 'iu')
    iu = 1;
    continue;
  end

  pattern  = ['^vismulti@' varargin{i}] ;
  reports  = fetchreports(pattern) ;

  if isempty(reports)
    warning(sprintf('No reports found for pattern ''%s''', pattern)) ;
    stats{i} = [] ;
  else
    if iu
    stats{i}.acc = zeros(length(reports), length(reports{1}.accuracies_int));
    else
    stats{i}.acc = zeros(length(reports), length(reports{1}.accuracies));
    end
    stats{i}.cat_names = {};
    stats{i}.pixels = zeros(length(reports),1);
    for r = 1:length(reports)
      stats{i}.cat_names{r} = reports{r}.cat_names;
      if iu
      stats{i}.acc(r,:) = reports{r}.accuracies_int;
      else
      stats{i}.acc(r,:) = reports{r}.accuracies;
      end
      stats{i}.pixels(r) = reports{r}.pixelscorrect;
    end
  end
  fprintf('\n');
  for zz = 1:length(reports)
    fprintf('vismulti: %s\n', reports{zz}.file);
    for c = 1:length(stats{i}.cat_names{zz})
      cat_name = stats{i}.cat_names{zz}{c};
      cat_name = cat_name(1:min(5,length(cat_name)));
      for sp = 1:(5-length(cat_name))
        fprintf(' ');
      end
      fprintf('  %s', cat_name);
    end
    fprintf('   mean');
    fprintf('\n');
    fprintf('& %4.0f ', stats{i}.acc(zz,:));
    fprintf('& %4.0f ', mean(stats{i}.acc(zz,:)));
    fprintf('\n');
    fprintf('Pixel %4.0f ', stats{i}.pixels(zz));
    fprintf('\n');
  end

end
