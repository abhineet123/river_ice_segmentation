function stats=visloc(varargin)
% VISLOC Visualize localization
%   STATS=VISLOC(TAG)
%
%   [STATS1, STATS2, ...] = VISLOC(TAG1, TAG2, ...)

% AUTORIGHTS
% Copyright (c) 2009 Brian Fulkerson and Andrea Vedaldi
% Blocks is distributed under the terms of the modified BSD license.
% The full license may be found in LICENSE.

if nargin < 1
  varargin{1} = '.*' ;
end

if nargout == 0
  cla ; hold on ;  
  title('Recall-Precision') ;
end

cor = get(gca,'colororder') ;
h   =[] ;
leg ={} ;

j = 1 ;
for i=1:length(varargin)

  pattern  = ['^visloc@' varargin{i}] ;
  reports  = fetchreports(pattern) ;
  for r = 1:length(reports)
    fprintf('fetchreports: loaded file ''%s'' (eer: %.2f)%%.\n', ...
            reports{r}.file, reports{r}.eer*100) ;
  end

  if isempty(reports)
    warning(sprintf('No reports found for pattern ''%s''', pattern)) ;
    stats{i} = [] ;
  else
    stats{i} = reportstats(reports) ;
  end

  if nargout == 0 & ~isempty(stats{i})
    h(j) = plot(stats{i}.recall, stats{i}.precision);
    h2   = plot(stats{i}.recall, stats{i}.precision, ':');
    h3   = plot(stats{i}.recall, stats{i}.precision, ':');
     
    cl = cor(mod(i-1,size(cor,1))+1,:) ;  
    set(h(j),'color',cl) ;
    set(h2,  'color',cl) ;
    set(h3,  'color',cl) ;
    
    set(h(j),'linewidth',3) ;
   
    stats{i}.eer_min = stats{i}.eer;
    stats{i}.eer_max = stats{i}.eer;

    leg{j} = sprintf('%s (%.3g/%.3g/%.3g)', ...
                     varargin{i},           ...
                     stats{i}.eer_min,      ...
                     stats{i}.eer,          ...
                     stats{i}.eer_max       ) ;
    j = j + 1 ;
  end
end

if nargout == 0
  line([0 1], [1 0], 'color','b', 'linestyle', ':', 'linewidth', 2) ;
  line([0 1], [0 1], 'color','y', 'linestyle', '-', 'linewidth', 1) ;
  
  axis square ;
	xlim([0 1]) ; xlabel('Recall') ;
	ylim([0 1]) ; ylabel('Precision') ;
  
  legend_h = legend(h, leg{:}, 'Location', 'SouthWest') ;
  set(legend_h, 'Interpreter', 'None');
end



% --------------------------------------------------------------------
function stats = reportstats(reports)
% --------------------------------------------------------------------

stats.precision = reports{1}.precision;
stats.recall    = reports{1}.recall;
stats.eer       = reports{1}.eer;
