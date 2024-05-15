function stats=visavg(varargin)
% VISAVG
%   STATS=VISAVG(TAG)
%
%   [STATS1, STATS2, ...] = VISAVG(TAG1, TAG2, ...)

global wrd ;

if length(nargin) < 1
  varargin = '.*' ;
end

if nargout == 0
  cla ; hold on ;  
  title('ROC (visavg)') ;
end

cor = get(gca,'colororder') ;
h   =[] ;
leg ={} ;

j = 1 ;
for i=1:length(varargin)

  pattern  = ['^vis@' varargin{i}] ;
  reports  = fetchreports(pattern) ;

  if isempty(reports)
    warning(sprintf('No reports found for pattern ''%s''', pattern)) ;
    stats{i} = [] ;
  else
    stats{i} = reportstats(reports) ;
  end

  if nargout == 0 & ~isempty(stats{i})
    h(j) = plot(stats{i}.tn,     stats{i}.tp        ) ;
    h2   = plot(stats{i}.tn_max, stats{i}.tp_max,':') ;
    h3   = plot(stats{i}.tn_min, stats{i}.tp_min,':') ;
    
    cl = cor(mod(i-1,size(cor,1))+1,:) ;  
    set(h(j),'color',cl) ;
    set(h2,  'color',cl) ;
    set(h3,  'color',cl) ;
    
    set(h(j),'linewidth',3) ;
    
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
	xlim([0 1]) ; xlabel('true negative') ;
	ylim([0 1]) ; ylabel('true positve') ;
  
  legend_h = legend(h, leg{:}, 'Location', 'SouthWest') ;
  set(legend_h, 'Interpreter', 'None');
end


% --------------------------------------------------------------------
function stats = reportstats(reports)
% --------------------------------------------------------------------

T   = 101 ;
R   = length(reports) ;
rho = zeros(T,R) ;

for r=1:R
  [tmp,th] = repr(reports{r}.tp,reports{r}.tn,T) ;
  rho(:,r) = tmp(:) ;  
end

% weighed logit transformation
method = 'minmax' ;
%method = 'logit' ;
switch method
  case 'logit'
    wgt = sqrt(min(tan(th),cot(th)).^2 + 1) + 1 / 50;
    wgt = wgt(:) ;

    rho_ = min(rho ./ wgt(:,ones(1,R)), 1) ;
    xi   = logit(rho_) ;
    mu   = mean(xi,2) ;
    va   = std(xi,0,2) ;
    
    a    = ilogit(mu         ) .* wgt ;
    b    = ilogit(mu + 2.5*va) .* wgt ;
    c    = ilogit(mu - 2.5*va) .* wgt ;
        
  case 'linear'    
    mu = mean(rho,2) ;
    va = std(rho,0,2) ;    
    a  = mu ;
    b  = mu+2.5*va ;
    c  = mu-2.5*va ;
    
  case 'minmax'
    b  = max(rho,[],2) ;
    c  = min(rho,[],2) ;
    a  = (b + c) / 2 ;
end

tp  = a' .* sin(th) ;
tn  = a' .* cos(th) ;
tp1 = b' .* sin(th) ;
tn1 = b' .* cos(th) ;
tp2 = c' .* sin(th) ;
tn2 = c' .* cos(th) ;

eer  = calc_eer(tp,tn) ;
eer1 = calc_eer(tp1,tn1) ;
eer2 = calc_eer(tp2,tn2) ;

stats.tp     = tp ;
stats.tn     = tn ;
stats.tp_max = tp1 ;
stats.tn_max = tn1 ;
stats.tp_min = tp2 ;
stats.tn_min = tn2 ;

stats.eer    = calc_eer(tp,tn) ; 
stats.eer_max= calc_eer(tp1,tn1) ;
stats.eer_min= calc_eer(tp2,tn2)  ;

% --------------------------------------------------------------------
function [rho,th] = repr(tp,tn,T)
% --------------------------------------------------------------------

if tp(1) < tp(end)
  tp = fliplr(tp(:)) ;
  tn = fliplr(tn(:)) ;
end

rho = zeros(1,T) ;
th  = linspace(pi/2,0,T) ;

for i=1:T

  t = tan(th(i)) ;
    
  k = max(find(tp >= t * tn)) ;
  
  tp1 = tp(k) ;
  tn1 = tn(k) ;
  
  if k < length(tp),
    tp2 = tp(k+1) ;
    tn2 = tn(k+1) ;
  else
    tp2 = 0 ;
    tn2 = 1 ;
  end

  dn = tn2 - tn1 ;
  dp = tp2 - tp1 ;
  
  A = [dp dn ; t -1] ;
    
  if rank(A) == 2
    z = A \ [ tn1 * dp - tp1 * dn ; 0 ] ;
  else
    z = 0.5 * [tn1+tn2 ; tp1+tp2] ;
  end
  rho(i) = norm(z) ;
end

% --------------------------------------------------------------------
function z = logit(p)
% --------------------------------------------------------------------
z = log(p ./ (1 - p)) ;

% --------------------------------------------------------------------
function p = ilogit(z)
% --------------------------------------------------------------------
p = 1 ./ (1 + exp(-z)) ;

% --------------------------------------------------------------------
function eer = calc_eer(tp,tn)
% --------------------------------------------------------------------
i1  = max(find(tp >= tn)) ;
i2  = min(find(tn >= tp)) ;
eer = max(tn(i1), tp(i2)) ;
