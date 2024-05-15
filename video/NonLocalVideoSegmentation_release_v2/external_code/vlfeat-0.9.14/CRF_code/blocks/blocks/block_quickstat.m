function bk = block_quickstat(bk, varargin)
% BLOCK_QUICKSTAT Generate statistics about quick shift segmentations
%   This block computes statistics on the set of quick shift
%   segmentations.
% 
%   BK = BLOCK_QUICKSTAT() Initializes the block with the default
%   options.
%
%   BK = BLOCK_QUICKSTAT(BK) Executes the block with options and inputs
%   BK.
%
%   Required inputs:
%   
%   db::
%     The image database.
%
%   qseg::
%     The quick shift segmentations.
%
%   Fetchable attributes:
%
%   stats::
%     Statistics about each image. Returns a structure containing:
%     degrees:    A histogram of the degrees found in each image
%     counts:     A vector containing the size of all superpixels in
%                 the database.
%     segsperim:  The number of segments found in each image.

% AUTORIGHTS
% Copyright (c) 2009 Brian Fulkerson and Andrea Vedaldi
% Blocks is distributed under the terms of the modified BSD license.
% The full license may be found in LICENSE.

global wrd ;

if nargin == 0
  bk             = bkinit('quickstat', 'qseg', 'db') ;
  bk.fetch       = @fetch__ ;
  return ;
end

% --------------------------------------------------------------------
%                                                    Check/load inputs
% --------------------------------------------------------------------

[bk, dirty] = bkbegin(bk) ;
if ~ dirty, return ; end

db = bkfetch(bk.db.tag, 'db') ;

% --------------------------------------------------------------------
%                                                       Do computation
% --------------------------------------------------------------------

keys = 1:length(db.segs) ;

segsperim = zeros(length(keys), 1);

degrees = zeros(1000,1);
counts  = [];
for t=1:length(keys)

  seg_id = db.segs(keys(t)).seg ;
  
  segs = bkfetch(bk.qseg.tag, 'segs', seg_id);
  for i = 1:length(segs)
    d = length(segs(i).adj);
    if d > 1000, d = 1000; end
    degrees(d) = degrees(d) + 1;
  end
  counts = [counts segs.count];
  segsperim(t) = length(segs);
  
  fprintf('block_quickstat: %3.0f%% completed\n', t / length(keys) * 100) ;
end
save(fullfile(wrd.prefix, bk.tag, 'stats.mat'), ...
       'degrees', 'counts', 'segsperim', '-MAT') ;

bk = bkend(bk) ;

% --------------------------------------------------------------------
function varargout = fetch__(bk, what, varargin)
% --------------------------------------------------------------------

global wrd ;

path = fullfile(wrd.prefix, bk.tag, 'stats.mat') ;
data = load(path, '-MAT') ;

switch lower(what)

  case 'stats'
    varargout{1} = data;

  otherwise
    error(sprintf('Unknown ''%s''.', what)) ;
end


