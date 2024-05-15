function varargout = bkfetch(bk, varargin)
% BKFETCH Fetch data from a block
%  BK = BKFETCH(BK) returns BK unchanged.
%
%  BK = BKFETCH(TAG) returns the block BK corresponding to TAG.
%
%  VAL = BKFETCH(BK, PROP) loads the property PROP of the block BK.
%  The properties that can be fetched depend on the block type.
%  BKFETCH(TAG, PROP) also loads the property PROP, but first reads
%  the block from disk.
%
%  BKFETCH(BK, PROP, ARG1, ARG2, ...) passes additional arguments to
%  the block to specify the property to be fetched.
%
%  [VAL1, VAL2, ...] = BKFETCH(...) is used when multiple values need
%  to be returned.

% AUTORIGHTS
% Copyright (c) 2009 Brian Fulkerson and Andrea Vedaldi
% Blocks is distributed under the terms of the modified BSD license.
% The full license may be found in LICENSE.

global wrd ;

if isstruct(bk) & ~isfield(bk, 'tag')
  error('BK structure malformed (has no .TAG field)') ;
end

if ischar(bk) & isempty(bk)
  error('TAG name empty') ;
end

if length(varargin) == 0
  if isstruct(bk)
    varargout{1} = bk ;
  else
    file = fullfile(wrd.prefix, bktag(bk), 'cfg.mat') ;
    if ~exist(file, 'file')
      error(sprintf('Block %s does not exist', bktag(bk)));
    end
    varargout{1} = load(file, '-MAT') ;
  end
else
  if isstr(bk)
    bk = bkfetch(bk) ;
  end
  [varargout{1:nargout}] = bk.fetch(bk, varargin{:}) ;
end
