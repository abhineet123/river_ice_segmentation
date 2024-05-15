function bk = block_aibdict(bk)
% BLOCK_AIBDICT  Use AIB block to create a compressed dictionary.
%   This block cuts an AIB tree to form a compact dictionary.
%
%   BK = BLOCK_AIBDICT() Initializes the block with the default
%   options.
%
%   BK = BLOCK_AIBDICT(BK) Executes the block with options and inputs
%   BK.
%
%   Required inputs:
%
%   dict::
%     A dictionary to be compressed, e.g. from BLOCK_DICTIONARY()
%
%   aib::
%     An instance of BLOCK_AIB()
%
%   Options:
%   
%   bk.nwords::
%     The size of the final dictionary. Default 40.
%
%   bk.discard_zero::
%     Throw away bins which had 0 observations during AIB? Default 1.
%
%   Fetchable attributes:
%
%   type::
%     Type of dictionary 'aib'
%
%   aibmap::
%     The mapping from the dictionary to the compressed dictionary
%
%   dict::
%     The dictionary, may also be retrieved with 'dictionary'
%
%   Block functions:
%
%   push::
%     [WORDS,HIST,SEL] = PUSH(DICT, DATA) pushes the data through the
%     dictionary. sel indexes which data items correspond to which
%     words.

% AUTORIGHTS
% Copyright (c) 2009 Brian Fulkerson and Andrea Vedaldi
% Blocks is distributed under the terms of the modified BSD license.
% The full license may be found in LICENSE.

global wrd ;

if nargin == 0
  bk = bkinit('aibdict', 'dict', 'aib') ;
  bk.fetch      = @fetch__ ;
  bk.push       = [] ;
  bk.nwords     = 40 ;
  bk.discard_zero = 1; % Throw away bins which had 0 observations during AIB
  return ;
end

% --------------------------------------------------------------------
%                                                      Virutal methods
% --------------------------------------------------------------------

bk.push = @push__ ;

function [w,h,sel] = push__(dict, d)

bkdict = bkfetch(bk.dict.tag) ;
dict   = bkfetch(bkdict, 'dictionary') ;
aibmap = bkfetch(bk, 'aibmap') ;

[w,h,sel]  = bkdict.push(dict, d) ;

w = aibmap(w);
if bk.discard_zero
    zsel = find(w);
    w = w(zsel);
    sel = sel(zsel);
    h = hist(w, 1:bk.nwords);
    h = h';
else
    h = hist(w, 0:bk.nwords);
    h = h';
end

end

% --------------------------------------------------------------------
%                                                              Do work
% --------------------------------------------------------------------

[bk, dirty] = bkbegin(bk) ;
if ~ dirty, return ; end

bkaib = bkfetch(bk.aib.tag);
parents = bkfetch(bkaib, 'tree');
[cut, aibmap, short] = vl_aibcut(parents, bk.nwords);

save(fullfile(wrd.prefix, bk.tag, 'data.mat'), 'aibmap', 'cut', 'short', '-MAT') ;

bk = bkend(bk) ;
end

% --------------------------------------------------------------------
function varargout = fetch__(bk, what, varargin)
% --------------------------------------------------------------------
global wrd ;

switch lower(what)
  case 'type'
    varargout{1} = 'aib' ;
  case 'aibmap'
    data = load(fullfile(wrd.prefix, bk.tag, 'data.mat')) ;
    varargout{1} = data.aibmap ;
  case {'dict', 'dictionary'}
    varargout{1} = bkfetch(bk.dict.tag, 'dictionary') ;
  otherwise
    error(sprintf('Unknown ''%s''.', what)) ;
end
end
