function bk = block_dict_sel(bk)
% BLOCK_DICT_SEL  Select a dictionary instance
%   This block selects a dictionary instance from the dictionary
%   training block.
%
%   BK = BLOCK_DICT_SEL() Initializes the block with the default
%   options.
%
%   BK = BLOCK_DICT_SEL(BK) Executes the block with options and inputs
%   BK.
%
%   Required Inputs:
%
%   dict::
%     The dictionary to fetch
%
%   Options:
%
%   bk.selection::
%     Which dictionary to select. Default 1.
%
%   Fetchable attributes:
%
%   type::
%     The type of the dictionary, retrieved from the input dictionary.
%
%   dict::
%     The selected dictionary.
%     
%   Block functions:
%
%   push::
%     [WORDS,HIST,SEL] = PUSH(DICT, DATA) pushes the data through the
%     selected dictionary. sel indexes which data items correspond to
%     which words.

global wrd ;

if nargin == 0
  bk = bkinit('dict_sel', 'dict') ;
  bk.fetch      = @fetch__ ;
  bk.push       = [] ;
  bk.selection  = 1 ;
  return ;
end

% --------------------------------------------------------------------
%                                                      Virutal methods
% --------------------------------------------------------------------

bk.push = @push__ ;

function [w,h,sel] = push__(dict, d)
bkdict = bkfetch(bk.dict.tag);
dict   = bkfetch(bkdict, 'dictionary', bk.selection) ;
[w,h,sel]  = bkdict.push(dict, d) ;
end

% --------------------------------------------------------------------
%                                                              Do work
% --------------------------------------------------------------------

[bk, dirty] = bkbegin(bk) ;
if ~ dirty, return ; end
bk = bkend(bk) ;
end

% --------------------------------------------------------------------
function varargout = fetch__(bk, what, varargin)
% --------------------------------------------------------------------
global wrd ;

switch lower(what)
  case 'type'
    varargout{1} = bkfetch(bk.dict.tag, 'type') ;
  case {'dict', 'dictionary'}
    varargout{1} = bkfetch(bk.dict.tag, 'dictionary', bk.selection) ;
  otherwise
    error(sprintf('Unknown ''%s''.', what)) ;
end
end
