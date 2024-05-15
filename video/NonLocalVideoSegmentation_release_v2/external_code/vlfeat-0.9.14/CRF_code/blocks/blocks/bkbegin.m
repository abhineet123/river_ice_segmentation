function [bk, dirty] = bkbegin(bk)
% BKBEGIN  Begin a block
%   [BK, DIRTY] = BKBEGIN(BK) Begins a block BK by first checking if
%   it exists. If the block exists, it checks if any input blocks have
%   changed since this block was last executed. It also checks if any
%   input parameters have been changed, added, or removed. If any of
%   these conditions is true, DIRTY is set and the timestamp on the
%   block is updated. BK contains the modified block.
%
%   See also: BKINIT(), BKEND(), BKPLUG()

% AUTORIGHTS
% Copyright (c) 2009 Brian Fulkerson and Andrea Vedaldi
% Blocks is distributed under the terms of the modified BSD license.
% The full license may be found in LICENSE.

global wrd ;

if ~isfield(wrd, 'bless_all'), wrd.bless_all = 0 ; end
if ~isfield(wrd, 'pretend'),   wrd.pretend   = 0 ; end

bk_file = fullfile(wrd.prefix, bk.tag, 'cfg.mat') ;
type    = bk.type ;
dirty   = 0 ;

% Check if we are running in `pretend' mode
if wrd.pretend
  fprintf('block_%s: Pretending [%s]\n', type, bk.tag) ;      
  return ;
end

% Check if we are running in `bless' mode
if wrd.bless_all && exist(bk_file, 'file')
  fprintf('block_%s: Blessing [%s]\n', type, bk.tag) ;      
  bk = bkbless(bk) ;
  return ;
end

ensuredir(fullfile(wrd.prefix, bk.tag)) ;

% --------------------------------------------------------------------
%                                   Check if configuration has changed
% --------------------------------------------------------------------

if exist(bk_file, 'file')
  try
    bk_old = load(bk_file, '-MAT') ;
  catch
    fprintf('block_%s: configuration corrputed!\n', type) ;
    bk_old = struct ;
    dirty = 1 ;
  end

  if ~ xdiff(type, bk, bk_old)   
    fprintf('block_%s: configuration changed!\n', type) ;
    dirty = 1 ;
  else
    % Since nothing changed, use old config. This preserves started,
    % ended and timestamp.
    bk = bk_old ;
  end
  
else  
  fprintf('block_%s: configuration changed (was void)!\n', type) ;
  dirty = 1 ;
end

% --------------------------------------------------------------------
%                                         Check if inputs have changed
% --------------------------------------------------------------------

for i = 1:length(bk.inputs)
  in_name    = bk.inputs{i} ;
  in_tag     = bk.(in_name).tag ;
  
  % unassigned, but required? stop
  if isempty(in_tag)
    error(sprintf('block_%s: input ''%s'' missing!', type, in_name));
    continue ; 
  end
  
  in         = bkfetch(in_tag) ;
  if in.timestamp > bk.(in_name).timestamp
    fprintf('block_%s: input ''%s'' changed!\n', ...
            type, in_name) ;
    dirty = 1 ;
  end
  bk.(in_name).timestamp = in.timestamp ;
end

if ~dirty
  fprintf('block_%s Up-to-date [%s] %.3g hrs\n', [bk.type ':'], bk.tag, ...
          (bk.timestamp - bk.started) * 24) ;
else
  fprintf('block_%s Needs update [%s] \n', [bk.type ':'], bk.tag) ;
  bk.started = now ;
end

% --------------------------------------------------------------------
function eq = xdiff(type,a,b,path)
% --------------------------------------------------------------------
% XDIFF(A,B) compares configurations A and B. The function ignores the
% irrelevant fields (timestamp, split, started), NaN and also function
% handles.

if nargin < 4
  path = '' ;
end

eq = 1 ;

if ~strcmp(class(a), class(b))
  fprintf('block_%s: ***  Data type of ''%s'' differs\n', type, path) ;
  eq = 0 ;
  return ;
end

switch class(a)
  
  case 'function_handle'
    if ~ strcmp(func2str(a), func2str(b))
      fprintf('block_%s: ***  Function handles of ''%s'' differs\n', ...
              type, path) ;
      eq = 0 ;
      return ;
    end
    
  case 'struct'
    fields = fieldnames(a) ;
    for i=1:length(fields)
      n = fields{i} ;
      
      if strcmp(n, 'timestamp'), continue ; end
      if strcmp(n, 'split'),     continue ; end
      if strcmp(n, 'started'),   continue ; end
      
      xpath = sprintf('%s.%s', path, n) ;
      
      if ~isfield(b, n)
        fprintf('block_%s: *** OLD config is missing the field ''%s''\n', ...
                type, xpath) ;
        eq = 0 ;
      else
        eq = eq & xdiff(type, a.(n), b.(n), xpath) ;
      end
      
    end
    
    fields = fieldnames(b) ;
    for i=1:length(fields)
      n = fields{i} ;
      if ~isfield(a, n)
        xpath = sprintf('%s.%s', path, n) ;
        fprintf('block_%s: *** NEW config is missing the field ''%s''\n', ...
                type, xpath) ;
        eq = 0 ;
      end
    end
    
  otherwise
    eq = eq & isequalwithequalnans(a, b) ;
    if ~eq
      fprintf('block_%s: *** Value of ''%s'' differs (%s --> %s)\n', ...
              type, path, tostr(b), tostr(a)) ;     
    end
end
