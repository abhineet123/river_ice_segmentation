function bk = bkbless(bk)
% BKBLESS  Forces a block to be up-to-date
%  BK = BKBLESS(BK) "blesses" a block BK, forcing it to be up-to-date
%  if it exists. A call to BKBEGIN(BK) on the returned block should
%  not mark the block as dirty.

% AUTORIGHTS
% Copyright (c) 2009 Brian Fulkerson and Andrea Vedaldi
% Blocks is distributed under the terms of the modified BSD license.
% The full license may be found in LICENSE.

global wrd ;

% read current config
tag     = bktag(bk) ;
bk_old  = bkfetch(tag) ;

% now scan the input to update timestamps
for i=1:length(bk.inputs)  
  in_name    = bk.inputs{i} ;
  in_tag     = bk.(in_name).tag ;
  in         = bkfetch(in_tag) ;
  bk.(in_name).timestamp = in.timestamp ;
end

bk.timestamp = bk_old.timestamp ;
bk.started   = bk_old.started ;

save(fullfile(wrd.prefix, bk.tag, 'cfg.mat'), '-STRUCT', 'bk') ;
