function bk = bkend(bk)
% BKEND End a block
%  BK = BKEND(BK) Ends the block by updating the completion timestamp
%  and saving the configuration.
%
%  See also: BKBEGIN(), BKINIT(), BKPLUG()

% AUTORIGHTS
% Copyright (c) 2009 Brian Fulkerson and Andrea Vedaldi
% Blocks is distributed under the terms of the modified BSD license.
% The full license may be found in LICENSE.

global wrd ;

bk.timestamp = now ;
save(fullfile(wrd.prefix, bk.tag, 'cfg.mat'), '-STRUCT', 'bk') ;
fprintf('block_%s: end %.3g hrs\n', bk.type, ...
        (bk.timestamp - bk.started) * 24) ;
