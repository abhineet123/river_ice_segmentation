function bk = bkinit(type, varargin)
% BKINIT  Initialize a block
%   BK = BKINIT('TYPE') initialize a block of type TYPE with no
%   required inputs. Inputs may still be connected with BKPLUG().
%
%   BK = BKINIT('TYPE', 'I1','I2') initializes a block with required
%   inputs named I1, I2. To connect blocks you must still call
%   BKPLUG() for each input.
%
%   See also: BKBEGIN(), BKEND(), BKPLUG()

% AUTORIGHTS
% Copyright (c) 2009 Brian Fulkerson and Andrea Vedaldi
% Blocks is distributed under the terms of the modified BSD license.
% The full license may be found in LICENSE.

bk.type      = type ;
bk.tag       = '' ;
bk.timestamp = NaN ;
bk.started   = NaN ;
bk.inputs    = {} ;

for k=1:length(varargin)
  input = varargin{k} ;
  bk.inputs{end+1}     = input ;
  bk.(input).tag       = '' ;
  bk.(input).timestamp = 0 ;
end
