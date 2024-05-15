function sig = getintegralsig(integral, rs, cs, re, ce)
% GETINTEGRALSIG Get integral histogram signature
%   SIG = GETINTEGRALSIG(INTEGRAL, RS, CS, RE, CE) computes the
%   histogram of the window defined by (RS:RE,CS:CE) using integral
%   signatures INTEGRAL.

% AUTORIGHTS
% Copyright (c) 2009 Brian Fulkerson and Andrea Vedaldi
% Blocks is distributed under the terms of the modified BSD license.
% The full license may be found in LICENSE.

if rs ~= 0 && cs ~= 0
    sig = integral(re,ce,:) + integral(rs,cs,:) -  integral(rs,ce,:) - ...
    integral(re,cs,:);
elseif rs == 0 && cs == 0
    sig = integral(re,ce,:);
elseif rs == 0
    sig = integral(re,ce,:) - integral(re,cs,:);
else %if cs == 0
    sig = integral(re,ce,:) - integral(rs,ce,:);
end

sig = sig(:);
