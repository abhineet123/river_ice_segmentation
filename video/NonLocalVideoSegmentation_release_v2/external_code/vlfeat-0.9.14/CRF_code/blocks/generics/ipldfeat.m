function [f,d] = ipldfeat(I, varargin)
% IPLDFEAT  SIFT IPLD impelmentation interface
%  [F,D] = IPLDFEAT(I) uses Dorko's IPLD SIFT code to compute
%  frames F and descriptors D.
%
%  Options:
%  
%  Detector ['DoG']::
%    Detector type (DoG, Harris, Hessian, LoG).
%
%  Angle [0]::
%    Do oriented keypoints.
%
%  Adapt [0] (incomplete and untested!!)::
%    Do affine adaptation (implies 'Angle')
%
%  Frames [[]]::
%    Compute descriptors of these frames. Due to a limitation of IPLD
%    interface, no angle can be specified.
%
%  Magnif [[3]]::
%    SIFT descriptor magnification factor. This parameter is
%    compatible with VLFeat SIFT. For a frame of scale 1, Magnif=3
%    implies that the size of a spatial bin is 3 pixels. Since there
%    are four of those along each dimension, this corresponds to a
%    patch sioze of 12.
%
%  Verbosity [0]
%    Verbosity level.

% AUTORIGHTS
% Copyright (c) 2009 Brian Fulkerson and Andrea Vedaldi
% Blocks is distributed under the terms of the modified BSD license.
% The full license may be found in LICENSE.

detect_type = 'dog' ;
descr_type  = 'sift' ;
do_affine   = 0 ;
do_angle    = 0 ;
verb        = 0 ;
nodups      = 1 ;
magnif      = 3 ;
frames      = [] ;

for k=1:2:length(varargin)
  opt=varargin{k} ;
  arg=varargin{k+1}; 
  switch lower(opt)
    case 'detector'
      switch lower(arg)
        case 'dog'          
          detect_type = 'dog' ;
        case 'harris'
          detect_type = 'harris' ;
        case 'hessian'
          detect_type = 'hessian' ;
        case 'log'
          detect_type = 'log' ;
        otherwise
          error(sprintf('Unknown detector ''%s''', arg)) ;
      end
      
    case 'adapt'
      do_affine = arg ;
      
    case 'angle'
      do_angle = arg ;
      
    case 'magnif'
      magnif = arg ;
      
    case 'verbosity'
      verb = arg ;
      
    case 'frames'
      frames = arg ;
      
    otherwise
      error(sprintf('Unknown option ''%s''', opt)) ;
  end
end

if do_affine, do_angle = 1 ; end

% -----------------------------------------------------------------------
%                                                                Do stuff
% -----------------------------------------------------------------------

if size(I,3) > 1
  I = rgb2gray(I) ; 
end

ipldroot      = '/share/ipld' ;
ipld_detect   = fullfile(ipldroot, 'Detect') ;
ipld_descr    = fullfile(ipldroot, 'ComputeDescriptor') ;
ipld_convert  = fullfile(ipldroot, 'corners2text') ;
ipld_iconvert = fullfile(ipldroot, 'text2corners') ;
ipld_dump     = fullfile(ipldroot, 'dumpcontents') ;
tmpdir        = '~/tmp' ;

cl = clock ;
cl(end) = cl(end) * 1000000 ;
[st,hst] = system('hostname') ;
pid = vl_getpid ;
sfx = sprintf('-%.0f-%s-%d', cl(end), strtrim(hst), pid) ;
i_name = fullfile(tmpdir, ['ipldtemp'  sfx '.pgm' ]) ;
k_name = fullfile(tmpdir, ['ipldtemp'  sfx '.key' ]) ;
d_name = fullfile(tmpdir, ['ipldtemp'  sfx '.desc']) ;
t_name = fullfile(tmpdir, ['ipldtemp'  sfx '.txt' ]) ;
a_name = fullfile(tmpdir, ['ipldtemp'  sfx '.angl']) ;

if verb 
  fprintf('ipldfeat: detector type   : ''%s''\n', detect_type  ) ;
  fprintf('ipldfeat: descriptor type : ''%s''\n', descr_type   ) ;
  fprintf('ipldfeat: descriptor type : ''%s''\n', descr_type   ) ;  
  fprintf('ipldfeat: do angle        : %d\n',     do_angle     ) ;
  fprintf('ipldfeat: do adaptation   : %d\n',     do_affine    ) ;  
end

if verb > 1
  fprintf('ipldfeat: detector   : ''%s''\n', ipld_detect  ) ;
  fprintf('ipldfeat: descriptor : ''%s''\n', ipld_descr   ) ;
  fprintf('ipldfeat: convert    : ''%s''\n', ipld_convert ) ;
  fprintf('ipldfeat: iconvert   : ''%s''\n', ipld_iconvert) ;
  fprintf('ipldfeat: tmp dir    : ''%s''\n', tmpdir       ) ;
  fprintf('ipldfeat: image name : ''%s''\n', i_name       ) ;
  fprintf('ipldfeat: key name   : ''%s''\n', k_name       ) ;
  fprintf('ipldfeat: txt name   : ''%s''\n', t_name       ) ;
  fprintf('ipldfeat: desc name  : ''%s''\n', d_name       ) ;
  fprintf('ipldfeat: angl name  : ''%s''\n', a_name       ) ;
end


% --------------------------------------------------------------------
%                                                   Save image on disk
% --------------------------------------------------------------------

imwrite(I, i_name, 'PGM', 'Encoding','rawbits');

% fix for picky IPLD parser
fix(i_name) ;

% --------------------------------------------------------------------
%                                                         Run detector
% --------------------------------------------------------------------

if isempty(frames)
  
  if do_affine
    aff_opt = '-aff' ;
  else
    aff_opt ='' ;
  end
  
  if do_angle
    ang_opt = '-angle' ;
  else
    ang_opt ='' ;
  end
  
  syscall = sprintf('%s %s %s -dtype %s %s %s', ...            
                    ipld_detect, ... 
                    aff_opt, ...
                    ang_opt, ...
                    detect_type, ...
                    i_name, ...
                    k_name) ;
  [st,rs] = system(syscall) ;
  
  if verb
    fprintf('ipldfeat: running ''%s''\n', syscall) ;
  end
  
  if st | verb
    fprintf(rs) ;
  end

  if st
    error('Error in Detect') ;
  end
  
else
  % save manually specified frames, adjusted to our
  % format
  
  if size(frames,1) ~= 3 & size(frames,1) ~= 5
    error('Frame format not supported') ;
  end
  
  pos  = frames(1:2,:) - 1 ;
  scal = frames(3,:) ;
  
  if size(frames,1) == 5
    aff = frames(3:5,:) ./ scal([1 1 1],:) ;
    aff_opt = ' -addaffine copy ' ;
  else
    aff = [] ;
    aff_opt = '' ;
  end
   
  out = [pos ; scal ; zeros(1,numel(scal)) ; aff]' ;
  save(t_name, 'out', '-ASCII') ;
  
  syscall = sprintf('%s %s %s %s', ...            
                    ipld_iconvert, ...           
                    aff_opt, ...
                    t_name, ...
                    k_name) ;

  [st,rs] = system(syscall) ;
  
  if verb
    fprintf('ipldfeat: running ''%s''\n', syscall) ;
  end
  
  if st | verb
    fprintf(rs) ;
  end

  if st
    error('Error in text2corner') ;
  end

end


% --------------------------------------------------------------------
%                                                       Run descriptor
% --------------------------------------------------------------------
% -siftscone default is 12 (our magnif is 3, with 2 spatial bins =
%            6)
% matches vlfeat a bit better with siftscone 17

syscall = sprintf('%s -siftscone %f -dtype %s %s %s %s', ...
                  ipld_descr, ...
                  magnif / 3 * 12, ...
                  descr_type, ...
                  i_name, ...
                  k_name, ...
                  d_name) ;
[st,rs] = system(syscall) ;

if verb
  fprintf('ipldfeat: running ''%s''\n', syscall) ;
end

if st | verb
  fprintf(rs) ;
end

if st
  error('Error in ComputeDescriptor') ;
end

% --------------------------------------------------------------------
%                                                              Convert
% --------------------------------------------------------------------

syscall = sprintf('%s -addaffine -adddesc %s %s', ...
                  ipld_convert, ...
                  d_name, ...
                  t_name) ;
[st,rs] = system(syscall) ;

if verb
  fprintf('ipldfeat: running ''%s''\n', syscall) ;
end

if st | verb
  fprintf(rs) ;
end

if st
  error('Error in corners2text') ;
end


syscall = sprintf('%s %s | sed -n -e ''s\\Angle: \\\\gp;'' > %s', ...
                  ipld_dump, ...
                  d_name, ...
                  a_name) ;
[st,rs] = system(syscall) ;

if verb
  fprintf('ipldfeat: running ''%s''\n', syscall) ;
end

if st | verb
  fprintf(rs) ;
end

if st
  error('Error in dumpcontents') ;
end

% --------------------------------------------------------------------
%                                                         Read results
% --------------------------------------------------------------------

data = load(t_name, '-ASCII') ;

if isempty(data)
  f = [] ;
  d = [] ;
  return ;
end

angl = load(a_name, '-ASCII')' ;
angl = pi + angl ;

pos  = data(:,1:2)' + 1 ;
scal = data(:,3)' ;
aff  = data(:,5:7)' ; 
d    = data(:,8:end)' ;
  
% SIFT descriptor to our format
d = uint8(d * 512) ;
d_ = d ;
p=[1 2 3 4 5 6 7 8] ;
%q=[1 8 7 6 5 4 3 2] ;
%q=[5 4 3 2 1 8 7 6] ; % -old
q = p ;

for j=0:3
  for i=0:3
    d(8*(i+4*j)+p,:) = d_(8*((3-i)+4*(3-j))+q,:) ;
  end
end

% frames to our format
if do_affine
  f = [pos ; aff(1,:) .* scal ; aff(2,:) .* scal ; aff(3,:) .* scal] ;
else
  f = [pos ; scal ; angl] ;
end

delete(i_name) ;
delete(k_name) ;
delete(d_name) ;
delete(t_name) ;
delete(a_name) ;

% --------------------------------------------------------------------
%                                                          Remove dups
% --------------------------------------------------------------------

if nodups
  if verb
    fprintf('ipldfeat: removing duplicates ...\n') ;
  end

  [f,sel] = unique(f','rows') ;
  f = f' ;
  d = d(:,sel) ;
  
  if verb,
    fprintf('ipldfeat: %d frames remain\n', size(f,2)) ;
  end
end

% --------------------------------------------------------------------
function fix(s1)
% --------------------------------------------------------------------
f	= fopen(s1,'r');
a	= fread(f,inf);
fclose(f);
id	= find(a==32);
a(id(1))	= 10;
a(id(3))	= 10;
f	= fopen(s1,'w');
fwrite(f,a);
fclose(f);
