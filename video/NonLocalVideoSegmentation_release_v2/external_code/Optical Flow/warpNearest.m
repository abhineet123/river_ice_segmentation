function [im_output,I]=warpNearest(im_input,vx,vy)
    if nargin==0
        im_input=[1 2 3;4 5 6;7 8 9];
        vx=zeros(size(im_input));
        vy=zeros(size(im_input));
        vx(2,2)=1;
    end
    % warp i1 according to flow field in vx vy
    [M,N]=size(im_input);
    [x,y]=meshgrid(1:N,1:M);
    im_output=interp2(x,y,im_input,x+vx,y+vy,'nearest');
    I=find(isnan(im_output));
    im_output(I)=zeros(size(I));
