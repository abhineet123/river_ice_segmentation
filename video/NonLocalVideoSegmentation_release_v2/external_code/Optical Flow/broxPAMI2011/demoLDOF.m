im1 = double(imread('tennis492.ppm'));
im2 = double(imread('tennis493.ppm'));
flow = mex_LDOF(im1,im2);