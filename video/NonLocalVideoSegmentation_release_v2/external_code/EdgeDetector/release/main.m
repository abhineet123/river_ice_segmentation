model=importdata('D:\video_cosegmentation\segmentation_code_011013\release\release\models\forest\modelFinal.mat');
model.opts.multiscale=1;          % for top accuracy set multiscale=1\
I = imread('peppers.png');
tic, E=edgesDetect(I,model); toc
figure(1); im(I); figure(2); im(1-E);

