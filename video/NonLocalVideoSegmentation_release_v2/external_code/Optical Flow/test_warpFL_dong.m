function test_warpFL_dong()
    im1=imread('D:\Data\SegTrack\Dataset\birdfall2\birdfall2_00018.png');
    im2=imread('D:\Data\SegTrack\Dataset\birdfall2\birdfall2_00019.png');
    load('D:\Data\SegTrack\Optical Flows\birdfall2\birdfall2_00018_to_birdfall2_00019.opticalflow(Ce Liu).mat');
    warpI2=warpByOpticalFlow_dong(im2,vx,vy);
    close all;
    figure;
    subplot(131);
    imshow(im1(70:110,60:100,:));
    subplot(132);
    imshow(im2(70:110,60:100,:));
    subplot(133);
    imshow(warpI2(70:110,60:100,:));