% Train SVM with digital images and binary solutions
%


clc
clear
close all
k = 0;
% 
SVM_Vector = zeros(10000000,1);
SVM_Matrix = zeros(10000000, 81);
features = zeros(3,11);

FName = 'SVM Train_';
filetype = '.tif';
 
w = waitbar(0, 'My Progress Bar');

cd 'C:\Users\user1\Documents\Thesis\SVM Training\SVM Training Images'


 
for No = 1:225
        
    w = waitbar(No/225,w,['Image No. ', num2str(No)]);
    
    clear vars SPmatrix_b SPmatrix_g SPmatrix_r x y 
    
        
    imgsource = strcat(FName, num2str(No), filetype);
    img = imread(imgsource);
   
    img_hsv = rgb2hsv(img);
    img_filt = imgaussfilt3(img,5);
    
    [row, col, ch] = size(img);
    
    [L,N] = superpixels(img_filt, 10000);
    bw = boundarymask(L);
    
    img1 = img_hsv(:,:,1); 
    img2 = img_hsv(:,:,2);    
    img3 = img_hsv(:,:,3);

    binarysource = strcat(FName, num2str(No), ' Binary', filetype);
    binary = imread(binarysource);
    binary = im2bw(binary);
            
    for m = 1:N
           
        [x, y] = find(L==m);
        
        x_m = median(x)/col;
        y_m = median(y)/row;
        
        [length, ncol] = size(x);
        
        k = k + 1;
        
        data = zeros(3,length); 
        data_binary = zeros(1,length);
        
                
        for a = 1:length
            data(1,a) = double(img1(x(a),y(a)));
            data(2,a) = double(img2(x(a),y(a)));
            data(3,a) = double(img3(x(a),y(a)));                                   
            data_binary(1,a) = binary(x(a),y(a));
            
        end

       check = max(size(data_binary));
        
        if sum(data_binary) == check || sum(data_binary) == 0
        
        % Superpixel Information (33/81 features)    
            
        features(:,1) = mean(data,2);
        features(:,2) = std(data,0,2);
        features(:,3) = max(data, [], 2);
        features(:,4) = min(data, [], 2);
        features(:,5) = median(data,2);
        features(:,6) = rms(data,2);
        features(:,7) = skewness(data,1,2);
        features(:,8) = kurtosis(data,1,2);
        features(:,9) = var(data, 0, 2);
        features(:,10) = x_m;
        features(:,11) = y_m;
        
        SVM_Matrix(k,1:11) = features(1,:);
        SVM_Matrix(k,12:22) = features(2,:);
        SVM_Matrix(k,23:33) = features(3,:);
        
         x_m = ceil(median(x));
            y_m = ceil(median(y));
            
            if y_m > 51
                y_m_west = y_m - 50;                
            else
                y_m_west = 1;
            end
            
            if y_m < col-50                
                y_m_east = y_m + 50;                
            else
                y_m_east = col;                
            end
                
            if x_m > 51
                x_m_north = x_m - 50;                
            else
                x_m_north = 1;
            end
            
            if x_m > row-50                
                x_m_south = row;                
            else
                x_m_south = x_m + 50;                
            end
            
           
            
            NW = img_hsv(x_m_north:x_m, y_m_west:y_m, :);
            NE = img_hsv(x_m:x_m_south, y_m_west:y_m, :);
            SW = img_hsv(x_m_north:x_m, y_m:y_m_east, :);
            SE = img_hsv(x_m:x_m_south, y_m:y_m_east, :);
                               
            % Neighbourhood Information (48/81 features)
            
            SVM_Matrix(k,34) = mean(mean(NW(:,:,1))); %NW Quadrant
            SVM_Matrix(k,35) = std(std(NW(:,:,1)));
            SVM_Matrix(k,36) = min(min(NW(:,:,1)));
            SVM_Matrix(k,37) = max(max(NW(:,:,1)));
            
            SVM_Matrix(k,38) = mean(mean(NW(:,:,2))); 
            SVM_Matrix(k,39) = std(std(NW(:,:,2)));
            SVM_Matrix(k,40) = min(min(NW(:,:,2)));
            SVM_Matrix(k,41) = max(max(NW(:,:,2)));
            
            SVM_Matrix(k,42) = mean(mean(NW(:,:,3)));
            SVM_Matrix(k,43) = std(std(NW(:,:,3)));
            SVM_Matrix(k,44) = min(min(NW(:,:,3)));
            SVM_Matrix(k,45) = max(max(NW(:,:,3)));
            
            SVM_Matrix(k,46) = mean(mean(NE(:,:,1))); %NE Quadrant
            SVM_Matrix(k,47) = std(std(NE(:,:,1)));
            SVM_Matrix(k,48) = min(min(NE(:,:,1)));
            SVM_Matrix(k,49) = max(max(NE(:,:,1)));
            
            SVM_Matrix(k,50) = mean(mean(NE(:,:,2)));
            SVM_Matrix(k,51) = std(std(NE(:,:,2)));
            SVM_Matrix(k,52) = min(min(NE(:,:,2)));
            SVM_Matrix(k,53) = max(max(NE(:,:,2)));
            
            SVM_Matrix(k,54) = mean(mean(NE(:,:,3)));
            SVM_Matrix(k,55) = std(std(NE(:,:,3)));
            SVM_Matrix(k,56) = min(min(NE(:,:,3)));
            SVM_Matrix(k,57) = max(max(NE(:,:,3)));
            
            SVM_Matrix(k,58) = mean(mean(SW(:,:,1))); %SW Quadrant
            SVM_Matrix(k,59) = std(std(SW(:,:,1)));
            SVM_Matrix(k,60) = min(min(SW(:,:,1)));
            SVM_Matrix(k,61) = max(max(SW(:,:,1)));
            
            SVM_Matrix(k,62) = mean(mean(SW(:,:,2)));
            SVM_Matrix(k,63) = std(std(SW(:,:,2)));
            SVM_Matrix(k,64) = min(min(SW(:,:,2)));
            SVM_Matrix(k,65) = max(max(SW(:,:,2)));
            
            SVM_Matrix(k,66) = mean(mean(SW(:,:,3)));
            SVM_Matrix(k,67) = std(std(SW(:,:,3)));
            SVM_Matrix(k,68) = min(min(SW(:,:,3)));
            SVM_Matrix(k,69) = max(max(SW(:,:,3)));
            
            SVM_Matrix(k,70) = mean(mean(SE(:,:,1))); %SE Quadrant
            SVM_Matrix(k,71) = std(std(SE(:,:,1)));
            SVM_Matrix(k,72) = min(min(SE(:,:,1)));
            SVM_Matrix(k,73) = max(max(SE(:,:,1)));
            
            SVM_Matrix(k,74) = mean(mean(SE(:,:,2)));
            SVM_Matrix(k,75) = std(std(SE(:,:,2)));
            SVM_Matrix(k,76) = min(min(SE(:,:,2)));
            SVM_Matrix(k,77) = max(max(SE(:,:,2)));
            
            SVM_Matrix(k,78) = mean(mean(SE(:,:,3)));
            SVM_Matrix(k,79) = std(std(SE(:,:,3)));
            SVM_Matrix(k,80) = min(min(SE(:,:,3)));
            SVM_Matrix(k,81) = max(max(SE(:,:,3)));

       
        SVM_Vector(k,1) = mode(data_binary);
        
        else
            k = k-1;
        end
    end
    
end

delete(w);

SVM_Vector = SVM_Vector(1:k,:);
SVM_Matrix = SVM_Matrix(1:k,:);

% Save compiled training matrix and binary training vector
cd 'C:\Users\user1\Documents\Thesis\SVM Training\Trained Models'
save('SVM Training Data.mat', 'SVM_Vector', 'SVM_Matrix');

%Run fitcsvm to create Mdl for
Mdl = fitcsvm(SVM_Matrix, SVM_Vector);
cd 'C:\Users\user1\Documents\Thesis\SVM Training\Trained Models'
save('Trained SVM Model.mat', 'Mdl');
