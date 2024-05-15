% Rearrange and predict class labels for superpixels

clc
clear
close all


FName = 'SVM Validation_';
filetype = '.tif';

FormatImage = zeros(100000,33);
AI_Matrix = zeros(100000,33);
FormatImage2 = zeros(100000,33);
features = zeros(3,11);


w = waitbar(0, 'My Progress Bar');


for No = 1:25

    
    w = waitbar(No/25 ,w,['Image # ', num2str(No)]);
       
        
        k = 0;
        
        cd 'C:\Users\user1\Documents\Thesis\SVM Training\Genesee Model\Validation Images'
        
        clearvars x y
        
        imgsource = strcat(FName, num2str(No), filetype);
        img = imread(imgsource);
        
        img_hsv = rgb2hsv(img);
        img_filt = imgaussfilt3(img,5);
        
        [row, col, ch] = size(img_hsv);

      
        [L,N] = superpixels(img_filt, 10000);
        bw = boundarymask(L);
             
        img1 = img_hsv(:,:,1);
        img2 = img_hsv(:,:,2);
        img3 = img_hsv(:,:,3);
        
        
        % Build matrix of superpixel features for new image
        for m = 1:N
            
            
            [x, y] = find(L==m);
            
            x_m = median(x)/col;
            y_m = median(y)/row;
            
            [length, ncol] = size(x);
            
            k = k + 1;
            
            data = zeros(3,length);
            
            for a = 1:length
                data(1,a) = double(img1(x(a),y(a)));
                data(2,a) = double(img2(x(a),y(a)));
                data(3,a) = double(img3(x(a),y(a)));
                 
            end
            
            
            % Superpixel features (33/81 features)
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
            
          
            FormatImage(k,1:11) = features(1,:);
            FormatImage(k,12:22) = features(2,:);
            FormatImage(k,23:33) = features(3,:);
            
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
            FormatImage(k,34) = mean(mean(NW(:,:,1))); %NW Quadrant
            FormatImage(k,35) = std(std(NW(:,:,1)));
            FormatImage(k,36) = min(min(NW(:,:,1)));
            FormatImage(k,37) = max(max(NW(:,:,1)));
            
            FormatImage(k,38) = mean(mean(NW(:,:,2))); 
            FormatImage(k,39) = std(std(NW(:,:,2)));
            FormatImage(k,40) = min(min(NW(:,:,2)));
            FormatImage(k,41) = max(max(NW(:,:,2)));
            
            FormatImage(k,42) = mean(mean(NW(:,:,3)));
            FormatImage(k,43) = std(std(NW(:,:,3)));
            FormatImage(k,44) = min(min(NW(:,:,3)));
            FormatImage(k,45) = max(max(NW(:,:,3)));
            
            FormatImage(k,46) = mean(mean(NE(:,:,1))); %NE Quadrant
            FormatImage(k,47) = std(std(NE(:,:,1)));
            FormatImage(k,48) = min(min(NE(:,:,1)));
            FormatImage(k,49) = max(max(NE(:,:,1)));
            
            FormatImage(k,50) = mean(mean(NE(:,:,2)));
            FormatImage(k,51) = std(std(NE(:,:,2)));
            FormatImage(k,52) = min(min(NE(:,:,2)));
            FormatImage(k,53) = max(max(NE(:,:,2)));
            
            FormatImage(k,54) = mean(mean(NE(:,:,3)));
            FormatImage(k,55) = std(std(NE(:,:,3)));
            FormatImage(k,56) = min(min(NE(:,:,3)));
            FormatImage(k,57) = max(max(NE(:,:,3)));
            
            FormatImage(k,58) = mean(mean(SW(:,:,1))); %SW Quadrant
            FormatImage(k,59) = std(std(SW(:,:,1)));
            FormatImage(k,60) = min(min(SW(:,:,1)));
            FormatImage(k,61) = max(max(SW(:,:,1)));
            
            FormatImage(k,62) = mean(mean(SW(:,:,2)));
            FormatImage(k,63) = std(std(SW(:,:,2)));
            FormatImage(k,64) = min(min(SW(:,:,2)));
            FormatImage(k,65) = max(max(SW(:,:,2)));
            
            FormatImage(k,66) = mean(mean(SW(:,:,3)));
            FormatImage(k,67) = std(std(SW(:,:,3)));
            FormatImage(k,68) = min(min(SW(:,:,3)));
            FormatImage(k,69) = max(max(SW(:,:,3)));
            
            FormatImage(k,70) = mean(mean(SE(:,:,1))); %SE Quadrant
            FormatImage(k,71) = std(std(SE(:,:,1)));
            FormatImage(k,72) = min(min(SE(:,:,1)));
            FormatImage(k,73) = max(max(SE(:,:,1)));
            
            FormatImage(k,74) = mean(mean(SE(:,:,2)));
            FormatImage(k,75) = std(std(SE(:,:,2)));
            FormatImage(k,76) = min(min(SE(:,:,2)));
            FormatImage(k,77) = max(max(SE(:,:,2)));
            
            FormatImage(k,78) = mean(mean(SE(:,:,3)));
            FormatImage(k,79) = std(std(SE(:,:,3)));
            FormatImage(k,80) = min(min(SE(:,:,3)));
            FormatImage(k,81) = max(max(SE(:,:,3)));

            
        end
        
        
        FormatImage = FormatImage(1:k,:);

        
        cd 'C:\Users\user1\Documents\Thesis\SVM Training\Trained Models'
        
        % Load trained model and predict binary image
        load('Trained SVM Model.mat');
        [label, score] = predict(Mdl, FormatImage);
        
        Binary = zeros(row, col);
        
        % Input superpixel predictions into blank matrix to create binary
        % image
        for m = 1:N
            Binary(L == m) = label(m);
        end
        
        Binaryfile = strcat(FName, num2str(No),' Binary (None)', filetype);
        
        cd 'C:\Users\user1\Documents\Thesis\SVM Training\Genesee Model\Validation Images'
        
         imwrite(Binary, Binaryfile);
        
        % Morphological Operations
        filled = ones(row,col);
        holes = filled & ~Binary;
        bigholes = bwareaopen(holes, 5000);
        smallholes = holes & ~bigholes;
        Binary2 = Binary|smallholes;
        
        Binary2 = bwareaopen(Binary2, 250);
    
        [height,width] = size(Binary2);
        Surface_Concentration = bwarea(Binary2)/(height*width)*100;
        
        % Write new binary file
        Binaryfile = strcat(FName, num2str(No),' Binary (Prediction)', filetype);
        imwrite(Binary2, Binaryfile);

end

delete(w);