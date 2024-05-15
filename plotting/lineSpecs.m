line_width = 2;
enable_markers = 0;

% line_cols = {
% 	'magenta', 'green', 'cyan', 'forest_green','blue', 'red', 'purple', 'dark_orange',...
% 	'peach_puff', 'maroon', 'white', 'black'
% 	};
% line_cols = {
% 	'forest_green', 'blue', 'red', 'purple', 'dark_orange',...
%     'peach_puff', 'maroon', 'white', 'black'
% 	};
line_cols = {
	'red', 'purple', 'dark_orange'
	};


% line_cols = {
% 	'red_1', 'red_2', 'red_3', 'red_4', 'red_5'
% 	};


% line_styles = {':', ':', ':', '-', '-', '-', '-', '-', '-', '-', '-', '-'};
% line_styles = {'-.', '-.', '-.', '-', '-', '-', '-', '-', '-', '-', '-', '-'};
% line_styles = {'--', '--', '--', '-', '-', '-', '-', '-', '-', '-', '-', '-'};
line_styles = {'-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'};
markers = {'o', '+', '*', 'x', 'p', 'd', 'o','+', '*', 'x'};


mode = 0;
if mode == 1
%     line_cols = {'blue', 'forest_green', 'blue', 'forest_green'};
    line_cols = {'blue', 'purple', 'blue', 'purple'};
    line_styles = {'-', '-', '-', ':', ':', ':'};
    markers = {'o', '+','o', '+'};
    % valid_columns = [1, 3];
elseif mode == 2
    line_cols = {'blue', 'purple', 'blue', 'purple'};
    line_styles = {'-', '-', ':', ':', ':'};
    markers = {'o', '+', 'o', '+'};
elseif mode == 3
%         line_cols = {'blue', 'blue', 'forest_green', 'forest_green',...
%             'magenta', 'magenta', 'red', 'red',...
%             'cyan', 'cyan', 'purple', 'purple'};
        
%         line_cols = {'blue', 'red', 'forest_green', 'cyan','purple','green',...
%             'blue', 'forest_green', 'green','red', 'cyan','purple'};
        
        line_cols = {'blue', 'blue', 'blue',...
            'red','red','red',...
            'forest_green',...
            'forest_green', 'forest_green',...
            'purple', 'purple','purple'};
        
        
        line_styles = {
            '-', '-', '-', '-',...
            '-', '-', '-', '-',...
            '-', '-','-', '-'...
            '-', '-', '-', '-'};
        markers = {
            'o', 'o','o', 'o',...
            'o', 'o','o', 'o',...
            'o', 'o','o', 'o',...
            'o', 'o','o', 'o',...
            };
        
elseif mode == 4
%         line_cols = {'blue', 'blue', 'forest_green', 'forest_green',...
%             'magenta', 'magenta', 'red', 'red',...
%             'cyan', 'cyan', 'purple', 'purple'};
        
%         line_cols = {'blue', 'red', 'forest_green', 'cyan','purple','green',...
%             'blue', 'forest_green', 'green','red', 'cyan','purple'};
        
        line_cols = {'blue', 'blue', 'blue', 'blue',...
            'red','red','red','red',...
            'forest_green',...
            'forest_green', 'forest_green', 'forest_green',...
            'purple', 'purple','purple','purple'};
        
        
        line_styles = {
            '-', '-', '-', '-',...
            '-', '-', '-', '-',...
            '-', '-','-', '-'...
            '-', '-', '-', '-'};
        markers = {
            'o', 'o','o', 'o',...
            'o', 'o','o', 'o',...
            'o', 'o','o', 'o',...
            'o', 'o','o', 'o',...
            };
end
% line_styles = {'-', '-', '-', '--', '--', '--'};

% line_styles = {'--', '-', '-', '-', '-'};
% line_styles = {'-', ':', '-', ':', '-', ':', '-', ':', '-', ':', '-', ':', '-', ':', '-', ':', '-', ':'};
% line_styles = {'-', '--', '-', '--', ':', '-', '--', '-', '--', '-', '--', '-', '--', '-', '--', '-', '--', '-', '--'};

% line_styles = {'-', '-', '--', '--', '--'};
% line_styles = {'-', '-', '--', '-'};

% line_styles = {'-o', '-+', '-*', '-x', '-s', '-p'};


% line_specs = {'-og', '-+r'};
% line_specs = {'--or', '-+g', '--xm', '-xm'};

% line_specs = {'-or', '-+g', '-*b', '--xm'};

% line_specs = {'-or', ':+r', '-*b', ':xb'};
% line_specs = {'-or', '--+r', '-*b', '--xb'};

% line_specs = {'-or', '-+g', '--*b', '--xm'};
% line_specs = {'--or', '-+b', '-*g', '-xm'};

% line_specs = {'--or', '-+g', '-*b', '-xm', '-sc', '-pk'};

% line_specs = {'-or', '-+g', '-*b',...
%     '--or', '--+g', '--*b'};

% line_specs = {'-+g', '-*b', '-xm',...
%     '--+g', '--*b', '--xm'};

% line_specs = {'-or', '-+g', '-*b', '-xm',...
%     '--or', '--+g', '--*b', '--xm'};

% line_specs = {'-or', '-+g', '-*b', '-xm', '-sc',...
%     '--or', '--+g', '--*b', '--xm', '--sc'};

% line_specs = {'-or', '--*r', '-+g', '--xg'};
% line_specs = {'-or', '-+g', '--*r', '-+g', '--xg'};

line_cols_all = {
	'magenta', 'green', 'cyan', 'forest_green','blue', 'red', 'purple', 'dark_orange',...
	'peach_puff', 'maroon', 'white', 'black'
	};

line_cols = [line_cols, line_cols_all];


% set(0,'DefaultAxesFontName', 'Times New Roman');
% k=importdata('radon.txt');

% plot_title_override='UNet';
% k=importdata('unet_summary.txt');

% plot_title_override='SegNet';
% k=importdata('segnet_summary.txt');

% plot_title_override='Deeplab';
% k=importdata('deeplab_summary.txt');

% plot_title_override='DenseNet';
% k=importdata('densenet_summary.txt');

% plot_title_override='Comparing models';
% plot_title_override='Selective Training';

% plot_title_override='Recall rates using 5000 video images for training';
% plot_title_override='Recall rates on 20K 3-class test set without static images';

% y_label_override = 'Recall (%)';
% y_label_override = 'pixel accuracy';
% y_label_override = 'acc/IOU';
% y_label_override = 'Recall / Precision (%)';