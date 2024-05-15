if ~exist('no_clear', 'var')
    clearvars -except prev_fname batch_fname_id batch_fnames;
end

%Get input filename from clipboard
cb = 1;

metric='rec_prec';

% metric='tp_fp_uex';
% metric='roc_auc_uex';

% metric='auc_roc';
% metric='auc';
% metric='auc_ap';
% metric='fpr_fnr';
% metric='fpr_fnr_sub';
% metric='fnr';
% metric='auc_ap_fpr_fnr';

% metric='auc_partial';
% metric='auc_uex-iw';

y_limits = [0, 100];
x_limits = [0, 100];

rec_prec_mode = 2;
thresh_mode = 3;
bar_plot = 0;

% 1: normalize by max_x 
% 2: normalize by norm_factor, 
% 3: extend in both directions using the max X and Y and discarding annoying anomalous outlier values
enable_auc = 3;
% if enable_auc=3, plot the extended x, y data (used for computing the AUC) instead of the original data
plot_ext = 1;

% if max_x and min_x used to compute the range over which AUC is normalized comes from the data, otherwise max_x=100 and min_x=0
relative_norm = 0;

norm_factor = 100;

% remove last row corresponding to conf 1.0 that can sometimes distort plots
remove_last_row = 1;
adjust_y = 0;
sort_x = 1;

bar_vals = 1;
bar_vals_font_size=12;

enable_y_label = 1;
enable_x_label = 1;
vertcal_x_label = 0;
colored_x_label = 0;
add_ylabel_to_title = 0;

paper = 0;

transparent_bkg = 1;
transparent_legend = 1;
colored_legend = 1;
white_text = 1;
grey_text = 0;

axes_font_size = 24;
legend_font_size = 24;
title_font_size = 30;
% title_interpreter = 'tex';
title_interpreter = 'none';

hide_grid = 1;

% if ~is_iw
% 	y_limits = [0, 100];
% 	x_limits = [0, 100];
% end

colRGBDefs;
lineSpecs;

if exist('contains', 'builtin')      
	is_bar = contains(metric, 'auc', 'IgnoreCase', true) || contains(metric, 'fpr', 'IgnoreCase', true) || contains(metric, 'fnr', 'IgnoreCase', true);
	is_iw = contains(metric, '-iw', 'IgnoreCase', true);
	is_tp_fp = contains(metric, 'tp_fp', 'IgnoreCase', true);
	is_roc_auc = contains(metric, 'roc_auc', 'IgnoreCase', true);
else
	is_bar=0;
	is_iw=0;
	is_tp_fp=0;
	is_roc_auc=0;
end


if is_iw
	thresh_mode=0;
	enable_auc = 0;
	adjust_y = 0;
elseif is_roc_auc
	thresh_mode=0;
elseif is_bar
	bar_plot=1;
elseif is_tp_fp
	rec_prec_mode = 1
end



fname = 'combined_summary.txt';
prev_fname = 'combined_summary.txt';
if cb
	fname = clipboard('paste')
	if isdir(fname)
		fname = sprintf('%s/%s.csv', fname, metric);
		prev_fname = fname;
	elseif exist('isfile', 'builtin')  && isfile(fname)
		prev_fname = fname;
	else
		fname = prev_fname
	end
end

out_dir='C:/UofA/PhD/Reports/plots';
enable_ap = 0;

if paper
	transparent_bkg=0;
	transparent_legend=0;
	white_text=0;
	grey_text=0;
end


if bar_plot
    rec_prec_mode=0;
	y_label_override='metric';
	% enable_x_label=0;
end


set(0,'DefaultAxesFontSize', axes_font_size);
set(0,'DefaultAxesFontWeight', 'bold');

if grey_text
    set(0,'DefaultAxesXColor', col_rgb{strcmp(col_names,'light_slate_gray')});
	set(0,'DefaultAxesYColor', col_rgb{strcmp(col_names,'light_slate_gray')});
elseif white_text
    set(0,'DefaultAxesXColor', col_rgb{strcmp(col_names,'white')});
    set(0,'DefaultAxesXColor', col_rgb{strcmp(col_names,'white')});
    set(0,'DefaultAxesYColor', col_rgb{strcmp(col_names,'white')});
else
	set(0,'DefaultAxesXColor', col_rgb{strcmp(col_names,'black')});
    set(0,'DefaultAxesXColor', col_rgb{strcmp(col_names,'black')});
    set(0,'DefaultAxesYColor', col_rgb{strcmp(col_names,'black')});
end

set(0,'DefaultAxesLineWidth', 2.0);
% set(0,'DefaultAxesGridLineStyle', ':');    
%     set(0,'DefaultAxesGridAlpha', 1.0);
%     set(0, 'DefaultAxesGridColor', col_rgb{strcmp(col_names,'white')});
%     set(0,'DefaultAxesMinorGridColor', col_rgb{strcmp(col_names,'white')});

 

k = importdata(fname);

if rec_prec_mode
    n_items = size(k.data, 1);
    
    if thresh_mode
        n_lines = int32(size(k.data, 2) / 3);
    else
        n_lines = int32(size(k.data, 2) / 2);
    end
    
    x_data = zeros(n_items, n_lines);
    y_data = zeros(n_items, n_lines);
        
    x_auc_data = zeros(n_items+2, n_lines);
    y_auc_data = zeros(n_items+2, n_lines);
    
    fileID = fopen(fname,'r');
    plot_title = fscanf(fileID,'%s', 1);
    plot_legend = cell(n_lines, 1);

	if enable_auc
		auc_out_path = sprintf('%s.auc%d', fname, enable_auc);
		auc_out_fid = fopen(auc_out_path,'w');
	end

    for line_id = 1:n_lines
        plot_legend{line_id} = fscanf(fileID,'%s', 1);        
        if thresh_mode
            thresh_data = k.data(:, 3*(line_id-1)+1);

            rec_data = k.data(:, 3*(line_id-1)+2);
            prec_data = k.data(:, 3*(line_id-1)+3);

            if thresh_mode==1
                x_data(:, line_id) = thresh_data;
                y_data(:, line_id) = rec_data;
            elseif thresh_mode==2
                x_data(:, line_id) = thresh_data;
                y_data(:, line_id) = prec_data;
            elseif thresh_mode==3

                rec_header = k.colheaders{2};
                prec_header = k.colheaders{3};

				if rec_prec_mode==2
					x_data(:, line_id) = rec_data;
					y_data(:, line_id) = prec_data;
				else
					x_data(:, line_id) = prec_data;
					y_data(:, line_id) = rec_data;
				end
            else
                error('Invalid thresh_mode: %d', thresh_mode)
            end
        else
            rec_data = k.data(:, 2*line_id-1);
            prec_data = k.data(:, 2*line_id);
            x_data(:, line_id) = rec_data;
            y_data(:, line_id) = prec_data;
        end
        if enable_auc
            X = x_data(:, line_id);
            Y = y_data(:, line_id);

			valid = isfinite(Y);

			X = X(valid);
			Y = Y(valid);	
								
			max_x = max(X);
			min_x = min(X);

	
            if enable_auc==1
                [unique_X, unique_X_idx] = unique(X, 'stable');
                [unique_Y, unique_Y_idx] = unique(Y, 'stable');


                unique_idx = intersect(unique_X_idx,unique_Y_idx);

                X_u = X(unique_idx);
                Y_u = Y(unique_idx);

				% XY = cat(2, X, Y);
				% XY_u = unique(XY,'rows');
				% X_u = XY_u(:, 1);
				% Y_u = XY_u(:, 2);

				X_i = X_u;
				Y_i = Y_u;

				if max_x < 100
					x_i = linspace(max_x+0.01, 100, 50)';
					y_i = interp1(X_u,Y_u,x_i,'linear','extrap');
					X_i = cat(1, X_i, x_i);
					Y_i = cat(1, Y_i, y_i);
				end

				if min_x > 0
					x_i = linspace(0, min_x-0.01, 50)';
					y_i = interp1(X_u,Y_u,x_i,'linear','extrap');
					X_i = cat(1, x_i, X_i);
					Y_i = cat(1, y_i, Y_i);
				end

				% figure;
				% plot(X_i, Y_i);

				auc = trapz(X_i,Y_i);

                norm_auc = auc / norm_factor;
			elseif enable_auc==2
				auc = trapz(X,Y);
				range_x = max_x - min_x;
                norm_auc = auc / range_x;
			elseif enable_auc==3			
				auc_X = X;
				auc_Y = Y;
				
				% find the row where recall becomes maximum and copy this row into any rows before it while making the precision 0 and leaving the recall be
				[max_x, max_x_ind] = max(auc_X);
				if max_x_ind == 1		
					auc_X = [max_x; auc_X];
					auc_Y = [0; auc_Y];
				else
					for i__ = 1:max_x_ind
						auc_X(i__) = max_x;
						auc_Y(i__) = 0;
					end
					auc_X = [max_x; auc_X];
					auc_Y = [0; auc_Y];
				end


				% find the row where the precision becomes maximum and copy this row into any rows after it while making the recall zero and leaving the position be
				[max_y, max_y_ind] = max(auc_Y);	

				n_data = length(auc_Y);

				if max_y_ind == n_data
					auc_Y = [auc_Y; max_y];
					auc_X = [auc_X; 0];
				else
					for i__ = max_y_ind:n_data
						auc_Y(i__) = max_y;
						auc_X(i__) = 0;
					end
					auc_Y = [auc_Y; max_y];
					auc_X = [auc_X; 0];
				end

				x_auc_data(:, line_id) = auc_X;
				y_auc_data(:, line_id) = auc_Y;

				auc = trapz(auc_X,auc_Y);
				if relative_norm
					max_x = max(auc_X);
					min_x = min(auc_X);
				else
					max_x = 100;
					min_x = 0;
				end
				range_x = max_x - min_x;
                norm_auc = auc / range_x;
			else
				error('invalid enable_auc')
            end
            if norm_auc < 0
                norm_auc = -norm_auc;
            end
			fprintf(auc_out_fid,'%s\t%.2f\n', plot_legend{line_id}, norm_auc);
            plot_legend{line_id} = sprintf('%s (%.2f)', plot_legend{line_id}, norm_auc);            
        end
    end
	if enable_auc==3 & plot_ext
		rec_data = x_auc_data;
		prec_data = y_auc_data;

		x_data = x_auc_data;
		y_data = y_auc_data;
	end			

    if enable_ap
        [ap, mrec, mprec] = VOCap(flipud(rec_data/100.0),...
            flipud(prec_data/100.0));
        x_data = mrec*100.0;
        y_data = mprec*100.0;
        ap = ap*100;
        fprintf('%s ap: %f%%\n', plot_legend{line_id}, ap);
    end


    % plot_legend
    fclose(fileID);
	if enable_auc
    	fclose(auc_out_fid);
	end

    if thresh_mode
        thresh_label = k.colheaders(1);
        rec_label = k.colheaders(2);
        prec_label = k.colheaders(3);                
        if thresh_mode==1
            x_label = thresh_label;
            y_label = rec_label;
        elseif thresh_mode==2
            x_label = thresh_label;
            y_label = prec_label;
        elseif thresh_mode==3
            if rec_prec_mode==2
				x_label = rec_label;
                y_label = prec_label;
            else
                x_label = prec_label;
                y_label = rec_label;
            end

        else
            error('Invalid thresh_mode: %d', thresh_mode)
        end
    else
        x_label = k.colheaders(1);
        y_label = k.colheaders(2);
    end
%         x_ticks = zeros(n_lines, 1);
%         xtick_labels = cell(n_lines, 1);
%         for item_id = 1:n_items
%             xtick_labels{item_id} = sprintf('%d', 10*item_id);
%             x_ticks(item_id) = 10*item_id;
%         end
    %         x_ticks
    %         xtick_labels
else
    if isfield(k,'colheaders')
        n_items = size(k.data, 1);
        n_lines = size(k.data, 2) - 1;
        y_data = k.data(:, 2:end);
        % x_ticks = k.data(:, 1);
        x_data = repmat(k.data(:, 1),1, n_lines);
        plot_legend = {k.colheaders{2:end}};
        plot_title = k.textdata{1, 1};

        x_label = k.textdata{2, 1};   
        y_label = '';
        try
            if contains(x_label, '__')
                str_arr = split(x_label, '__');
                x_label = str_arr{1};
                y_label = str_arr{2};
            end
        end
        if strcmp(y_label, '')
            y_label = sprintf('%s', plot_legend{1});
            for line_id = 2:n_lines
                if ~strcmp(plot_legend{line_id}, '_') ||  ~strcmp(plot_legend{line_id}, '__')
                    y_label = sprintf('%s/%s', y_label, plot_legend{line_id});
                end
            end
        end 
        
        
    elseif size(k.textdata, 2) == size(k.data, 2) + 1
        n_items = size(k.data, 1);
        n_lines = size(k.data, 2);
        y_data = k.data(:, 1:end);
        % x_ticks = k.data(:, 1);
        x_data = repmat(transpose(1:n_items),1, n_lines);
        plot_legend = {k.textdata{3, 2:end}};
        plot_title = k.textdata{1, 1};
        x_label = k.textdata{2, 1};
        xtick_labels = k.textdata(4:end, 1);
        if n_items > 1
            x_ticks = 1:n_items;
            xlim([1, n_items]);
        end
        
        if exist('valid_columns', 'var')
            x_data = x_data(:, valid_columns);
            y_data = y_data(:, valid_columns);
            plot_legend = plot_legend(:, valid_columns);
            n_lines = size(x_data, 2);
            
            line_cols = line_cols(valid_columns);
            line_styles = line_styles(valid_columns);
            markers = markers(valid_columns);
            temp = strsplit(x_label, '---');
            
            x_label = temp{1};
            y_label = temp{valid_columns(1)+1};     
            
            if add_ylabel_to_title
                plot_title = sprintf('%s %s', plot_title, y_label);     
            end
        else
            if bar_plot
                y_label = sprintf('%s', xtick_labels{1});
                for line_id = 2:size(xtick_labels)
                    y_label = sprintf('%s/%s', y_label, xtick_labels{line_id});
                end
            else
                y_label = sprintf('%s', plot_legend{1});
                for line_id = 2:n_lines
                    if ~strcmp(plot_legend{line_id}, '_') ||  ~strcmp(plot_legend{line_id}, '__')
                        y_label = sprintf('%s/%s', y_label, plot_legend{line_id});
                    end
                end
            end
        end
        
    else
        n_lines = size(k.data, 2);
        n_items = size(k.data, 1);
        y_data = k.data;
        %     x_label='Model';
        
        n_text_lines = size(k.textdata, 2);
        n_text_items = size(k.textdata, 1);
        if n_text_items == n_items + 3
            y_label = k.textdata(1, 1)
            k.textdata = k.textdata(2:end, :);
            n_text_items = n_text_items - 1;
        end
        if n_text_items == n_items + 2
            plot_title = k.textdata(1, 1);
            k.textdata = k.textdata(2:end, :);
        end
        x_label = k.textdata(1, 1);
        plot_legend = {k.textdata{1, 2:end}};
        xtick_labels = k.textdata(2:end, 1);
        x_ticks = 1:n_items;
        x_data = repmat((1:n_items)',1, n_lines);           

    end
end
if exist('xtick_labels', 'var')           
    for j = 1:n_items
        if xtick_labels{j}(1)=='_'
            xtick_labels{j} = xtick_labels{j}(2:end);
        end
    end
end

%     y_data
%     x_data
%     line_cols
%     line_styles
% n_lines
% plot_title
% metric

if exist('contains', 'builtin')         
	if ~contains(plot_title, metric, 'IgnoreCase', true)
		plot_title = sprintf('%s-%s', plot_title, metric);   
	end  
end

fig_h = figure;
propertyeditor(fig_h,'on');

if bar_plot
	if adjust_y & exist('y_limits', 'var')
		n_groups = size(k.data, 1);
		n_bars = size(k.data, 2);
		all_max_datum = 0;
		for k1 = 1:n_groups
			datum = k.data(k1, :);
			max_datum = max(datum);
			max_y_lim = y_limits(2);
			if max_datum - max_y_lim > 5;
				mult_ratio = max_datum / max_y_lim;
				mult_ratio_int = ceil(mult_ratio/10)*10;
				k.data(k1, :) = datum / mult_ratio_int;
				max_datum = max(k.data(k1, :))
				xtick_labels{k1} = sprintf('%s (/%d)', xtick_labels{k1}, mult_ratio_int);
			end
			if all_max_datum < max_datum
				all_max_datum = max_datum;
			end
		end
		if max_y_lim - all_max_datum < 5
			y_limits(2) = y_limits(2) + 5;
		end
	end

    bar_plt = bar(k.data);
    % bar_child=get(bar_plt,'Children');

    if bar_vals
        y1 = k.data;
        for k1 = 1:size(y1,2)
            ctr(k1,:) = bsxfun(@plus, bar_plt(1).XData, bar_plt(k1).XOffset');    % Note: ‘XOffset’ Is An Undocumented Feature, This Selects The ‘bar’ Centres
            ydt(k1,:) = bar_plt(k1).YData;                                     % Individual Bar Heights
			if white_text
				text_col = 'w';
			else
				text_col = 'k';
			end

            text(ctr(k1,:), ydt(k1,:), sprintfc('%.1f', ydt(k1,:)), 'HorizontalAlignment','center', 'VerticalAlignment','bottom', 'FontSize', bar_vals_font_size, 'Color',text_col, 'FontWeight', 'bold')
        end
		h=gca; h.XAxis.TickLength = [0 0];
    end
end

% Y_as = k.data ;
% a = (1:size(Y_as,1)).';
% x = [a-1.0 a-0.75 a-0.5 a-0.25 a a+0.25 a+0.5 a+0.75 a+1.0];
% for row=1:size(Y_as,1)
%     for col=1:size(Y_as,2)
%         text(x(row,col),Y_as(row,col),num2str(Y_as(row,col),'%0.2f'),...
%             'HorizontalAlignment','center',...
%             'VerticalAlignment','bottom')
%     end
% end

if remove_last_row
	x_data(end,:) = [];
	y_data(end,:) = [];
end


bar_cols = [];

final_legend = {};
for i = 1:n_lines



    y_datum = y_data(:, i);
    x_datum = x_data(:, i);

	valid = isfinite(y_datum);

	x_datum = x_datum(valid);
	y_datum = y_datum(valid);

	if sort_x
		[x_datum,idx] = sort(x_datum);
		y_datum = y_datum(idx);
	end

	if ~bar_plot & adjust_y & exist('y_limits', 'var')
		max_y_datum = max(y_datum);
		max_y_lim = y_limits(2);
		if max_y_datum > max_y_lim;
			mult_ratio = max_y_datum / max_y_lim;
			mult_ratio_int = ceil(mult_ratio/10)*10;
			y_datum = y_datum / mult_ratio_int;
			plot_legend{i} = sprintf('%s (/%d)', plot_legend{i}, mult_ratio_int);
		end
	end		

    line_col = line_cols{i};

    line_style = line_styles{i};
    if enable_markers && exist('markers', 'var')            
        marker = markers{i};
    else
        marker = 'none';
    end
    vis = 'on';
    line_width_ = line_width;
    
    if strcmp(plot_legend{i}, '_')
        fprintf('Turning off legend for line %d\n', i)
%             vis = 'off';
        line_width_ = 2;
        marker = 'none';
        line_style = ':';
    elseif strcmp(plot_legend{i}, '__')
        fprintf('Turning off legend for line %d\n', i)
%             vis = 'off';
        marker = 'none';
        line_style = '--';
        line_width_ = 3;
    else
        final_legend{end+1} = plot_legend{i};
    end
    %     line_spec = line_specs{i};
    if bar_plot
        % bar_cols = [bar_cols; col_rgb{strcmp(col_names,line_col)}];      
        % bar(k.data(:, i), 'FaceColor', col_rgb{strcmp(col_names,line_col)});
        % set(bar_child{i},'FaceColor',col_rgb{strcmp(col_names,line_col)});
        % bar_plt(i).FaceColor =  col_rgb{strcmp(col_names,line_col)};
        set(bar_plt(i),'FaceColor', col_rgb{strcmp(col_names,line_col)});
        % bar_val = k.data(2, line_id);
        % plot_legend{line_id} = sprintf('%s (%.2f)', plot_legend{line_id}, bar_val);
    else
        plot(x_datum, y_datum,...
            'Color', col_rgb{strcmp(col_names,line_col)},...
            'LineStyle', line_style,...s
            'LineWidth', line_width_,...
            'Marker', marker,...
            'HandleVisibility', vis);
    end
    hold on
end

hold off
% if bar_plot
%     mydata=rand(1,5);
%     bar_child2 = cell2mat(bar_child)
%     set(bar_child2, 'CData', k.data);
%     colormap(bar_cols);
% end
if colored_legend
	for line_id = 1:n_lines
		line_col_rgb = col_rgb{strcmp(col_names,line_cols{line_id})};
		final_legend{line_id} = sprintf('\\color[rgb]{%f,%f,%f}%s',...
		line_col_rgb(1),...
		line_col_rgb(2),...
		line_col_rgb(3),...			
		final_legend{line_id});
	end
end

h_legend=legend(final_legend, 'Interpreter','tex');
set(h_legend,'FontSize', legend_font_size);
set(h_legend,'FontWeight','bold');
set(h_legend,'LineWidth', 1.0);

if grey_text
    set(h_legend,'TextColor', col_rgb{strcmp(col_names,'light_slate_gray')});
elseif white_text
    set(h_legend,'TextColor', col_rgb{strcmp(col_names,'white')});
else
	set(h_legend,'TextColor', col_rgb{strcmp(col_names,'black')});
end
%     grid on;   

% h_legend_child=get(h_legend,'Children');
% for i = 1:n_lines
%     line_col = line_cols{i};
%     if bar_plot        
%         % set(bar_child{i},'FaceColor',col_rgb{strcmp(col_names,line_col)});
%         set(h_legend_child(i),'FaceColor',col_rgb{strcmp(col_names,line_col)});
%         continue;
%     end
% end

grid(gca,'on')

% ax = gca;
% ax.GridAlpha=0.25;
% ax.GridLineStyle=':';
%     set (gca, 'GridAlphaMode', 'manual');
%     set (gca, 'GridAlpha', 1.0);
% set (gca, 'GridLineStyle', '-');

if exist('xtick_labels', 'var')         
    xtick_label_cols = cell(length(xtick_labels), 1);
    for x_label_id=1:length(xtick_labels)
        curr_x_label=xtick_labels{x_label_id};

        split_x_label = strsplit(curr_x_label, '---');       


        if length(split_x_label)==2
            new_x_label = split_x_label{1};
            xtick_label_col = split_x_label{2};            
        else
            new_x_label = curr_x_label;
            xtick_label_col='black';
        end


%         
%         if contains(curr_x_label, '@')
%             split_x_label = split(curr_x_label, '@');
%             new_x_label = split_x_label{1};
%             xtick_label_col = split_x_label{2};            
%         else
%             new_x_label = curr_x_label;
%             xtick_label_col='black';
%         end

		new_x_label = strrep(new_x_label,'_','-');
        xtick_labels{x_label_id} = new_x_label;
        xtick_label_cols{x_label_id} = col_rgb{strcmp(col_names,xtick_label_col)};        
    end
end

%     xlim([1, n_items]);

% ax = gca;
% ax.LineWidth = 4;
try
	ax.GridAlpha = 0.05;
catch
	...
end

try
    if exist('x_ticks', 'var')
        xticks(x_ticks);
    end
    if exist('xtick_labels', 'var')
        xticklabels(xtick_labels);
    end
catch
    if exist('x_ticks', 'var')
        set(gca, 'XTick', x_ticks)
    end
    
    if exist('xtick_labels', 'var')
        set(gca, 'xticklabel', xtick_labels)
    end
end
% ylabel('metric value');
if exist('y_label_override', 'var')
    y_label = y_label_override;
else
	y_label = strtrim(y_label);
	%y_label=sprintf("%s (%%)", y_label);
end
if enable_y_label
	ylabel(y_label, 'fontsize',20, 'FontWeight','bold', 'Interpreter', 'none');
end


%     ax = gca;
%     outerpos = ax.OuterPosition;
%     ti = ax.TightInset; 
%     left = outerpos(1) + ti(1);
%     bottom = outerpos(2) + ti(2);
%     ax_width = outerpos(3) - ti(1) - ti(3);
%     ax_height = outerpos(4) - ti(2) - ti(4);
%     ax.Position = [left bottom ax_width ax_height];
%     ax.Position = outerpos;    

x_label = strtrim(x_label);
if enable_x_label
	% x_label=x_label{1};
	% x_label=sprintf('%s (%%)', x_label);
    xlabel(x_label, 'fontsize',20, 'FontWeight','bold', 'Interpreter', 'none');
end
if colored_x_label
    xticklabel_rotate([],0,[], xtick_label_cols, 'fontsize',20, 'FontWeight','bold',...
        'Interpreter', 'none');
elseif vertcal_x_label
    xticklabel_rotate([],90,[], xtick_label_cols, 'fontsize',20, 'FontWeight','bold',...
        'Interpreter', 'none');
end
% ylim([0.60, 0.90]);

if exist('x_limits', 'var')
    xlim(x_limits);
end
if exist('y_limits', 'var')
    ylim(y_limits);
end
if exist('plot_title_override', 'var')
    plot_title = plot_title_override
end
plot_title = strtrim(plot_title);
title_obj = title(plot_title, 'fontsize',title_font_size, 'FontWeight','bold',...
    'Interpreter', title_interpreter);    

if grey_text
    set(title_obj,'Color', col_rgb{strcmp(col_names,'light_slate_gray')});
elseif white_text
    set(title_obj,'Color', col_rgb{strcmp(col_names,'white')});
else
    set(title_obj,'Color', col_rgb{strcmp(col_names,'black')});
end

if transparent_bkg
    set(gca,'color','none')
    if transparent_legend
        set(h_legend,'color','none');
    end
end

if hide_grid
    grid off;
end

out_name = strrep(plot_title,'-','_');

% out_path=sprintf('%s/%s.png', out_dir, plot_title)
if paper
	set(gcf,'color','w');
else	
	% out_img = sprintf('%s/%s.png', out_dir, out_name);
	% fprintf('save_img(fig_h, "%s", "%s", "%s", 1)\n', out_dir, out_name, "png");
	% fprintf('export_fig %s -transparent -painters\n', out_img);
	set(gcf,'color','k');
end

fprintf('save_img("%s", "%s", "%s", %d)\n', out_dir, out_name, 'pdf', paper);
fprintf('save_img("%s", "%s", "%s", %d)\n', out_dir, out_name, 'png', paper);
fprintf('save_img("%s", "", "%s", %d)\n', out_dir, 'eps', paper);


% out_fig = sprintf('export_fig %s/%s.fig', out_dir, out_name);
% exportgraphics(fig_h, out_fig);

% fprintf('exportgraphics(fig_h, "%s");\n', out_img);
% fprintf('savefig("%s");\n', out_img);



% if white_text
% 	ax = gca;
% 	set(0,'DefaultAxesGridAlpha', 1.0);
% 	set(0, 'DefaultAxesGridColor', col_rgb{strcmp(col_names,'white')});
% 	set(0,'DefaultAxesMinorGridColor', col_rgb{strcmp(col_names,'white')});
% end





