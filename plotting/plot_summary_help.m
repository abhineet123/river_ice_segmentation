%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%{
export_fig 'C:\UofA\PhD\Reports\221121_ipsc_paper\swin.png' -transparent
export_fig 'C:\UofA\PhD\Reports\221121_ipsc_paper\idol.png' -transparent

rec_prec_mode = 0: plot each column as one line
    first line: plot title
    second line: x-label
    y_label is the concatenation of legend entries

rec_prec_mode > 0 and thresh_mode > 0: plot triplets of columns
    conf_thresh: column 1, rec: column 2, prec: column 3

    thresh_mode = 3
		rec_prec_mode = 1
			rec on y axis and prec on x axis
			first line: plot title + legend for each line separated by spaces
						or
			first line: plot title
			second line: legend for each line separated by spaces / tabs

		rec_prec_mode = 2
			rec on x axis and prec on y axis

    	plot_title_override and y_label_override for custom values

    thresh_mode 1: thresh vs rec
    thresh_mode 2: thresh vs prec

thresh_mode = 0: 
	rec_prec_mode = 2	
		plot pairs of columns - first col on x axis, second on y axis
%}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%