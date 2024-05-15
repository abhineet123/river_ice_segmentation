function save_img(path, fname, ext, is_paper)
	% fname = clipboard('paste')
	if strlength(fname) == 0
		h=get(gca,'Title');
		fname=get(h,'String')
		% fname = clipboard('paste')
	end

	out_path = fullfile(path, sprintf('%s.%s', fname, ext))
	% export_fig(out_path, '-transparent');

	if ~is_paper
		export_fig(out_path, '-transparent');
	else
		% export_fig(out_path);
		exportgraphics(gca, out_path)
	end
	out_fig_path = fullfile(path, sprintf('%s.fig', fname));
    savefig(out_fig_path)
end
