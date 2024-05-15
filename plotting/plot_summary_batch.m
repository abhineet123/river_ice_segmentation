batch_fnames = clipboard('paste')
batch_fnames = splitlines(batch_fnames)

for batch_fname_id = 1:length(batch_fnames) 
    fname = batch_fnames{batch_fname_id};
	if length(fname) == 0
		continue
	end
	clipboard('copy',fname);
	plot_summary;	
end 






