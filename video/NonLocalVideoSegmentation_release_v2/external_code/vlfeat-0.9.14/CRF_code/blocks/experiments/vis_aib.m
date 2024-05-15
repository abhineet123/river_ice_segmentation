global wrd;
wrd.prefix = 'work-nonorm/graz/';

cats = {'bike', 'cars', 'person'};
for c = 1:length(cats)
    figure(c); clf;
    visavg(['gz_' cats{c} '.*_ikm200.*'], ...
           ['gz_' cats{c} '.*_hikm_8000.*'] ...
    );
end
