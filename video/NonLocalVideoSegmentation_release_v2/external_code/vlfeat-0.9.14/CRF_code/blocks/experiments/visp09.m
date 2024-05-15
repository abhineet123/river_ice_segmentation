global wrd;
wrd.prefix = '/mnt/e0.2/scratch/brian/releasedata/superpixel';
fprintf('Pascal 09 scoring\n\n\n');
vismulti('p09.*');

fprintf('Intersection / Union \n\n\n');
vismulti('iu', 'p09.*');

