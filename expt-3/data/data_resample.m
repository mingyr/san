% resample the data according to the designed artificial distribution
% the ratio between one modality vs. the other is around 1:3

load('VEP.mat');
new_subjects = [];

for i = 1:length(subjects)
    data = subjects(i).dataB;
    labels = subjects(i).labelsB;
    p_index = find(labels == 1);
    n_index = find(labels == -1);
    
    n_index = n_index(randperm(numel(n_index)));
    n_index = n_index(1: 3 * numel(p_index));
    
    indices = [p_index(:); n_index(:)];
    indices = indices(randperm(numel(indices)));
    
    new_subjects(i).data = data(:, :, indices);
    new_subjects(i).labels = labels(indices);
end

save('vep_data', 'new_subjects');