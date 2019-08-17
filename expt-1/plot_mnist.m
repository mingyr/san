clear;
clc;

load('stats.mat');

p_index = find(classes == 1);
n_index = find(classes == 0);
    
p_data = logits(p_index);
n_data = logits(n_index);

p_labels = classes(p_index);
n_labels = classes(n_index);

figure;
histogram(p_data, 20);
set(gca,'FontSize',16);

figure;
histogram(n_data, 20);
set(gca,'FontSize',16);

% gscatter(logits, logits, classes, 'br','xo');
