clear
close all
clc

%prepare dataset and FIS
load("..\Feature\feature_normalized_range.mat");
load("..\Feature\target_matrix.mat");
x = feature_normalized_range(:,[4 6 1]);
t = target_matrix(:,3:5);
t_index = vec2ind(t')';
fis = readfis('fis_generated.fis');

%get target class values
run = x(t_index == 1, :);
sit = x(t_index == 2, :);
walk = x(t_index == 3, :);
t_values = t_index';

%get prediction class values
y = round(evalfis(fis, x))';
equal = (y == t_values);
equal_run = equal(1:990);
equal_sit = equal(991:1980);
equal_walk = equal(1981:end);

fprintf('Percentage of correct classification prediction: %f%%\n', sum(equal) / size(equal, 2) * 100);

fprintf('Correct SIT prediction: %f%%\n', sum(equal_sit) / size(equal_sit, 2) * 100);
fprintf('Correct WALK prediction: %f%%\n', sum(equal_walk) / size(equal_walk, 2) * 100);
fprintf('Correct RUN prediction: %f%%\n', sum(equal_run) / size(equal_run, 2) * 100);
