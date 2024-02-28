clear
close all
clc

%prepare and normalize dataset
load("..\Feature\feature_normalized_range.mat");
load("..\Feature\target_matrix.mat");
x = feature_normalized_range(:,[1 2 5 6 8 9 82])';
t_not_normalized = target_matrix(:,1)';
t = normalize(t_not_normalized,'range');

%separate training and test dataset
[features, windows] = size(x);
probability = 0.85;
indexes = randperm(windows);
x_train = x(:, indexes(1:round(probability*windows)));
t_train = t(:, indexes(1:round(probability*windows)));
x_test = x(:, indexes(round(probability*windows)+1:windows));
t_test = t(:, indexes(round(probability*windows)+1:windows));

%initialize and train network
error_goal = 0;
K = 300;
Ki = 20;
spread = 0.15;
net = newrb(x_train, t_train, error_goal, spread, K, Ki);

%test network
y = net(x_test);
perf = perform(net, t_test, y);
plotregression(t_test,y,'Regression');
save('rbf_mean_performance.mat','perf');
