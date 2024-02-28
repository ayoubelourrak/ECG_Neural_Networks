clear
close all
clc

%prepare and normalize dataset
load("..\Feature\feature_normalized_range.mat");
load("..\Feature\target_matrix.mat");
x = feature_normalized_range(:,[1 2 4 5 7 8 9 82])';
t_not_normalized = target_matrix(:,2)';
t = normalize(t_not_normalized,'range');

%separate training and test dataset
[features, windows] = size(x);
probability = 0.85;
indexes = randperm(windows);
x_train = x(:, indexes(1:round(probability*windows)));
t_train = t(:, indexes(1:round(probability*windows)));
x_test = x(:, indexes(round(probability*windows)+1:windows));
t_test = t(:, indexes(round(probability*windows)+1:windows));

%calculate range of spread
distance = pdist(x');
disp(strcat("minimum distance: ",string(min(distance))));
disp(strcat("maximum distance: ",string(max(distance))));
minimum = round(min(distance),1) + 0.05;
maximum = round(max(distance),1);

%calculate best spread for the network
for spread = minimum:0.05:maximum
    disp(strcat("spread ",string(spread)));
    error_goal = 0;
    K = 300;
    Ki = 20;
    net = newrb(x_train, t_train, error_goal, spread, K, Ki);

    y = net(x_test);
    perf = perform(net, t_test, y);
    save(strcat("performance_spread_std_",string(spread),".mat"),"perf");

    net.trainFcn = 'trainbr';
    net.trainParam.epochs = 100;
    net.trainParam.showCommandLine = true;
    net.trainParam.showWindow = false;
    net = train(net, x_train, t_train, 'useParallel','yes');
    y_br = net(x_test);
    perf_br = perform(net, t_test, y_br);
    save(strcat("performance_trainbr_spread_std_",string(spread),".mat"),"perf_br");
end
