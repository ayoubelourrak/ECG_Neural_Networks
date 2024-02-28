clear
close all
clc

%prepare dataset and train function
load("..\Feature\feature_normalized_range.mat");
load("..\Feature\target_matrix.mat");
x = feature_normalized_range(:,[1 2 5 6 8 9 82])';
t = target_matrix(:,1)';
train_function = "trainbr";

%initialize network
net = feedforwardnet(12);

net.trainFcn = train_function;
net.divideFcn = 'dividerand';
net.divideMode = 'sample';
net.divideParam.trainRatio = 0.70;
net.divideParam.valRatio = 0.20;
net.divideParam.testRatio = 0.10;
net.trainParam.showCommandLine = false;
net.trainParam.showWindow = false;
net.trainParam.max_fail = 15;
net.performFcn = 'mse';

%train network
[net, tr] = train(net, x, t, 'useParallel','yes');

x_test = x(:, tr.testInd);
t_test = t(:, tr.testInd);

%test network
y = net(x_test);
perf = perform(net, t_test, y);
disp(string(perf));
plotregression(t_test,y,'Regression');
save('performance_mean_final','perf');
