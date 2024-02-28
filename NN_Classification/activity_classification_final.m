clear
close all
clc

%prepare dataset and train function
load("..\Feature\feature_normalized_range.mat");
load("..\Feature\target_matrix.mat");
x = feature_normalized_range(:,[1 2 4 5 6 7 8 9 82])';
t = target_matrix(:,3:5)';
train_function = "traincgb";

%initialize network
n = patternnet([28,20]);
net = configure(n, x, t);

net.trainFcn = train_function;
net.divideFcn = 'dividerand';
net.divideMode = 'sample';
net.divideParam.trainRatio = 0.70;
net.divideParam.valRatio = 0.10;
net.divideParam.testRatio = 0.20;
net.trainParam.showCommandLine = false;
net.trainParam.showWindow = false;

%train network
[net, tr] = train(net, x, t, 'useParallel','yes');

x_test = x(:, tr.testInd);
t_test = t(:, tr.testInd);

%test network
y = net(x_test);
perf = perform(net, t_test, y);
plotconfusion(t_test,y);
save('classification_performance_2.mat','perf');
