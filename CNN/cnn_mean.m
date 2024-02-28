clear
close all
clc

%prepare dataset
load("cnn_data.mat");
load("cnn_target.mat");
x = cnn_input;
t = normalize(cnn_output, 'range');

%remove outliers from dataset
[t, to_remove] = rmoutliers(t);
x = x(~to_remove);
fprintf("Remove %i outliers from data\n", sum(to_remove));

%separate training and test dataset
c = cvpartition(numel(x),'HoldOut', 0.2);
train_i = training(c);
test_i = test(c);
x_train = x(train_i);
x_test = x(test_i);
t_train = t(train_i);
t_test = t(test_i);

%options
options = trainingOptions('adam', ...%
    MaxEpochs = 50, ...
    MiniBatchSize = 100, ...%
    Shuffle = 'every-epoch' , ...
    LearnRateSchedule = 'none', ...%
    LearnRateDropFactor = 0.9, ...
    ValidationData =  {x_test t_test}, ...
    ValidationFrequency = 1, ...%
    ExecutionEnvironment = 'auto', ...%
    Plots = 'training-progress', ...
    OutputNetwork="best-validation-loss", ...
    ValidationPatience=inf, ...
    Verbose = 0 ...
);

%layers
filter_size = 20;
num_filters = 50;
pool_width = 4;
layers = [
    sequenceInputLayer(11)
    
    convolution1dLayer(filter_size, num_filters, 'Stride', 2, 'Padding', 'same')
    batchNormalizationLayer
    leakyReluLayer
    maxPooling1dLayer(pool_width, 'Stride', pool_width, 'Padding', 'same')


    convolution1dLayer(filter_size, 2 * num_filters, 'Stride', 2, 'Padding', 'same')
    batchNormalizationLayer
    leakyReluLayer
    maxPooling1dLayer(pool_width, 'Stride', pool_width, 'Padding', 'same')
    
    
    convolution1dLayer(filter_size, 4 * num_filters, 'Stride', 2, 'Padding', 'same')
    batchNormalizationLayer
    leakyReluLayer
    maxPooling1dLayer(pool_width, 'Stride', pool_width, 'Padding', 'same')


    globalAveragePooling1dLayer
    fullyConnectedLayer(100)
    dropoutLayer(0.3)
    fullyConnectedLayer(1)
    regressionLayer
];

%train network
net = trainNetwork(x_train, t_train, layers, options);


%test network
y_test = predict(net, x_test, ...
                ExecutionEnvironment='auto', MiniBatchSize = 100);

figure; 
plotregression(t_test, y_test);
