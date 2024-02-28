clear
close all
clc

%prepare dataset
load("rnn_data.mat");
load("rnn_target.mat");
x = cell(size(rnn_data));
t = cell(size(rnn_target));

for i = 1:numel(rnn_data)
    train_data = rnn_data{i};
    test_data = rnn_target{i};
    x{i} = train_data(:,1:end-1);
    t{i} = test_data(:,2:end);
end

%separate training and test dataset
c = cvpartition(numel(x),'HoldOut', 0.1);
train_i = training(c);
test_i = test(c);
x_train = x(train_i);
x_test = x(test_i);
t_train = t(train_i);
t_test = t(test_i);

%normalize data
muX = mean(cat(2,x_train{:}),2);
sigmaX = std(cat(2,x_train{:}),0,2);

muT = mean(cat(2,t_train{:}),2);
sigmaT = std(cat(2,t_train{:}),0,2);

for n = 1:numel(x_train)
    x_train{n} = (x_train{n} - muX) ./ sigmaX;
    t_train{n} = (t_train{n} - muT) ./ sigmaT;
end
for n = 1:numel(x_test)
    x_test{n} = (x_test{n} - muX) ./ sigmaX;
    t_test{n} = (t_test{n} - muT) ./ sigmaT;
end

num_channels = size(rnn_data{1},1);

%options
options = trainingOptions('adam', ...
    MaxEpochs = 20, ...
    Shuffle = 'every-epoch', ...
    SequencePaddingDirection = 'left', ...
    Plots = 'training-progress', ...
    Verbose = false, ...
    MiniBatchSize = 100, ...
    ExecutionEnvironment = 'auto' ...             
);

%layers
layers = [
    sequenceInputLayer(num_channels)
    lstmLayer(150)
    fullyConnectedLayer(1)
    regressionLayer
];


%train and test network
net = trainNetwork(x_train, t_train, layers, options);
y = predict(net, x_test, ExecutionEnvironment = 'auto', SequencePaddingDirection='left');

mse=zeros(1,numel(y));
for p=1:numel(y)
    mse(p)=mean((t_test{p}-y{p}).^ 2,'all');
end
performance=mean(mse);
save('performance_rnn.mat','performance')

% open loop forecasting
offset = 250;
test_performance=zeros(numel(rnn_target),1);
for m = 1:numel(x_test)
    net = resetState(net);
    [net,~] = predictAndUpdateState(net,x_test{m}(:,1:offset));
    numTimeSteps = size(x_test{m},2);
    numPredictionTimeSteps = numTimeSteps - offset;
    output = zeros(1,numPredictionTimeSteps);
    target = x_test{m}(1,offset+1:offset+numPredictionTimeSteps);

    for k = 1:numPredictionTimeSteps
        x_k = x_test{m}(:,offset+k);
        [net,output(1,k)] = predictAndUpdateState(net,x_k);
    end
    
    test_performance(m,1) = mean((target-output).^2);
    
end
save('openloop_performance_rnn.mat','test_performance')

% plots
plotregression(target,output)
figure
plot(x_test{m}(1,:))
hold on
plot(offset+1:offset+numPredictionTimeSteps,[output(1,:)],'--')
legend(['Input' 'Forecasted'])
