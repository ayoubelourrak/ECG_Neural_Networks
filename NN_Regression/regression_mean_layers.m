clear
close all
clc

disp("Start:");
tim = now;
disp(datetime(tim,'ConvertFrom','datenum'));

%prepare dataset
load("..\Feature\feature_normalized_range.mat");
load("..\Feature\target_matrix.mat");
x = feature_normalized_range(:,[1 2 5 6 8 9 82])';
t = target_matrix(:,1)';
train_function = ["trainbr","trainlm","trainbfg","trainrp","trainscg","traincgb","traingd"];

for k = 1:7
    disp(train_function(k));
    performance_one_layer = zeros(1,19);
    index = 1;
    %train and test net with one layer
    for i = 7:5:100
        disp(string(i));
        net = feedforwardnet(i);
    
        net.trainFcn = train_function(k);
        net.divideFcn = 'dividerand';
        net.divideMode = 'sample';
        net.divideParam.trainRatio = 0.80;
        net.divideParam.valRatio = 0.10;
        net.divideParam.testRatio = 0.10;
        net.trainParam.showCommandLine = false;
        net.trainParam.showWindow = false;
        net.trainParam.max_fail = 15;
        net.performFcn = 'mse';
        
    
        [net, tr] = train(net, x, t);
        
        x_test = x(:, tr.testInd);
        t_test = t(:, tr.testInd);
        
        y = net(x_test);
        perf = perform(net, t_test, y);
        performance_one_layer(index) = perf;
        index = index + 1;
    end
    
    performance_two_layer = zeros(9);
    index1 = 1;
    index2 = 1;
    %train and test net with two layers
    for i = 7:5:50
        index2 = 1;
        for j = 7:5:50
            disp(strcat(string(i)," ",string(j)));
            net = feedforwardnet([i,j]);
        
            net.trainFcn = train_function(k);
            net.divideFcn = 'dividerand';
            net.divideMode = 'sample';
            net.divideParam.trainRatio = 0.80;
            net.divideParam.valRatio = 0.10;
            net.divideParam.testRatio = 0.10;
            net.trainParam.showCommandLine = false;
            net.trainParam.showWindow = false;
            net.performFcn = 'mse';
            
            [net, tr] = train(net, x, t);
            
            x_test = x(:, tr.testInd);
            t_test = t(:, tr.testInd);
            
            y = net(x_test);
            perf = perform(net, t_test, y);
            performance_two_layer(index1, index2) = perf;
            index2 = index2 + 1;
        end
        index1 = index1 + 1;
    end
    %save performance matrix
    save(strcat("performance_one_layer_",train_function(k),".mat"), "performance_one_layer");
    save(strcat("performance_two_layer_",train_function(k),".mat"), "performance_two_layer");
end


disp("End: ");
tim = now;
disp(datetime(tim,'ConvertFrom','datenum'));