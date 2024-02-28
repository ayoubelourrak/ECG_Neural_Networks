clear
close all
clc

%prepare dataset and variables
load("..\Feature\feature_normalized_range.mat");
load("..\Feature\target_matrix.mat");
x = feature_normalized_range(:,[1 2 4 5 6 7 8 9 82]);
t = target_matrix(:,3:5);
fileID = fopen("fuzzy_selection2.txt","a+");
d = dictionary;
vett = zeros(1,3*30);

%select best features
fprintf(fileID,"Selection best features :\n");
for i = 1:30
    fprintf(fileID,'%s',strcat(string(i),"° try: "));
    
    opts = statset('display','iter', 'UseParallel',true);
    fs = sequentialfs(@fuzzy_criterion,x,t,'cv', 10,'options',opts, 'nfeatures', 3);
    idx = (i-1) * 3 + 1;
    features_selected = find(fs);
    disp(features_selected);
    vett(idx:idx+2)=features_selected;
    d(features_selected) = [0 0 0];
    fprintf(fileID,'%s\n', num2str(features_selected));
end
for i = vett
    d(i) = d(i) + 1;
end
disp(d);
fclose(fileID);

function performance = fuzzy_criterion(data_train, target_train, data_test, target_test)

    net = patternnet([28,20]);

    net.divideParam.trainRatio = 0.9; 
    net.divideParam.valRatio = 0.1; 
    net.divideParam.testRatio = 0.0;
    net.trainParam.showCommandLine = false;
    net.trainParam.showWindow = false;

    data_train_t = data_train';
    target_train_t = target_train';
    
    net=train(net, data_train_t, target_train_t);
    
    data_test_t = data_test';
    target_test_t = target_test';
    
    y = net(data_test_t);
    performance = perform(net,target_test_t, y);
    
end