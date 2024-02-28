clear
close all
clc

load feature_normalized_range.mat;
load target_matrix.mat;
fileID = fopen("f_selection1.txt","a+");

fprintf(fileID,"Selection for target_mean:\n");
for i = 1:5
    fprintf(fileID,'%s',strcat(string(i),"° try: "));
    
    opts = statset('display','iter', 'UseParallel',true);
    fs_mean = sequentialfs(@criterion,feature_normalized_range,target_matrix(:,1),'cv', 10,'options',opts, 'nfeatures', 10);
    fprintf(fileID,'%s\n', num2str(find(fs_mean)));
end

fprintf(fileID,"\n\nSelection for target_std:\n");
for i = 1:5
    fprintf(fileID,'%s',strcat(string(i),"° try: "));
    
    opts = statset('display','iter', 'UseParallel',true);
    fs_std = sequentialfs(@criterion,feature_normalized_range,target_matrix(:,2),'cv', 10,'options',opts, 'nfeatures', 10);
    fprintf(fileID,'%s\n', num2str(find(fs_std)));
end

fclose(fileID);

function performance = criterion(data_train, target_train, data_test, target_test)

    net=feedforwardnet(50);

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