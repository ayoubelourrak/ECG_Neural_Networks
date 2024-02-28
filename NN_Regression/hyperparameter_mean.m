clear
close all
clc

train_function = ["trainbr","trainlm","trainbfg","trainrp","trainscg","traincgb","traingd"];
row = 0;
column = 0;
train_i = "none";
levels = "none";
value = 100000000000; %initialize min value

%find best configuration with one layer for each training function
for i = 1:7
    disp(train_function(i));
    load(strcat("performance_one_layer_",train_function(i)));
    matrix = performance_one_layer;
    mini = min(matrix(:));
    if mini < value
        [row, column] = find(matrix == mini);
        value = mini;
        train_i = train_function(i);
        levels = "one";
    end
    clear("performance_one_layer");
end

%find best configuration with two layers for each training function
for i = 1:7
    disp(train_function(i));
    load(strcat("performance_two_layer_",train_function(i)));
    matrix = performance_two_layer;
    mini = min(matrix(:));
    if mini < value
        [row, column] = find(matrix == mini);
        value = mini;
        train_i = train_function(i);
        levels = "two";
    end
    clear("performance_two_layer");
end

%output best configuration
disp(strcat("level: ",levels," train function: ",train_i," row: ",string(row)," column: ",string(column)," error: ",string(value)));