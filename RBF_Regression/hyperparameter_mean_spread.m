clear
close all
clc

train_function = "trainbr";
train_i = "none";
spread = 0;
value = 100000000000; %initialize min value

%find best configuration that not use trainbr
for i = 0.05:0.05:1.8
    load(strcat("performance_spread_",string(i)));
    mini = perf;
    if mini < value
        value = mini;
        spread = i;
    end
    clear("perf");
end


%find best configuration that use trainbr
for i = 0.05:0.05:1.8
    load(strcat("performance_",train_function,"_spread_",string(i)));
    mini = perf_br;
    if mini < value
        value = mini;
        train_i = train_function;
        spread = i;
    end
    clear("perf_br");
end

%output best configuration
disp(strcat("spread: ",string(spread)," train function: ",train_i," error: ",string(value)));