clear; close all; clc;

data_folder = "..\Feature\";
mode = ["run", "sit", "walk"];


%load dataset
load Feature\data.mat;
load Feature\target.mat;
windows_num = 50; % number of non overlapping window for each file
rnn_data = cell([windows_num*3*22,1]);
rnn_target = cell([windows_num*3*22,1]);

%get data windows from each file
for i = 1:3
    for j = 1:22
        activity_data = table2array(data{i,j}(:,2:12));
        activity_target = table2array(target{i,j}(:,2));
        rows = size(activity_data, 1);
        dim = ceil(rows/windows_num);
        mode_offset = (i-1) * 22 * windows_num;
        person_offset = (j-1) * windows_num;
        offset = mode_offset + person_offset;

        for k = 1:windows_num
            start_row = (k-1) * dim + 1;
            end_row = start_row + dim - 1;
            if end_row > rows
                end_row = rows;
            end
            window_data = activity_data(start_row:end_row,:);
            window_target = activity_target(start_row:end_row,:);
            rnn_data{offset+k} = [window_target, window_data]';
            rnn_target{offset+k} = window_target';
        end
        display(strcat(string(i)," ",string(j)))
    end
end

%save dataset
save("RNN/rnn_data.mat","rnn_data");
save("RNN/rnn_target.mat","rnn_target");
