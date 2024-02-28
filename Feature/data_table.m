clear
close all
clc

dataset_folder = "..\dataset\";
mode = ["run", "sit", "walk"];

for j = 1:22
    for i = 1:3
        data_file = strcat('s',string(j),'_',mode(i),'_','timeseries.csv');
        opts = detectImportOptions(strcat(dataset_folder,data_file));
        data_cell = readtable(strcat(dataset_folder,data_file),opts);
        data(i,j) = {data_cell};

        data_file_target = strcat('s',string(j),'_',mode(i),'_','targets.csv');
        opts_target = detectImportOptions(strcat(dataset_folder,data_file_target));
        data_cell_target = readtable(strcat(dataset_folder,data_file_target),opts_target);
        target(i,j) = {data_cell_target};

        display(strcat(string(i)," ",string(j)))
    end
end

save("data.mat","data");
save("target.mat","target")