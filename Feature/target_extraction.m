clear
close all
clc

load target.mat;
windows_num = 23;
target_matrix = zeros(45*3*22,5);

for i = 1:3
    for j = 1:22
        activity_target = table2array(target{i,j}(:,2));
        rows = size(activity_target, 1);
        dim = floor(rows/windows_num);
        mode_offset = (i-1) * 22 * (windows_num*2-1);
        person_offset = (j-1) * (windows_num*2-1);
        offset = mode_offset + person_offset;
        disp(string(offset));

        for k = 1:windows_num
            start_row = (k-1) * dim + 1;
            end_row = start_row + dim - 1;
            window_row = (k-1) * 2 + 1;
            target_matrix(offset+window_row,1) = mean(activity_target(start_row:end_row));
            target_matrix(offset+window_row,2) = std(activity_target(start_row:end_row));
            target_matrix(offset+window_row,3:5) = full(ind2vec(i,3))';
        end

        for k = 1:(windows_num-1)
            start_row = floor((k-1/2)*dim) + 1;
            end_row = start_row + dim - 1;
            window_row = k * 2;
            target_matrix(offset+window_row,1) = mean(activity_target(start_row:end_row));
            target_matrix(offset+window_row,2) = std(activity_target(start_row:end_row));
            target_matrix(offset+window_row,3:5) = full(ind2vec(i,3))';
        end

        disp(strcat(string(i)," ",string(j)))
    end
end

save("target_matrix.mat","target_matrix");