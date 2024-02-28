clear
close all
clc

load data.mat;
windows_num = 23;
feature = zeros(45*3*22,11*23);

for i = 1:3
    for j = 1:22
        activity_data = table2array(data{i,j}(:,2:12));
        rows = size(activity_data, 1);
        dim = floor(rows/windows_num);
        mode_offset = (i-1) * 22 * (windows_num*2-1);
        person_offset = (j-1) * (windows_num*2-1);
        offset = mode_offset + person_offset;
        display(string(offset));

        for k = 1:windows_num
            start_row = (k-1) * dim + 1;
            end_row = start_row + dim - 1;
            window_row = (k-1) * 2 + 1;
            feature(offset+window_row,:) = f_extract(activity_data(start_row:end_row,:));
        end

        for k = 1:(windows_num-1)
            start_row = floor((k-1/2)*dim) + 1;
            end_row = start_row + dim - 1;
            window_row = k * 2;
            feature(offset+window_row,:) = f_extract(activity_data(start_row:end_row,:));
        end

        display(strcat(string(i)," ",string(j)))
    end
end

%Remove correlated feature
correlation_coef = corrcoef(feature);
[correlated_features, ~] = find( tril( (abs(correlation_coef) > 0.9), -1 ) );
correlated_features = unique(sort(correlated_features));

feature_without_correlated_features = feature;
feature_without_correlated_features(:, correlated_features) = [];

%Normalization of the matrix
feature_normalized_zscore = normalize(feature_without_correlated_features,'zscore');
feature_normalized_norm = normalize(feature_without_correlated_features,'norm');
feature_normalized_center = normalize(feature_without_correlated_features,'center');
feature_normalized_scale = normalize(feature_without_correlated_features,'scale');
feature_normalized_range = normalize(feature_without_correlated_features,'range');
feature_normalized_medianiqr = normalize(feature_without_correlated_features,'medianiqr');

save("feature_without_correlated_features.mat","feature_without_correlated_features");
save("feature_normalized_zscore.mat","feature_normalized_zscore");
save("feature_normalized_norm.mat","feature_normalized_norm");
save("feature_normalized_center.mat","feature_normalized_center");
save("feature_normalized_scale.mat","feature_normalized_scale");
save("feature_normalized_range.mat","feature_normalized_range");
save("feature_normalized_medianiqr.mat","feature_normalized_medianiqr");
