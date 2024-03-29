function [features] = f_extract(data_array)
    time_features = [
        mean(data_array)...
        std(data_array)...
        max(data_array)...
        min(data_array)...
        median(data_array)...
        geomean(data_array)...
        harmmean(data_array)...
        trimmean(data_array,10)...
        skewness(data_array)...
        kurtosis(data_array)...
        iqr(data_array)
        ];
    freq_data_array = abs(fft(data_array));
    freq_features = [
        mean(freq_data_array)...
        std(freq_data_array)...
        max(freq_data_array)...
        min(freq_data_array)...
        obw(freq_data_array)...
        median(freq_data_array)...
        geomean(freq_data_array)...
        harmmean(freq_data_array)...
        trimmean(freq_data_array,10)...
        skewness(freq_data_array)...
        kurtosis(freq_data_array)...
        iqr(freq_data_array)
        ];
    features = [time_features, freq_features];
end

