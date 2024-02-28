clear
close all
clc

%prepare dataset
load("..\Feature\feature_normalized_range.mat");
load("..\Feature\target_matrix.mat");
x = feature_normalized_range(:,[4 6 1]);
t = target_matrix(:,3:5);
t_index = vec2ind(t')';

% Generate FIS 
genfis_options = genfisOptions('GridPartition', ...
                               NumMembershipFunctions = 5);

fis = genfis(x, t_index, genfis_options);

% Tune FIS
[in, out, rules] = getTunableSettings(fis);
tunefis_options = tunefisOptions('Method','anfis');
tunefis_options.MethodOptions.EpochNumber = 30;
tunefis_options.UseParallel = true;
fis_final = tunefis(fis, [in; out], x, t_index, tunefis_options);

writeFIS(fis_final, 'fis_generated.fis');
