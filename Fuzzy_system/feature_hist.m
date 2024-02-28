clear
close all
clc

%prepare dataset
load("..\Feature\feature_normalized_range.mat");
load("..\Feature\target_matrix.mat");
x = feature_normalized_range(:,[4 6 1]);
t = target_matrix(:,3:5);
t_index = vec2ind(t')';

run_rows = x(t_index == 1, :);
sit_rows = x(t_index == 2, :);
walk_rows = x(t_index == 3, :);

figure; 
tiledlayout("flow");

%% FEATURE 1
i = 1;
ax1 = nexttile(0+i);
histogram(run_rows(:,i), 20, 'BinWidth', 0.025, 'BinLimits',[0, 1]);
title("Histogram of feature " + num2str(i) + ": RUN");
xticks(linspace(0, 1, 11));
xline([0.5 0.7 0.85],'-r','LineWidth',1);

ax2 = nexttile(3+i);
histogram(sit_rows(:,i), 20, 'BinWidth', 0.025, 'BinLimits',[0, 1]);
title("Histogram of feature " + num2str(i) + ": SIT");
xticks(linspace(0, 1, 11));
xline([0.5 0.7 0.85],'-r','LineWidth',1);

ax3 = nexttile(6+i);
histogram(walk_rows(:,i), 20, 'BinWidth', 0.025, 'BinLimits',[0, 1]);
title("Histogram of feature " + num2str(i) + ": WALK");
xticks(linspace(0, 1, 11));
xline([0.5 0.7 0.85],'-r','LineWidth',1);

%% FEATURE 2
i = 2;
ax4 = nexttile(0+i);
histogram(run_rows(:,i), 20, 'BinWidth', 0.025, 'BinLimits',[0, 1]);
title("Histogram of feature " + num2str(i) + ": RUN");
xticks(linspace(0, 1, 11));
xline([0.2 0.4 0.65],'-r','LineWidth',1);

ax5 = nexttile(3+i);
histogram(sit_rows(:,i), 20, 'BinWidth', 0.025, 'BinLimits',[0, 1]);
title("Histogram of feature " + num2str(i) + ": SIT");
xticks(linspace(0, 1, 11));
xline([0.2 0.4 0.65],'-r','LineWidth',1);

ax6 = nexttile(6+i);
histogram(walk_rows(:,i), 20, 'BinWidth', 0.025, 'BinLimits',[0, 1]);
title("Histogram of feature " + num2str(i) + ": WALK");
xticks(linspace(0, 1, 11));
xline([0.2 0.4 0.65],'-r','LineWidth',1);

%% FEATURE 3
i = 3;
ax7 = nexttile(0+i);
histogram(run_rows(:,i), 20, 'BinWidth', 0.025, 'BinLimits',[0, 1]);
title("Histogram of feature " + num2str(i) + ": RUN");
xticks(linspace(0, 1, 11));
xline([0.1 0.4 0.6],'-r','LineWidth',1);

ax8 = nexttile(3+i);
histogram(sit_rows(:,i), 20, 'BinWidth', 0.025, 'BinLimits',[0, 1]);
title("Histogram of feature " + num2str(i) + ": SIT");
xticks(linspace(0, 1, 11));
xline([0.1 0.4 0.6],'-r','LineWidth',1);

ax9 = nexttile(6+i);
histogram(walk_rows(:,i), 20, 'BinWidth', 0.025, 'BinLimits',[0, 1]);
title("Histogram of feature " + num2str(i) + ": WALK");
xticks(linspace(0, 1, 11));
xline([0.1 0.4 0.6],'-r','LineWidth',1);

linkaxes([ax1 ax2 ax3 ax4 ax5 ax6 ax7 ax8 ax9 ], 'xy');
