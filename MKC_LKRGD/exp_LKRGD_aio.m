clear;
clc;
data_path = fullfile(pwd, '..',  filesep, "data", filesep);
dirop = dir(fullfile(data_path, '*.mat'));
datasetCandi = {dirop.name};
exp_n = 'LKRGD';
LKRGD_res_agg(datasetCandi, exp_n);
