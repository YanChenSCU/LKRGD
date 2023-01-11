function LKRGD_res_agg(datasetCandi, exp_n)
data_path = fullfile(pwd, '..',  filesep, "data", filesep);
addpath(data_path);
lib_path = fullfile(pwd, '..',  filesep, "lib", filesep);
addpath(lib_path);
eval([exp_n, '_aio = zeros(length(datasetCandi), 11);']);
eval([exp_n, '_grid_aio = cell(1, length(datasetCandi));']);
for i1 = 1 : length(datasetCandi)
    data_name = datasetCandi{i1}(1:end-4);
    dir_name = [pwd, filesep, exp_n, filesep, data_name];
    try
        if ~exist(dir_name, 'dir')
            mkdir(dir_name);
        end
        prefix_mdcs = dir_name;
    catch
        disp(['create dir: ',dir_name, 'failed, check the authorization']);
    end
    
    %*********************************************************************
    % SK2DG
    %*********************************************************************
    fname2 = fullfile(prefix_mdcs, [data_name, '_12k_', exp_n, '.mat']);
    if exist(fname2, 'file')
        eval(['clear ', exp_n, '_result_summary']);
        load(fname2);
        eval([exp_n, '_aio(i1,:) = max(', exp_n, '_result_summary, [], 1);']);
        eval([exp_n, '_grid_aio{i1} = ', exp_n, '_result;']);
    end
end
eval(['save(''', exp_n, '-AIO.mat'',''', exp_n, '_aio'',''',exp_n, '_grid_aio'',', '''datasetCandi'');']);
rmpath(data_path);
rmpath(lib_path);
end