%
%
%
clear;
clc;
data_path = fullfile(pwd, '..',  filesep, "data", filesep);
addpath(data_path);
lib_path = fullfile(pwd, '..',  filesep, "lib", filesep);
addpath(lib_path);
code_path = genpath(fullfile(pwd, '..',  filesep, 'MKC_LKRGD'));
addpath(code_path);


dirop = dir(fullfile(data_path, '*.mat'));
datasetCandi = {dirop.name};
% datasetCandi = {'COIL20_1440n_1024d_20c_uni.mat','binaryalphadigs_1404n_320d_36c_uni.mat',...
%     'mfeat_pix_2000n_240d_10c_uni.mat','FACS_v2_Trachea-counts_1013n_13741d_7c_uni.mat',...
%     'MNIST_4000n_784d_10c_uni.mat','Macosko_6418n_8608d_39c_uni.mat'};

datasetCandi = {'hitech_2301n_22498d_6c_tfidf_uni.mat'};

exp_n = 'LKRGDF';
% profile off;
% profile on;
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
    
    clear X y Y;
    load(data_name);
    if exist('y', 'var')
        Y = y;
    end
    if size(X, 1) ~= size(Y, 1)
        Y = Y';
    end
    assert(size(X, 1) == size(Y, 1));
    nSmp = size(X, 1);
    nCluster = length(unique(Y));
    
    %*********************************************************************
    % LKRGDF
    %*********************************************************************
    fname2 = fullfile(prefix_mdcs, [data_name, '_12k_LKRGDF.mat']);
    if ~exist(fname2, 'file')
        Xs = cell(1,1);
        Xs{1} = X;
        Ks = Xs_to_Ks_12k(Xs);
        Ks2 = Ks{1,1};
        Ks = Ks2;
        clear Ks2 Xs;
        [nSmp, ~, nKernel] = size(Ks);
        
        nRepeat = 10;
        k_range = [5,10,15,20];
        t_range = [3,5,7,9];
        paramCell = LKRGDF_build_param(k_range, t_range);
        nParam = length(paramCell);
        LKRGDF_result = zeros(nParam, 1, nRepeat, 10);
        LKRGDF_time = zeros(nParam, 1);
        for iParam = 1:nParam
            disp(['LKRGDF iParam= ', num2str(iParam), ', totalParam= ', num2str(nParam)]);
            fname3 = fullfile(prefix_mdcs, [data_name, '_12k_LKRGDF_', num2str(iParam), '.mat']);
            if exist(fname3, 'file')
                load(fname3, 'result_10_s', 'tt');
                LKRGDF_time(iParam) = tt;
                for iRepeat = 1:nRepeat
                    LKRGDF_result(iParam, 1, iRepeat, :) = result_10_s(iRepeat, :);
                end
            else
                param = paramCell{iParam};
                opts = [];
                opts.k = param.k;
                opts.t = param.t;
                tic;
                [A, beta, Ss] = LKRGDF(Ks, nCluster, opts);
                A2 = ( abs(A) + abs(A') ) / 2 ;
                CKSym = BuildAdjacency(thrC(A2,0.7));
                H_normalized = SpectralClustering_ncut(CKSym,nCluster);                
                tt = toc;
                LKRGDF_time(iParam) = tt;
                result_10_s = zeros(nRepeat, 10);
                for iRepeat = 1:nRepeat
                    label = kmeans(H_normalized, nCluster, 'emptyaction', 'singleton', 'replicates', 100, 'display', 'off');
                    result_10 = my_eval_y(label, Y);
                    LKRGDF_result(iParam, 1, iRepeat, :) = result_10';
                    result_10_s(iRepeat, :) = result_10';
                end
                save(fname3, 'result_10_s', 'tt', 'param');
            end
        end
        a1 = sum(LKRGDF_result, 2);
        a3 = sum(a1, 3);
        a4 = reshape(a3, size(LKRGDF_result,1), size(LKRGDF_result,4));
        LKRGDF_grid_result = a4/nRepeat;
        LKRGDF_result_summary = [max(LKRGDF_grid_result, [], 1), sum(LKRGDF_time)/nParam];
        save(fname2, 'LKRGDF_result', 'LKRGDF_grid_result', 'LKRGDF_time', 'LKRGDF_result_summary');
        
        disp([data_name, ' has been completed!']);
    end
end
rmpath(data_path);
rmpath(lib_path);
rmpath(code_path);
% profile viewer;