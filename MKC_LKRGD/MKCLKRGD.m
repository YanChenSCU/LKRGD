function [H_normalized, A, beta, Ss, Gs] = MKCLKRGD(Ks, nCluster, opts, data_name)
%     Input
%         Ks: nSmp * nSmp * nGraph, kernel matrices
%         nCluster:
%         opts: configuaration
%             k, the neighobor size used in LKR and graph sparsification
%             lambda, the regularization parameter in LKR, default 1
%             t, heat kernel diffusion
%             gamma, the adaptive parameter of rank(L) = n-c, default 1
%
%

if ~exist('opts', 'var')
    opts = [];
end

if ~isfield(opts, 'k')
    opts.k = 5;
end

if ~isfield(opts, 't')
    opts.t = 1;
end

if ~isfield(opts, 'lambda')
    opts.lambda = 1;
end

if ~isfield(opts, 'gamma')
    opts.gamma = 1;
end
if exist('data_name', 'var') && ~isempty(data_name)
    fname = fullfile([data_name, '_MKLKR_Gs_k=', num2str(opts.k), '_lambda=', num2str(opts.lambda), '.mat']);
    if exist(fname, 'file')
        load(fname, 'Gs');
    end
end
if ~exist('Gs', 'var')
    Gs = MKLKR(Ks, opts.k, opts.lambda);
    if exist('data_name', 'var') && ~isempty(data_name)
        fname = fullfile([data_name, '_MKLKR_Gs_k=', num2str(opts.k), '_lambda=', num2str(opts.lambda), '.mat']);
        save(fname, 'Gs');
    end
end

clear Ks;
Ss = SGD_heat(Gs, opts.t, opts.k);
[H_normalized, A, beta] = MSG_fusion(Ss, nCluster, opts.k, opts.gamma);
end