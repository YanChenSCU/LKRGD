function [A, beta, Ss, H_normalized] = LKRGDF(Ks, nCluster, opts)
%     Input
%         Ks: nSmp * nSmp * nKernel, kernel matrices
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
    opts.t = 5;
end

if ~isfield(opts, 'lambda')
    opts.lambda = 1; % fixed without tunning
end

if ~isfield(opts, 'gamma')
    opts.gamma = 1; % update adaptively
end

[nSmp, ~, nKernel] = size(Ks);
avgK = sum(Ks, 3) - 10^8*eye(nSmp);
[~, Idx] = sort(avgK, 1, 'descend');
Idx = Idx(1:opts.k, :);
Idx2 = Idx';
colIdx = Idx2(:);
Ik = opts.lambda * eye(opts.k);

Ss = cell(1, nKernel);
e = ones(1,opts.k);
z = zeros(opts.k,1);
e2 = ones(opts.k, 1);
options = [];
options.Display = 'off';
As = zeros(nSmp, opts.k, nKernel);
k = opts.k;
t = opts.t;
parfor i1 = 1:nKernel
    %**********************************************
    %  Step1:SK-LKR
    %  Complexity
    %         (1)avgKernel, n * n addition
    %         (2)knn, m * n * n, top-k quick selection is O(n)
    %         (3)quadprog, m * n k3
    %
    %**********************************************
    Ai = As(:, :, i1);
    Ki = Ks(:, :, i1);
    for iSmp = 1:nSmp
        idx = Idx(:, iSmp); 
        ki = Ki(idx, iSmp);
        Kii = Ki(idx, idx') + Ik;
        v = quadprog(Kii, -ki, [], [], e, 1, z, e2, [], options);
        Ai(iSmp, :) = v;
    end
    rowIdx = repmat((1:nSmp)', k, 1);
    val = Ai(:);
    G = sparse(rowIdx, colIdx, val, nSmp, nSmp,nSmp * k);
    
    %**********************************************
    %  Step2:SG-D
    %  [1] Diffusion Improves Graph Learning, NIPS, 2019
    %  [2] Learning with Local and Global Consistency, NIPS, 2003
    %  [3] The heat kernel as the pagerank of a graph-PNAS-2007
    %**********************************************
    %*********************************************
    % Step 1: The symmetric transition matrix
    %*********************************************
    Ssym = (G + G')/2;
    DSsym = 1./sqrt(max(sum(Ssym, 2), eps));
    Gnorm = (DSsym * DSsym') .* Ssym;
    Gnorm = (Gnorm + Gnorm')/2;
    Gnorm = sparse(Gnorm);
    %*********************************************
    % Step 2: Heat Kernel diffusion with close-form
    %*********************************************
    L = eye(nSmp) - Gnorm;
    L = full(L);
    S = expm(- t * L);
    Ss{i1} = sparse(S);
end
%clear avgK Idx Idx2 Ik colIdx i1 A Ki iSmp idx ki Kii val rowIdx e z e2 options  ;

%warning off;

%********************************************
% Init beta
%********************************************
beta = ones(nKernel,1)/nKernel;
%********************************************
% Init A
%********************************************
B = zeros(nSmp);
for i1 = 1:nKernel
    B = (1/beta(i1))*Ss{1, i1} + B;
end
B = B/sum(1./beta);
B = B - 1e8*eye(nSmp);
[~, Idx] = sort(B, 2, 'descend');
Idx = Idx(:, 1:k);
A = zeros(nSmp);
for iSmp = 1:nSmp
    idxa0 = Idx(iSmp, :);
    ad = B(iSmp, idxa0);
    A(iSmp, idxa0) = EProjSimplex_new(ad);
end
%********************************************
% Init H
%********************************************
A2 = A + A';
L = diag(sum(A2,2)) - A2;
L = (L + L')/2;
[H, ~] = eigs(L, nCluster, 'SA');

objHistory = [];
converges = false;
iter = 0;
maxIter = 20;
while ~converges
    %*******************************************
    % Update A
    %*******************************************
    B = zeros(nSmp);
    for i1 = 1:nKernel
        B = (1/beta(i1))*Ss{1, i1} + B;
    end
    DH = EuDist2(H, H, 0);
    BDH = B - opts.gamma * DH;
    BDH = BDH/sum(1./beta) ;
    BDH = BDH - 1e8*eye(nSmp);
    [~, Idx] = sort(BDH, 2, 'descend');
    Idx = Idx(:, 1:k);
    A = zeros(nSmp);
    for iSmp = 1:nSmp
        idxa0 = Idx(iSmp, :);
        ad = BDH(iSmp, idxa0);
        A(iSmp, idxa0) = EProjSimplex_new(ad);
    end
    %*******************************************
    % Update H
    %*******************************************
    H_old = H;
    A2 = A + A';
    L = diag(sum(A2,2)) - A2;
    L = (L + L')/2;
    [H, eigValue] = eigs(L, nCluster+1, 'SA');
    H = H(:, 1:nCluster);
    eigValue = diag(eigValue);
    sum_ev_1 = sum(eigValue(1:nCluster));
    sum_ev = sum(eigValue(1:nCluster+1));
    if sum_ev_1 > 0.00000000001
        opts.gamma = 2 * opts.gamma;
    elseif sum_ev < 0.00000000001
        opts.gamma = opts.gamma / 2;
        H = H_old;
    else
        converges = true;
    end
    
    %*******************************************
    % Update beta
    %*******************************************
    e = zeros(nKernel, 1);
    for i1 = 1:nKernel
        Ei = A - Ss{1, i1};
        e(i1) = sum(sum(Ei.^2));
    end
    beta = sqrt(e)/sum(sqrt(e));
    
    LH = L * H;
    o2 = sum(sum(H.*LH));
    obj = sum(e./max(beta, 1e-10)) + opts.gamma*o2;
    objHistory = [objHistory; obj]; %#ok
    
    iter = iter + 1;
    if ((iter > 3) && (objHistory(end-1) - objHistory(end))/objHistory(end) < 1e-6 ) || iter >maxIter
        converges = true;
    end
end
disp(['The final gamma=', num2str(opts.gamma)]);
if nargout == 4
    Z = (A + A')/2;
    CKSym = BuildAdjacency(thrC(Z, 0.7));
    H_normalized = SpectralClustering_ncut(CKSym, nCluster);
end
end