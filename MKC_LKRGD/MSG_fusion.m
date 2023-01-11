function [H_normalized, A, beta] = MSG_fusion(Gs, nCluster, k, gamma)
warning off;
if ~exist('k', 'var')
    k = 5;
end

if ~exist('gamma', 'var')
    gamma = 1;
end

nGraph = length(Gs);
nSmp = size(Gs{1}, 1);
%********************************************
% Init beta
%********************************************
beta = ones(nGraph,1)/nGraph;
%********************************************
% Init A
%********************************************
B = zeros(nSmp);
for i1 = 1:nGraph
    B = (1/beta(i1))*Gs{1, i1} + B;
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
    for i1 = 1:nGraph
        B = (1/beta(i1))*Gs{1, i1} + B;
    end
    DH = EuDist2(H, H, 0);
    BDH = B - gamma * DH;
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
        gamma = 2 * gamma;
    elseif sum_ev < 0.00000000001
        gamma = gamma / 2;
        H = H_old;
    else
        converges = true;
    end
    
    %*******************************************
    % Update beta
    %*******************************************
    e = zeros(nGraph, 1);
    for i1 = 1:nGraph
        Ei = A - Gs{1, i1};
        e(i1) = sum(sum(Ei.^2));
    end
    beta = sqrt(e)/sum(sqrt(e));
    
    LH = L * H;
    o2 = sum(sum(H.*LH));
    obj = sum(e./max(beta, 1e-10)) + gamma*o2;
    objHistory = [objHistory; obj]; %#ok
    
    iter = iter + 1;
    if ((iter > 3) && (objHistory(end-1) - objHistory(end))/objHistory(end) < 1e-6 ) || iter >maxIter
        converges = true;
    end
end
H_normalized = bsxfun(@rdivide, H, max(sqrt(sum(H.^2, 2)), eps));
end