function As = MKLKR(Ks, knn_size, lambda)
%
%  Complexity
%         (1)avgKernel, n * n addition
%         (2)knn, n * n, top-k quick selection is O(n)
%         (3)quadprog, m * n k3
%

if ~exist('knn_size', 'var')
    knn_size = 5;
end
if ~exist('lambda', 'var')
    lambda = 1;
end
[nSmp, ~, nKernel] = size(Ks);

avgK = sum(Ks, 3) - 10^8*eye(nSmp);
[~, Idx] = sort(avgK, 1, 'descend');
Idx = Idx(1:knn_size, :);
Idx2 = Idx';
colIdx = Idx2(:);
Ik = lambda * eye(knn_size);

As = cell(1, nKernel);
e = ones(1,knn_size);
z = zeros(knn_size,1);
e2 = ones(knn_size, 1);
options = [];
options.Display = 'off';
parfor i1 = 1:nKernel
    A = zeros(nSmp, knn_size);
    Ki = Ks(:, :, i1);
    for iSmp = 1:nSmp
        idx = Idx(:, iSmp); %#ok
        ki = Ki(idx, iSmp);
        Kii = Ki(idx, idx') + Ik;
        v = quadprog(Kii, -ki, [], [], e, 1, z, e2, [], options);
        A(iSmp, :) = v;
    end
    rowIdx = repmat((1:nSmp)', knn_size, 1);
    val = A(:);
    As{1, i1} = sparse(rowIdx, colIdx, val, nSmp, nSmp,nSmp * knn_size);
end
end