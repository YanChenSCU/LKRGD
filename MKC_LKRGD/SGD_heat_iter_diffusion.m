function [Ss] = SGD_heat_iter_diffusion(Gs, t)
% [1] Diffusion Improves Graph Learning, NIPS, 2019
% [2] Learning with Local and Global Consistency, NIPS, 2003
% [3] The heat kernel as the pagerank of a graph-PNAS-2007
if ~exist('t', 'var')
    t = 1;
end
nGraph = length(Gs);

Ss = cell(1, nGraph);
for i1 = 1:nGraph
    %*********************************************
    % Step 1: The symmetric transition matrix
    %*********************************************
    Gsym = (Gs{i1} + Gs{i1}')/2;
    DGsym = 1./sqrt(max(sum(Gsym, 2), eps));
    Gnorm = (DGsym * DGsym') .* Gsym;
    Gnorm = (Gnorm + Gnorm')/2;
    Gnorm = sparse(Gnorm);
    %*********************************************
    % Step 2: Heat Kernel diffusion with close-form
    %*********************************************
    opt = [];
    opt.t = t;
    Si = heat_kernel_diffusion_iter(Gnorm, opt);
    Ss{i1} = sparse(Si);
end
end