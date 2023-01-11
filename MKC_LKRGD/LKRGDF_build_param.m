function paramCell = LKRGDF_build_param(ks, ts)

if ~exist('ks', 'var')
    ks = [5:2:15];
end

if ~exist('ts', 'var')
    ts = [0.1,1,5,10];
end

nParam = length(ks) * length(ts);
paramCell = cell(nParam, 1);
idx = 0;
for i1 = 1:length(ks)
    for i2 = 1:length(ts)
        param = [];
        param.k = ks(i1);
        param.t = ts(i2);
        idx = idx + 1;
        paramCell{idx,1} = param;
    end
end
