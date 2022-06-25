function J = nnCost(thCA, X, y, K, lam)
[~,L] = size(thCA);
[m,~] = size(X);

regSummand = 0;
thCAReduced = thCA;                                                         % removes bias terms for regularizing
for l = 1:L-1
    thCAReduced{1,l}(:,1) = [];
    thCAReduced{2,l} = size(thCAReduced{1,l});
    regSummand = regSummand + sum(thCAReduced{l}.^2,'all');
end

yabin = zeros([m K]);
[~,ypbin,~,~] = predictNN(thCA, X);

errorSummand = 0;
for k = 1:K
    yabin(:,k) = (y==k);
    errorSummand = ...
        errorSummand + ( sum( yabin(:,k) .* log(ypbin(:,k)) + ...
        (1-yabin(:,k)) .* log(1-ypbin(:,k)) ) );
end

J = -1/m * errorSummand + lam/2/m * regSummand;






