function th = regressionGradientDescent(Xtr, ytr, a, maxIters)
[~,n] = size(Xtr);
th = rand([1 n])';

for i = 1:maxIters
    Jnow = gather(regressionCost(Xtr, ytr, th))
    gradJ = regressionCostGradient(Xtr, ytr, th);
    th = th - a*abs(gradJ)';
    if i ~= 1 && abs(Jnow - Jlast) <= 0.001
        break
    end
    Jlast = Jnow;
end

