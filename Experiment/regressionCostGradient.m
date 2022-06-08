function gradJ = regressionCostGradient(X, y, th)
[m n] = size(X);

for j = 1:n
    gradJ(j) = sum((X*th-y)'*(X(:,j)));
end