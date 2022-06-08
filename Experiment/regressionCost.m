function J = regressionCost(X, y, th)
[m n] = size(X);
J = sum((X*th-y).^2)/2/m;