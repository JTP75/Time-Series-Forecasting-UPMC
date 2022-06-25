function gradJ = loss_mse_prime(y, yp)
    [m,~] = size(y);
    gradJ = 2*(yp-y)/m;
end