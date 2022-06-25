function J = loss_mse(y, yp)
    J = mean( (y-yp).^2 );
end