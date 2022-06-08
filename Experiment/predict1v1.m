function yp = predict1v1(mdls1v1, X)
[K,~] = size(mdls1v1);                                                      % get K from cell arr input
[m,~] = size(X);                                                            % get size of input data

yp_mat_Bin = zeros([K, K-1, m]);                                            % initialize 3d matrix for binary predictions
yp_mat = yp_mat_Bin;                                                        % initialize 3d matrix for non-Bin predictions
for i = 1:K
    for j = i+1:K
        yp_mat_Bin(i,j,:) = predict(mdls1v1{i,j},X);                        % prediction of each model, for each smpl
        yp_mat(i,j,:) = i*(yp_mat_Bin(i,j,:)==1)+j*(yp_mat_Bin(i,j,:)==0);  % if model predicts 1, set it to i
    end                                                                     % if model predicts 0, set it to j
end
for p = 1:m
    yp(p,1) = mode(nonzeros(yp_mat(:,:,p)));                                % modes of all ij models determine predictions
end