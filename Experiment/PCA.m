function eigvecs = PCA(A_centered,desired_variance)

workingmat = A_centered;
kmax = size(workingmat,2)/2;

covmat = workingmat' * workingmat;
all_eigvals = flip(eig(covmat));
total_variance = sum(all_eigvals);
for k = 1:kmax
    captured_variance = sum(all_eigvals(1:k))/total_variance; 
    if captured_variance >= desired_variance
        K = k;
        break
    end
end
[eigvecs,~] = eigs(covmat,K);
