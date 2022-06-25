function thCA = sGD(szs, X, y, lam, alpha, MaxEpochs)                       % sz = 4,8,3
[m,~] = size(X);
[~,L] = size(szs);
K = szs(L);

% rng('default');

thCA = cell([2,L]);
for l = 1:L-1
    thCA{1,l} = 1 - 2 * rand([ szs(l+1) , szs(l)+1 ]);
    thCA{2,l} = size(thCA{1,l});
end

ybin = zeros([m,K]);
for k = 1:K
    ybin(:,k) = y==k;
end

d = cell([1,L]);
delta = cell([1,L]);
gradJ = cell([1,L]);
for epoch = 1:MaxEpochs
    
    Jnow = nnCost(thCA, X, y, K, lam);
    
    for i = 1:m
        
        [yp(m,1),~,a,z] = predictNN(thCA,X(m,:));
        
        d{L} = a{L} - ybin(m,:);
        delta{L-1} = (d{L})' * a{L-1};
        gradJ{L-1} = [ delta{L-1}(:,1), ( delta{L-1}(:,2:szs(L-1)+1) + lam * thCA{1,L-1}(:,2:szs(L-1)+1) ) ];
        thCA{1,L-1} = thCA{1,L-1} - alpha*gradJ{L-1};
        
        for l = L-1:-1:2
            
            d{l} = ( d{l+1} * thCA{1,l}(:,2:szs(l)+1) ) .* sigmoid_prime(z{l});
            delta{l-1} = (d{l})' * a{l-1};
            gradJ{l-1} = [ delta{l-1}(:,1), ( delta{l-1}(:,2:szs(l-1)+1) + lam * thCA{1,l-1}(:,2:szs(l-1)+1) ) ];
            thCA{1,l-1} = thCA{1,l-1} - alpha*gradJ{l-1};
            
        end
        
    end
    
    fprintf('%d/%d\t\tCost = %.4f\n',epoch,MaxEpochs,Jnow)
    
    Jlast = Jnow;
    clear yp
end








