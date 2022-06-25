function [yp,h,a,z] = predictNN(thCA,X)                                     % FP
[~,L] = size(thCA);                                                         % hypothesis = a{L}
[m,~] = size(X);

a = cell([1,L]);
z = cell([1,L]);

a{1} = [ones([m 1]) X];

for l = 2:L
    z{l} = (thCA{1,l-1} * (a{l-1})')';
    a{l} = [ones([m 1]) sigmoid(z{l})];
end
a{L}(:,1) = [];

h = a{L};
[~,yp] = max(h,[],2);



