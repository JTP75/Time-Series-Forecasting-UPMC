function D2 = mseDIST(ZI,ZJ)
[m,~] = size(ZJ);
D2 = zeros([m,1]);

for i = 1:m
    D2(i) = sum((ZI-ZJ(i,:)).^2);
end











