function ybin = toBinary(y,K)

ybin = [];
for k = 1:K
    ybin = [ybin,y==k];
end

