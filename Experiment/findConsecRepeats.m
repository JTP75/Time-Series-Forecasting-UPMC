function idcs = findConsecRepeats(vec)
m = length(vec);
idcs = [];
for i = 1:m-1
    if(vec(i,1) == vec(i+1,1))
        idcs = [idcs i];
    end
end