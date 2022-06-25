function numbers = labels2numbers(labels)
[m,~] = size(labels);
unique_labels = {''};
numbers = zeros([m,1]);
labnum = 0;

if isa(labels,'cell')
    for i = 1:m
        label = labels{i};
        isLabel = cellfun(@(x)isequal(x,label),unique_labels);
        if(sum(find(isLabel))==0)
            unique_labels{1,labnum+1} = label; 
            labnum = labnum + 1;
        end
        isLabel = cellfun(@(x)isequal(x,label),unique_labels);
        numbers(i,1) = find(isLabel);
    end
end
