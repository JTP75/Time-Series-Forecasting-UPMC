function y_cat = num2cat(y_num)

flipout = false;
if size(y_num,1) ~= 1
    y_num = y_num';
    flipout = true;
end

Kset = [];
catset = {};
for k = y_num
    if sum(ismember(k,Kset))==0
        Kset = [Kset, k];
        catset{end+1} = num2str(k);
    end
end

y_cat = categorical(y_num,Kset,catset);

if flipout
    y_cat = y_cat';
end

ok = 1;


