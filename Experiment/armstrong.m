
armstrongs = [];
for i = 1:100000
    
    istr = num2str(i);
    len = length(istr);
    
    sum = 0;
    for j = 1:len
        sum = sum + str2num(istr(:,j))^len;
    end
    
    if i == sum
        armstrongs = [armstrongs i]; %#ok<AGROW>
    end
    
end
disp(armstrongs')
