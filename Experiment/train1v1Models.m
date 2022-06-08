function mdls = train1v1Models(X, y, K)

[m, n] = size(X);                                                           % size of X
mdls = {K-1,K};                                                             % mdl array (stored as 2D cell array)

for i = 1:K                                                                 % i runs from 1 to K
    for j = i+1:K                                                           % j runs from i+1 to K
        l=0;                                                                % exterior index for Xi/yi and Xj/yj
        o=0;                                                                % **
        for k = 1:m                                                         % for each sample:
            if y(k) == i                                                    % if smpl label = i:
                l=l+1;                                                      % increment Xi/yi idx
                Xi(l,:) = X(k,:);                                           % add smpl to Xi
                yi(l,1) = 1;                                                % set smpl label in yi to 1
            elseif y(k) == j                                                % if smpl label = j: 
                o=o+1;                                                      % increment Xj/yj idx
                Xj(o,:) = X(k,:);                                           % add smpl to Xj
                yj(o,1) = 0;                                                % set smpl label in yj to 0;
            end
        end
        Xij = [Xi ; Xj];                                                    % vertically concatenate Xi and Xj into Xij
        yij = [yi ; yj];                                                    % vertically concatenate yi and yj into yij
        mdls{i,j} = fitcsvm(Xij, yij);                                      % add i vs j model to mdl array
    end
end

% NOTE
% in model array, the column number / row number pair corresponds to column
% number vs row number model. however, the bottom left half is empty
% i.e. mdls{4,6} corresponds to the model comparing classes 4 and 6, but
% mdls{6,4} is an empty cell








