function [X_train, y_train, X_test, y_test] = splitData(X, y, training_proportion)

[m_examples, n_features] = size(X);                                         % get size of x
ti = randperm(m_examples, round(m_examples*training_proportion));           % create array of random indices (in X1)
for i = 1:round(m_examples*training_proportion)                             % fill training data with training indices
    X_train(i,1:n_features) = X(ti(i),1:n_features);                        % **
    y_train(i,1) = y(ti(i));                                                % **
end                                                                         % **
k = 1;                                                                      % temp k is index for test data
for i = 1:m_examples                                                        % loop i through examples
    match = 0;                                                              % set (or reset) match to zero
    for j = 1:round(m_examples*training_proportion)                         %   loop through training examples to test
        if i == ti(j)                                                       %   if index i is equal to any of them. if
            match = 1;                                                      %   so, then there is a match, so set match
        end                                                                 %   to 1 (true)
    end                                                                     % **
    if match == 0                                                           % skip if there is a match
        X_test(k,1:n_features) = X(i,1:n_features);                         % **
        y_test(k,1) = y(i);                                                 % **
        k = k + 1;                                                          % **
    end                                                                     % ****** fills testing sets with data that is
end                                                                         % ****** not already in the training set







