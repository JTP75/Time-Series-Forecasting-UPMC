function s = tuple2str(tup,varargin)

% defaults
delimiter = ',';
paranthetical = '(';

for i = 1:length(varargin)
    if strcmp(varargin{i},'Delim')
        delimiter = varargin{i+1};
    elseif strcmp(varargin{i},'Paran')
        paranthetical = varargin{i+1};
    end
end

if paranthetical == '('
    closer = ')';
elseif paranthetical == '['
    closer = ']';
elseif paranthetical == '{'
    closer = '}';
end

s = [paranthetical num2str(tup(1)) delimiter num2str(tup(2)) closer];









