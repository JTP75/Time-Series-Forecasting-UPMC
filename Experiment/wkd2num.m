function wnum = wkd2num(wnam)

if ~isa(wnam,'char')
    ME = MException('wkd2num:bad_arg', 'Argument must be a string');
    throw(ME)
end

switch wnam
    case 'Sun'
        wnum = 1;
    case 'Mon'
        wnum = 2;
    case 'Tue'
        wnum = 3;
    case 'Wed'
        wnum = 4;
    case 'Thu'
        wnum = 5;
    case 'Fri'
        wnum = 6;
    case 'Sat'
        wnum = 7;
    otherwise
        ME = MException('wkd2num:invalid_str', 'Invalid weekday string: ', wnam);
        throw(ME)
end