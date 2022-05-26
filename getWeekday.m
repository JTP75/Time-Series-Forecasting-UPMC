function dayName = getWeekday(dayNum)
switch dayNum
    case 2
        dayName = 'Mon';
    case 3
        dayName = 'Tue';
    case 4
        dayName = 'Wed';
    case 5
        dayName = 'Thu';
    case 6
        dayName = 'Fri';
    case 7
        dayName = 'Sat';
    case 1
        dayName = 'Sun';
    otherwise
        disp('Error in getWeekday(): Invalid dayNum entered.')
end
        