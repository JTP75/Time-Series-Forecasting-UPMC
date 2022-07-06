classdef NedocData
    
    properties
        % tables
        T_org
        T_imp                               % imputed table:    missing observations are filled with linearly spaced scores
        
        % input matrix struct
        X                                   % X (contains serial date and time, weekday, and month)
        X_t                                 % X_t (contains only serial date-time)
        X_lag                               % lag matrix
        
        % target & response vector structs
        y                                   % target
        yp                                  % response
        
        % boundary struct
        today                               % todays date struct (marks training/testing split)
        
        % definitions
        L                                   % length
        mu                                  % mean (TRAINING DATA)
        sig                                 % stdev (TRAINING DATA)
        PPD                                 % num of obs per day
    end
    
    methods
        
        % constructor
        function o = NedocData(TI,NPPD)
            o.T_org = TI;
            o = o.setPPD(NPPD);
            o.T_imp = o.T_org;
            
            o.X = [ (o.T_imp.Date) (o.T_imp.Time) (o.T_imp.Weekday) (o.T_imp.Month) ];
            o.X_t = o.T_imp.Date_Time;
            o.X_lag = [];
            
            o.L = height(o.T_imp);
        end
        
        % accessors
        function [X,y] = getmats(o,setspec,fulTimLag,std)
            if strcmp(setspec,'train')
                idcs = 1:o.today.i;
            elseif strcmp(setspec,'test')
                idcs = (o.today.i + 1):o.L.i;
            elseif strcmp(setspec,'all')
                idcs = 1:o.L.i;
            else
                desc = 'setspec (first) arg must be ''train'', ''test'', or ''all''';
                ME = MException('NedocData:getmats:Invalid_Arg',desc);
                throw(ME)
            end
            
            if strcmp(fulTimLag,'full')
                X = o.X(idcs,:);
                y = o.y(idcs);
            elseif strcmp(fulTimLag,'time')
                X = o.X_t(idcs,:);
                y = o.y(idcs);
            elseif strcmp(fulTimLag,'lag')
                X = o.X_lag(idcs,:);
                y = o.y(idcs);
            else
                desc = 'matrix specifier (second) arg must be ''full'', ''time'', or ''lag''';
                ME = MException('NedocData:getmats:Invalid_Arg',desc);
                throw(ME)
            end
            
            if nargin >= 4 && strcmp(std,'std')
                y = (y-o.mu)./o.sig;
            end
        end
        
        % mutators
        function o = setResp(o,ypred)
            if length(ypred) ~= o.L
                desc = ['Length of arg ' num2str(length(ypred)) 'does not match imputed length ' num2str(o.L)];
                ME = MException('NedocData:setResp:Invalid_Arg',desc);
                throw(ME)
            end
            o.yp = ypred;
        end
        function o = setToday(o,tdin)   % td is a serial date or date string 'mm/dd/yyyy'
            if isa(tdin,'char')
                td = datenum(tdin,'mm/dd/yyyy');
            elseif 0 <= tdin && tdin <= 1
                td = o.T_imp.Date(round(o.L * tdin));
            else
                td = tdin;
            end
            
            if td < o.T_imp.Date(1)
                fprintf('\nIn Nedoc_Data.setToday:\nDate Entered is before start of dataset.\n')
                td = o.T_imp.Date(1);
            elseif td > o.T_imp.Date(end)
                fprintf('\nIn Nedoc_Data.setToday:\nDate Entered is after end of dataset.\n')
                td = o.T_imp.Date(end);
            end
            
            date_str = datestr(td,'mm/dd/yyyy');
            date = td;
            idx = find(o.T_imp.Date == date, 1, 'last');
            
            o.today = struct('dt', date, 'dts', date_str, 'i', idx);
            
            o.mu  = mean(o.y(1:o.today.i));
            o.sig = std(o.y(1:o.today.i));
        end
        function o = setPPD(o, NPPD)
            if NPPD < 1
                NPPD = 1;
            elseif NPPD > 288
                NPPD = 288;
            end
            
            o.PPD = NPPD;
            % 12 for 24, 6 for 48
            rate = round(288 / o.PPD);
            
            Date_Time_DTA = downsample( o.T_org.Date_Time_DTA, rate );
            Date_Time = downsample( o.T_org.Date_Time, rate );
            Date = downsample( o.T_org.Date, rate );
            Time = downsample( o.T_org.Time, rate );
            Month = downsample( o.T_org.Month, rate );
            WKD_Name = downsample( o.T_org.WKD_Name, rate );
            Weekday = downsample( o.T_org.Weekday, rate );
            Score = downsample( o.T_org.Score, rate );
            Level = downsample( o.T_org.Level, rate );
            
            o.T_imp = table(Date_Time_DTA, Date_Time, Date, Time, Month, WKD_Name, Weekday, Score, Level);
            
            o.X = [ (o.T_imp.Date) (o.T_imp.Time) (o.T_imp.Weekday) (o.T_imp.Month) ];
            o.X_t = o.T_imp.Date_Time;
            o.X_lag = [];
            
            o.y = o.T_imp.Score;
            o.yp = [];              % <------- clear yp
            
            o.L = height(o.T_imp);
            o = o.setToday(o.today.dts);
            
        end
        
        % visualizers
        function [plotfig,avg_acc] = plot(o, ttl, startIn, nplots, varargin)
            % arg processing
            if isa(startIn,'char')
                if strcmp(startIn,'tmr')    % natural place to start: day after training set ends
                    start = o.today.dt;
                else
                    start = datenum(startIn,'mm/dd/yyyy');
                    start = start - 1;
                end
            else
                start = startIn;
            end
            periods = {'days','weeks'};
            period = periods{1};
            for i = 1:2:length(varargin)
                if strcmp('TimeFrame',varargin{i}) && any(strcmp(periods,varargin{i+1}))
                    period = varargin{i+1};
                end
            end
            
            figttl = ttl;
            accarr = zeros([1,nplots]);
            idcurr = find(o.T_imp.Date == start, 1, 'last') + 1;
            plotfig = figure('NumberTitle','off','Name',[figttl,': Daily Plots']);
            for i = 1:nplots
                isToday = false;
                subplot(ceil(sqrt(nplots)),ceil(sqrt(nplots)),i)
                
                hold on
                
                dayIdcs = idcurr:idcurr+o.PPD-1;
                dayLen = o.PPD;
                acc = 100 * sum(getLevel(o.y(dayIdcs)) == getLevel(o.yp(dayIdcs))) / dayLen;
                plot(o.T_imp.Date_Time(dayIdcs), o.y(dayIdcs), 'b-')
                plot(o.T_imp.Date_Time(dayIdcs), o.yp(dayIdcs), 'r--')
                if sum(ismember(o.today.i,dayIdcs)) ~= 0
                    xline(o.today.i, 'k-', 'LineWidth', 4);
                    isToday = true;
                end
                plttl = [datestr(o.T_imp.WKD_Name(idcurr,:)) ', '...
                    datestr(o.T_imp.Date_Time_DTA(idcurr)) ': Acc = ' num2str(acc) '%'];
                    
                accarr(i) = acc;
                
                super_suit = ones([dayLen,1]);
                lspec = 'g:';
                plot(o.T_imp.Date_Time(dayIdcs), 20*super_suit,lspec);
                plot(o.T_imp.Date_Time(dayIdcs), 60*super_suit,lspec);
                plot(o.T_imp.Date_Time(dayIdcs), 100*super_suit,lspec);
                plot(o.T_imp.Date_Time(dayIdcs), 140*super_suit,lspec);
                plot(o.T_imp.Date_Time(dayIdcs), 200*super_suit,lspec);
                
                title(plttl)
                xlabel('time')
                ylabel('NEDOC Score')
                if i == 1 && isToday
                    legend('Actual','Predictor','Today','NEDOC Levels')
                elseif i == 1
                    legend('Actual','Predictor','NEDOC Levels')
                elseif isToday
                    legend('Today')
                end
                
                axis([o.T_imp.Date_Time(dayIdcs(1)), o.T_imp.Date_Time(dayIdcs(end)), 0, 200])
                datetick('x','HH:mm PM','keepticks')
                hold off
                
                idcurr = dayIdcs(dayLen) + 1;
            end
            avg_acc = mean(accarr);
        end
        
    end
    
end