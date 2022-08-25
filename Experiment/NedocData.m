classdef NedocData
    
    properties
        % tables
        T_org                               % original imputed table:   filled missing times with linspace vals
        T_imp                               % imputed table:            downsampled version of above table
        
        % input matrix struct
        X                                   % X (contains serial date and time, weekday, and month)
        X_t                                 % X_t (contains only serial date-time)
        
        % target & response vector structs
        y                                   % target
        yp                                  % response
        yp_mean                             % mean curve of responses (ensemble)
        
        % boundary struct
        today                               % todays date struct (marks training/testing split)
        
        % definitions
        L                                   % length
        L_day                               % numdays
        mu                                  % mean (TRAINING DATA)
        sig                                 % stdev (TRAINING DATA)
        PPD                                 % num of obs per day
    end
    
    methods
        
        % constructor
        function o = NedocData(TI,NPPD)
            o.T_org = TI;
            o.T_imp = o.T_org;
            
            o = o.setPPD(NPPD);
            
            o.X = [ (o.T_imp.Date) (o.T_imp.Time) (o.T_imp.Weekday) (o.T_imp.Month) ];
            o.X_t = o.T_imp.Date_Time;
            o.yp = cell(2,0);
            
            o.L = height(o.T_imp);
            o.L_day = length(unique(o.T_imp.Date));
        end
        
        % accessors
        function [X,y] = getmats(o,varargin)
            
            % defaults
            setspec = 'all';
            matspec = 'days';
            ndays = 1;
            lagspec = 1:14;
            doLag = false;
            useCellsForX = false;
            
            for i = 1:2:length(varargin)
                if strcmp(varargin{i},'setspec')
                    setspec = varargin{i+1};
                elseif strcmp(varargin{i},'matspec')
                    matspec = varargin{i+1};
                elseif strcmp(varargin{i},'ndays')
                    ndays = varargin{i+1};
                elseif strcmp(varargin{i},'cell')
                    useCellsForX = varargin{i+1};
                elseif strcmp(varargin{i},'lag')
                    doLag = varargin{i+1};
                elseif strcmp(varargin{i},'nlags')
                    lagspec = varargin{i+1};
                    if size(lagspec,1) == 1 && size(lagspec,2) == 1
                        lagspec = 1:lagspec;
                    end
                end
            end
            
            if strcmp(setspec,'train')
                idcs = 1:o.today.i;
            elseif strcmp(setspec,'test')
                idcs = (o.today.i + 1):o.L;
            elseif strcmp(setspec,'all')
                idcs = 1:o.L;
            else
                desc = 'setspec arg must be ''train'', ''test'', or ''all''';
                ME = MException('NedocData:getmats:Invalid_Arg',desc);
                throw(ME)
            end
            
            if doLag
                if strcmp(matspec,'full')
                    LM = lagmatrix(o.X,lagspec);
                    X = LM(idcs,:);
                    y = o.y(idcs);
                elseif strcmp(matspec,'time')
                    LM = lagmatrix(o.X_t,lagspec);
                    X = LM(idcs,:);
                    y = o.y(idcs);
                elseif strcmp(matspec,'days')
                    dvsr = o.PPD * ndays;
                    
                    X = zeros([ floor(length(idcs) / dvsr), dvsr ]);
                    for i = 1:dvsr:length(idcs) - (dvsr-1)
                        X((i+(dvsr-1))/dvsr,:) = o.y(i:i+(dvsr-1));
                    end
                    y = [];
                else
                    desc = 'matspec arg must be ''full'', ''time'', or ''days''';
                    ME = MException('NedocData:getmats:Invalid_Arg',desc);
                    throw(ME)
                end
            else
                if strcmp(matspec,'full')
                    X = o.X(idcs,:);
                    y = o.y(idcs);
                elseif strcmp(matspec,'time')
                    X = o.X_t(idcs,:);
                    y = o.y(idcs);
                elseif strcmp(matspec,'days')
                    dvsr = o.PPD * ndays;
                    
                    X = zeros([ floor(length(idcs) / dvsr), dvsr ]);
                    for i = 1:dvsr:length(idcs) - (dvsr-1)
                        X((i+(dvsr-1))/dvsr,:) = o.y(i:i+(dvsr-1));
                    end
                    y = [];
                else
                    desc = 'matspec arg must be ''full'', ''time'', or ''days''';
                    ME = MException('NedocData:getmats:Invalid_Arg',desc);
                    throw(ME)
                end
            end
            
            if useCellsForX
                X = mat2cellR(X);
            end
        end
        
        % mutators
        function o = pushResp(o,ypred,label)
            if length(ypred) ~= o.L
                desc = ['Length of arg (' num2str(length(ypred)) ') does not match imputed length (' num2str(o.L) ')'];
                ME = MException('NedocData:setResp:Invalid_Arg',desc);
                throw(ME)
            end
            if mean(ypred) <= 1 && mean(ypred) >= -1
                ypred = o.sig .* ypred + o.mu;
            end
            o.yp{1,end+1} = ypred;
            o.yp{2,end} = label;
            o = o.updateMean;
        end
        function o = popResp(o)
            if length(o.yp) < 1
                desc = 'yp cell array is aleady empty!';
                ME = MException('NedocData:popResp:Empty_Pop',desc);
                throw(ME)
            end
            o.yp(:,end) = [];
            o = o.updateMean;
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
            
            o.y = o.T_imp.Score;
            o.yp = [];              % <------- clear yp
            
            o.L = height(o.T_imp);
            if ~isempty(o.today)
                o = o.setToday(o.today.dts);
            end
        end
        
        % visualizers
        function plotfig = plot(o, ttl, startIn, nplots, varargin)
            % arg processing
            if isa(startIn,'char')
                if strcmp(startIn,'tmr')    % natural place to start: day after training set ends
                    start = o.today.dt;
                else
                    startIn_substr = startIn([1,2,3,4]);
                    if strcmp(startIn_substr,'tmr+')
                        addend_substr = startIn(5);
                        start = o.today.dt + str2num(addend_substr); %#ok<ST2NM>
                    elseif strcmp(startIn_substr,'tmr-')
                        diff_substr = startIn(5);
                        start = o.today.dt - str2num(diff_substr); %#ok<ST2NM>
                    else
                        start = datenum(startIn,'mm/dd/yyyy');
                        start = start - 1;
                    end
                end
            else
                start = startIn;
            end
            showmean= true;
            for i = 1:2:length(varargin)
                if strcmp('Showmean',varargin{i})
                    showmean = varargin{i+1};
                end
            end
            
            figttl = ttl;
            idcurr = find(o.T_imp.Date == start, 1, 'last') + 1;
            plotfig = figure('NumberTitle','off','Name',[figttl,': Daily Plots']);
            for i = 1:nplots
                yp_sum = zeros([o.PPD,1]);
                isToday = false;
                
                if o.T_imp.Date(idcurr) >= o.T_imp.Date(end)
                    fprintf('last day has been reached\n')
                    break;
                end
                
                subplot(ceil(sqrt(nplots)),ceil(sqrt(nplots)),i)
                
                hold on
                
                dayIdcs = idcurr:idcurr+o.PPD-1;
                dayLen = o.PPD;
                
                leg = {'Actual'};
                plot(o.T_imp.Date_Time(dayIdcs), o.y(dayIdcs), 'b-', 'LineWidth', 1.5)
                for ptor = 1:size(o.yp,2)
                    plot(o.T_imp.Date_Time(dayIdcs), o.yp{1,ptor}(dayIdcs), '--')
                    acc(ptor,1) = 100 * sum(getLevel(o.y(dayIdcs)) == getLevel(o.yp{1,ptor}(dayIdcs))) / dayLen;
                    leg{end+1} = o.yp{2,ptor};
                    yp_sum = yp_sum + o.yp{1,ptor}(dayIdcs);
                end
                
                if showmean
                    yp_mean = yp_sum / size(o.yp,2); %#ok<PROPLC>
                    plot(o.T_imp.Date_Time(dayIdcs), yp_mean, 'k-.', 'LineWidth', 1.5) %#ok<PROPLC>
                    leg{end+1} = 'Predictor Mean';
                end
                
                [~,wkdname] = weekday(o.T_imp.WKD_Name(idcurr,:));
                plttl = [ wkdname ', ' datestr(o.T_imp.Date_Time_DTA(idcurr)) ];
                
                if sum(ismember(o.today.i,dayIdcs)) ~= 0
                    xline(o.today.dt+1, 'k-', 'LineWidth', 4);
                    isToday = true;
                    leg{end+1} = 'Today';
                end
                
                super_suit = ones([dayLen,1]);
                lspec = 'g:';
                plot(o.T_imp.Date_Time(dayIdcs), 20*super_suit,lspec);
                plot(o.T_imp.Date_Time(dayIdcs), 60*super_suit,lspec);
                plot(o.T_imp.Date_Time(dayIdcs), 100*super_suit,lspec);
                plot(o.T_imp.Date_Time(dayIdcs), 140*super_suit,lspec);
                plot(o.T_imp.Date_Time(dayIdcs), 200*super_suit,lspec);
                leg{end+1} = 'NEDOC Levels';
                
                title(plttl)
                xlabel('time')
                ylabel('NEDOC Score')
                if i == 1 || isToday
                    legend(leg)
                end
                
                axis([o.T_imp.Date_Time(dayIdcs(1)), o.T_imp.Date_Time(dayIdcs(end)+1), 0, 200])
                datetick('x','HH:mm PM','keepticks')
                hold off
                
                idcurr = dayIdcs(dayLen) + 1;
            end
        end
        function accs = getaccs(o, setspec, tolerance)
            if strcmp(setspec,'train')
                idcs = 1:o.today.i;
            elseif strcmp(setspec,'test')
                idcs = (o.today.i+1):o.L;
            elseif strcmp(setspec,'all')
                idcs = 1:o.L;
            else
                desc = 'setspec argument must be ''train'', ''test'', or ''all''';
                ME = MException('NedocData:predictorAccs:Invalid_arg',desc);
                throw(ME);
            end
            
            if nargin == 2
                accs = cell(size(o.yp,2)+1,2);
                for i = 1:size(o.yp,2)
                    accs{i,1} = o.yp{2,i};
                    accs{i,2} = 100 * sum(getLevel(o.yp{1,i}(idcs)) == getLevel(o.y(idcs))) / length(idcs);
                end
                if ~isempty(o.yp_mean)
                    accs{i+1,1} = 'Predictor Mean';
                    accs{i+1,2} = 100 * sum(getLevel(o.yp_mean(idcs)) == getLevel(o.y(idcs))) / length(idcs);
                end
            else
                accs = cell(size(o.yp,2)+1,2);
                for i = 1:size(o.yp,2)
                    accs{i,1} = o.yp{2,i};
                    accs{i,2} = 100 * sum(abs(getLevel(o.yp{1,i}(idcs))-getLevel(o.y(idcs))) <= tolerance) / length(idcs);
                end
                if ~isempty(o.yp_mean)
                    accs{i+1,1} = 'Predictor Mean';
                    accs{i+1,2} = 100 * sum(abs(getLevel(o.yp_mean(idcs))-getLevel(o.y(idcs))) <= tolerance) / length(idcs);
                end
            end
        end
        
    end
    
    methods(Access = private)
        
        function o = updateMean(o)
            if size(o.yp,2)==0
                o.yp_mean = [];
                return
            end
            cumsum = zeros(size(o.yp{1,1}));
            for i = 1:size(o.yp,2)
                cumsum = cumsum + o.yp{1,i};
            end
            o.yp_mean = cumsum / size(o.yp,2);
        end
        
    end
    
end