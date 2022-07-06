classdef Nedoc_Data
    
    properties
        % tables
        T_cln                               % clean table:      NaNs/repeats removed, faithful to original set
        T_imp                               % imputed table:    missing observations are filled with linearly spaced scores
        T_day                               % day table:        assigns a kmeans/HAC class to each day
        
        % input matrix struct
        X                                   % X struct (contains serial date and time, weekday, and month)
        X_t                                 % X_t struct (contains only serial date-time or serial date)
        
        % target & response vector structs
        y                                   % target struct
        yp                                  % response struct
        
        % boundary struct
        today                               % todays date struct (marks training/testing split)
        
        % definitions
        cls_defs                            % class definition matrix (from kmeans/HAC)
        L                                   % lengths struct
        std__                               % bool indicating standardization
        mu                                  % mean (TRAINING DATA)
        sig                                 % stdev (TRAINING DATA)
        
        % exceptions
        splitNotSet = MException('Nedoc_Data:splitNotSet','Must call Nedoc_Data.setToday first');
        
    end
    
    methods
        
        % constructor
        function o = Nedoc_Data(TC,TI,TD,cldef)
            o.T_cln = TC;
            o.T_imp = TI;
            o.T_day = TD;
            
            X_cln = [ (o.T_cln.Date) (o.T_cln.Time) (o.T_cln.Weekday) (o.T_cln.Month) ];
            X_imp = [ (o.T_imp.Date) (o.T_imp.Time) (o.T_imp.Weekday) (o.T_imp.Month) ];
            X_day = [ (o.T_day.Date) (o.T_day.Weekday) (o.T_day.Month) ];
            o.X = struct('c',X_cln, 'i',X_imp, 'd',X_day);
            
            o.X_t = struct('c',(o.T_cln.Date)+(o.T_cln.Time),...
                           'i',(o.T_imp.Date)+(o.T_imp.Time),...
                           'd',(o.T_day.Date));
            
            y_cln = (o.T_cln.Score);
            y_imp = (o.T_imp.Score);
            y_day = (o.T_day.Day_Class);
            o.y = struct('c',y_cln, 'i',y_imp, 'd',y_day);
            
            o.L = struct('c',size(o.y.c,1), 'i',size(o.y.i,1), 'd',size(o.y.d,1));
            
            fields = {'c','i','d'} ;
            c = cell(length(fields),1);
            o.yp = cell2struct(c,fields);
            
            o.cls_defs = cldef;
            
            o.std__ = false;
        end
        
        % accessors
        function [idxList,dSz] = getDay(o,idx)
            m = o.L.c;
            
            if(idx > m)
                idx = m;
            end
            
            lowerBound = idx-60;
            upperBound = idx+60;
            
            if lowerBound < 1
                lowerBound = 1;
            end
            if upperBound > m
                upperBound = m;
            end
            
            wkd = o.X.c(idx,3);
            daySearchInterval = lowerBound:upperBound;
            
            idxList = find(o.X.c(daySearchInterval,3)==wkd);
            idxList = sort(idxList) + lowerBound - 1;
            [dSz,~] = size(idxList);
        end
        function [X,y] = getmats(o,setspec,field,fullORtime)
            if strcmp(setspec,'train')
                idcs_c = 1:o.today.c;
                idcs_i = 1:o.today.i;
                idcs_d = 1:o.today.d;
            elseif strcmp(setspec,'test')
                idcs_c = (o.today.c + 1):o.L.c;
                idcs_i = (o.today.i + 1):o.L.i;
                idcs_d = (o.today.d + 1):o.L.d;
            elseif strcmp(setspec,'all')
                idcs_c = 1:o.L.c;
                idcs_i = 1:o.L.i;
                idcs_d = 1:o.L.d;
            else
                fprintf('\nIn Nedoc_Data.getmats:\nInvalid set specifier. Use ''train'', ''test'', or ''all''.\n')
            end
            
            if strcmp(fullORtime,'full')
                if strcmp(field,'cln')
                    X = o.X.c(idcs_c,:);
                    y = o.y.c(idcs_c);
                elseif strcmp(field,'imp')
                    X = o.X.i(idcs_i,:);
                    y = o.y.i(idcs_i);
                elseif strcmp(field,'day')
                    X = o.X.d(idcs_d,:);
                    y = o.y.d(idcs_d);
                else
                    fprintf('\nIn Nedoc_Data.getmats:\nInvalid Field. Use ''cln'', ''imp'', or ''day''.\n')
                end
            elseif strcmp(fullORtime,'time')
                if strcmp(field,'cln')
                    X = o.X_t.c(idcs_c,:);
                    y = o.y.c(idcs_c);
                elseif strcmp(field,'imp')
                    X = o.X_t.i(idcs_i,:);
                    y = o.y.i(idcs_i);
                elseif strcmp(field,'day')
                    X = o.X_t.d(idcs_d,:);
                    y = o.y.d(idcs_d);
                else
                    fprintf('\nIn Nedoc_Data.getmats:\nInvalid Field. Use ''cln'', ''imp'', or ''day''.\n')
                end
            else
                fprintf('\nIn Nedoc_Data.getmats:\nInvalid full/time specifier. Use ''full'' or ''time''.\n')
            end
        end
        
        % mutators
        function o = setResp(o,y)
            if length(y) == o.L.c
                o.yp.c = y;
                o.yp.i = o.impute(y);
                o.yp.d = o.imp2cls(o.yp.i);
            elseif length(y) == o.L.i
                o.yp.c = o.desaturate(y);
                o.yp.i = y;
                o.yp.d = o.imp2cls(y);
            elseif length(y) == o.L.d
                o.yp.c = o.desaturate(o.cls2imp(y));
                o.yp.i = o.cls2imp(y);
                o.yp.d = y;
            else
                fprintf('Invalid typestr. use ''cln'', ''imp'', or ''day''.')
            end
        end
        function o = setToday(o,tdin)   % td is a serial date or date string 'mm/dd/yyyy'
            if isa(tdin,'char')
                td = datenum(tdin,'mm/dd/yyyy');
            elseif 0 <= tdin && tdin <= 1
                td = o.T_imp.Date(round(o.L.i * tdin));
            else
                td = tdin;
            end
            
            if td < o.T_day.Date(1)
                fprintf('\nIn Nedoc_Data.setToday:\nDate Entered is before start of dataset.\n')
                td = o.T_day.Date(1);
            elseif td > o.T_day.Date(o.L.d)
                fprintf('\nIn Nedoc_Data.setToday:\nDate Entered is after end of dataset.\n')
                td = o.T_day.Date(o.L.d);
            end
            
            date_str = datestr(td,'mm/dd/yyyy');
            date = td;
            i_cln = find(o.T_cln.Date == date, 1, 'last');
            i_imp = find(o.T_imp.Date == date, 1, 'last');
            i_day = find(o.T_day.Date == date);
            
            o.today = struct('dt', date, 'dts', date_str, 'c', i_cln, 'i', i_imp, 'd', i_day);
            
            o.mu  = struct('c',mean(o.y.c(1:o.today.c)), 'i',mean(o.y.i(1:o.today.i)));
            o.sig = struct('c',std(o.y.c(1:o.today.c)), 'i',std(o.y.i(1:o.today.i)));
        end
        function o = toggleStandardization(o)
            % NOTE: mu and sig are calculated from training set
            if isempty(o.mu) || isempty(o.sig)
                throw(o.splitNotSet);       % throw splitNotSet exception if mu or sig are empty
            end
            
            if ~o.std__
                o.y.c = ( o.y.c - o.mu.c ) / o.sig.c;
                o.y.i = ( o.y.i - o.mu.i ) / o.sig.i;
                o.cls_defs = ( o.cls_defs - o.mu.i ) / o.sig.i;
            elseif o.std__
                o.y.c = o.sig.c .* o.y.c + o.mu.c;
                o.y.i = o.sig.i .* o.y.i + o.mu.i;
                o.cls_defs = o.sig.i .* o.cls_defs + o.mu.i;
            end
            o.std__ = ~o.std__;
        end
        
        % converters
        function dayOut = saturate(o,dayInterval,listIn)
            % function returns days as 48 point column vectors, filling in missing score values with NaN
            M = 48;     % num of obs per day
            dayLen = length(dayInterval);
            
            % instantiate arrays
            timeArrIn = zeros([M,1]);
            timeArrIn(1:dayLen) = o.X.c(dayInterval,2);
            timeArrCorrected = (0:1/M:.99)';
            
            scoreArrOut = zeros([M,1]) + NaN;
            scoreArrOut(1:dayLen) = listIn(dayInterval);
            
            % fit data to 24 hour (48 points) day, leaving NaNs for missing values
            for i = 1:M
                if timeArrCorrected(i) >= timeArrIn(i)+.0001 || timeArrCorrected(i) <= timeArrIn(i)-.0001
                    timeArrIn(i:M) = circshift(timeArrIn(i:M),1);
                    scoreArrOut(i:M) = circshift(scoreArrOut(i:M),1);
                end
            end
            
            % response
            dayOut = scoreArrOut;
        end
        function listOut = desaturate(o,listIn)
            XI = o.X.i;
            XO = o.X.c;
            mO = o.L.c;
            mI = o.L.i;
            
            for i = 1:mO
                while XI(i,2) <= XO(i,2) - 0.001 || XI(i,2) >= XO(i,2) + 0.001
                    XI(i:mI,:) = circshift(XI(i:mI,:),-1,1);
                    listIn(i:mI) = circshift(listIn(i:mI),-1,1);
                end
            end
            listOut = listIn(1:mO);
        end
        function listOut = day2serial(o,classifiedList)
            listOut = [];
            for i = 1:o.L.d
                listOut = [listOut o.cls_defs(classifiedList(i,:),:)];
            end
            listOut = o.desaturate(listOut');
        end
        function listOut = impute(o,listIn)
            y_NaN = [];
            i = 1;
            while i < o.L.c
                [dayIdcs,dayLen] = o.getDay(i);
                y_NaN_ = o.saturate(dayIdcs,listIn);
                y_NaN = [y_NaN ; y_NaN_]; %#ok<AGROW>
                i = i + dayLen;
            end
            listOut = fillHoles(y_NaN);
        end
        function listOut = cls2imp(o,listIn)
            listOut = [];
            for i = 1:o.L.d
                listOut = [listOut o.cls_defs(listIn(i,:),:)]; %#ok<AGROW>
            end
            listOut = listOut';
        end
        function listOut = imp2cls(o,listIn)
            [~,n] = size(listIn);
            listOut = zeros([o.L.d,n]);
            for i = 1:48:o.L.i
                this_day = listIn(i:(i+47),:);
                dists = pdist2(this_day',o.cls_defs,'squaredeuclidean');
                [~,listOut(1+(i-1)/48,:)] = min(dists);
            end
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
            flds = {'cln','imp','day'};
            period = periods{1};
            fld = flds{1};
            for i = 1:2:length(varargin)
                if strcmp('TimeFrame',varargin{i}) && any(strcmp(periods,varargin{i+1}))
                    period = varargin{i+1};
                elseif strcmp('Field',varargin{i}) && any(strcmp(periods,varargin{i+1}))
                    fld = varargin{i+1};
                end
            end
            
            figttl = ttl;
            accarr = zeros([1,nplots]);
            if strcmp(fld,'cln')
                idcurr = find(o.T_cln.Date == start, 1, 'last') + 1;
            elseif strcmp(fld,'imp')
                idcurr = find(o.T_imp.Date == start, 1, 'last') + 1;
            elseif strcmp(fld,'day')
                idcurr = find(o.T_day.Date == start, 1, 'last') + 1;
            end
            
            plotfig = figure('NumberTitle','off','Name',[figttl,': Daily Plots']);
            for i = 1:nplots
                isToday = false;
                subplot(ceil(sqrt(nplots)),ceil(sqrt(nplots)),i)
                
                hold on
                if strcmp(fld,'cln')
                    [dayIdcs,dayLen] = o.getDay(idcurr);
                    acc = 100 * sum(getLevel(o.y.c(dayIdcs)) == getLevel(o.yp.c(dayIdcs))) / dayLen;
                    plot(o.y.c(dayIdcs), 'b-')
                    plot(o.yp.c(dayIdcs), 'r--')
                    if sum(ismember(o.today.c,dayIdcs)) ~= 0
                        xline(o.today.c - dayIdcs(1), 'k-', 'LineWidth', 4);
                        isToday = true;
                    end
                    plttl = [o.T_cln.WKD_Name(idcurr,:) ', '...
                        datestr(o.T_cln.Date_Time_DTA(idcurr)) ': Acc = ' num2str(acc) '%'];
                elseif strcmp(fld,'imp')
                    dayIdcs = idcurr:idcurr+48;
                    dayLen = 48;
                    acc = 100 * sum(getLevel(o.y.i(dayIdcs)) == getLevel(o.yp.i(dayIdcs))) / dayLen;
                    plot(o.y.i(dayIdcs), 'b-')
                    plot(o.yp.i(dayIdcs), 'r--')
                    if sum(ismember(o.today.i,dayIdcs)) ~= 0
                        xline(o.today.i, 'k-', 'LineWidth', 4);
                        isToday = true;
                    end
                    plttl = [o.T_imp.WKD_Name(idcurr,:) ', '...
                        datestr(o.T_imp.Date_Time_DTA(idcurr)) ': Acc = ' num2str(acc) '%'];
                elseif strcmp(fld,'day')
                    dayIdcs = idcurr;
                    dayLen = 1;
                    acc = 100 * sum(getLevel(o.y_cln(dayIdcs)) == getLevel(o.yp_cln(dayIdcs))) / dayLen;
                    plot(o.cls_defs(dayIdcs), 'b-')
                    plot(o.yp.c(dayIdcs), 'r--')
                    plttl = [o.T_day.WKD_Name(idcurr,:) ', '...
                        datestr(o.T_day.Date_Time_DTA(idcurr)) ': Acc = ' num2str(acc) '%'];
                end
                accarr(i) = acc;
                
                super_suit = ones([dayLen,1]);
                lspec = 'g:';
                plot(20*super_suit,lspec);
                plot(60*super_suit,lspec);
                plot(100*super_suit,lspec);
                plot(140*super_suit,lspec);
                plot(200*super_suit,lspec);
                
                title(plttl)
                xlabel('observations')
                ylabel('NEDOC Score')
                if i == 1 && isToday
                    legend('Actual','Predictor','Today','NEDOC Levels')
                elseif i == 1
                    legend('Actual','Predictor','NEDOC Levels')
                elseif isToday
                    legend('Today')
                end
                axis([1, dayLen, 0, 200])
                hold off
                
                idcurr = dayIdcs(dayLen) + 1;
            end
            avg_acc = mean(accarr);
        end
        
    end
    
end