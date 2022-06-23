classdef Nedoc_Data
    
    properties
        % tables
        T_cln
        T_imp
        T_day
        
        % input matrices
        X_cln
        X_imp
        X_day
        
        % target & response vectors
        y_cln
        y_imp
        y_day
        yp_cln
        yp_imp
        yp_day
        
        % boundaries
        today
        tomorrow
        
        % definitions
        cls_defs
        
    end
    
    methods
        
        % constructor
        function o = Nedoc_Data(TC,TI,TD,cldef)
            o.T_cln = TC;
            o.T_imp = TI;
            o.T_day = TD;
            
            o.X_cln = [ (o.T_cln.Date) (o.T_cln.Time) (o.T_cln.Weekday) (o.T_cln.Month) ];
            o.X_imp = [ (o.T_imp.Date) (o.T_imp.Time) (o.T_imp.Weekday) (o.T_imp.Month) ];
            o.X_day = [ (o.T_day.Date) (o.T_day.Weekday) (o.T_day.Month) ];
            
            o.y_cln = (o.T_cln.Score);
            o.y_imp = (o.T_imp.Score);
            o.y_day = (o.T_day.Day_Class);
            
            o.cls_defs = cldef;
        end
        
        % accessors
        function [idxList,dSz] = getDay(o,idx)
            m = length(o.y_cln)
            
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
            
            wkd = o.X_cln(idx,3);
            daySearchInterval = lowerBound:upperBound;
            
            idxList = find(o.X_cln(daySearchInterval,3)==wkd);
            idxList = sort(idxList) + lowerBound - 1;
            [dSz,~] = size(idxList);
        end
        
        % mutators
        function o = setResp(o,typestr,y)
            if strcmp(typestr,'cln')
                o.yp_cln = y;
                o.yp_imp = o.impute(y);
                % todo
            elseif strcmp(typestr,'imp')
                o.yp_cln = o.desaturate(y);
                o.yp_imp = y;
                % todo
            elseif strcmp(typestr,'day')
                o.yp_cln = o.desaturate(o.cls2imp(y));
                o.yp_imp = o.cls2imp(y);
                o.yp_day = y;
            else
                fprintf('Invalid typestr. use ''cln'', ''imp'', or ''day''.')
            end
        end
        
        % converters
        function y48pt_NaN = saturate(o,dayInterval,listIn)
            % function returns days as 48 point column vectors, filling in missing score values with NaN
            M = 48;     % num of obs per day
            dayLen = length(dayInterval);
            
            % instantiate arrays
            timeArrIn = zeros([M,1]);
            timeArrIn(1:dayLen) = o.X_cln(dayInterval,2);
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
            y48pt_NaN = scoreArrOut;
        end
        function listOut = desaturate(o,listIn)
            XI = o.X_imp;
            XO = o.X_cln;
            mO = length(o.y_cln);
            mI = length(listIn);
            
            for i = 1:mO
                while XI(i,2) <= XO(i,2) - 0.001 || XI(i,2) >= XO(i,2) + 0.001
                    XI(i:mI,:) = circshift(XI(i:mI,:),-1,1);
                    listIn(i:mI) = circshift(listIn(i:mI),-1,1);
                end
            end
            listOut = listIn(1:mO);
        end
        function listOut = day2serial(o,classifiedList)
            L = length(o.y_day);
            listOut = [];
            for i = 1:L
                listOut = [listOut o.cls_defs(classifiedList(i,:),:)];
            end
            listOut = o.desaturate(listOut');
        end
        function listOut = impute(o,listIn)
            y_NaN = [];
            i = 1;
            while i < length(o.y_cln)
                [dayIdcs,dayLen] = o.getDay(i);
                y_NaN_ = o.saturate(dayIdcs,listIn);
                y_NaN = [y_NaN ; y_NaN_]; %#ok<AGROW>
                i = i + dayLen;
            end
            listOut = fillHoles(y_NaN);
        end
        function listOut = cls2imp(this,listIn)
            L = length(this.dayClass);
            listOut = [];
            for i = 1:L
                listOut = [listOut this.dayClass_DEF(listIn(i,:),:)]; %#ok<AGROW>
            end
            listOut = listOut';
        end
        
        % visualizers
        function [plotfig,avg_acc] = generateRegPlots(o,ttl,startday,nplots,varargin)
            
            figChoice = 'day';
            for i = 1:2:length(varargin)
                if strcmp('PlotType',varargin{i})
                    switch varargin{i+1}
                        case 'daily'
                            figChoice = 'day';
                        case 'weekly'
                            figChoice = 'week';
                            %                         case ''
                    end
                end
            end
            
            persistent fignum;
            if isempty(fignum)
                fignum = 0;
            end
            fignum = fignum + 1;
            
            figttl = [num2str(fignum) ': ' ttl];
            accarr = zeros([1,nplots]);
            
            if isa(startday,'datetime')
                startday = datenum(startday);
            end
            
%             if startday < o.tomorrow
%                 fprintf('WARNING: some value predicted may be part of training set')
%                 figttl = ['(WARNING! Training Data Present!) ',figttl];
%             end
            
            if strcmp(figChoice,'day')
                
                idcurr = startday;
                
                plotfig = figure('NumberTitle','off','Name',[figttl,' Daily Plots']);
                
                for i = 1:nplots
                    
                    [dayIdcs,dayLen] = o.getDay(idcurr);
                    acc = 100 * sum(getLevel(o.y_cln(dayIdcs)) == getLevel(o.yp_cln(dayIdcs))) / dayLen;
                    accarr(i) = acc;
                    
                    subplot(ceil(sqrt(nplots)),ceil(sqrt(nplots)),i)
                    hold on
                    
                    plot(o.y_cln(dayIdcs), 'b-')
                    
                    plot(o.yp_cln(dayIdcs), 'r--')
                    
                    super_suit = ones(size(o.y_cln(dayIdcs)));
                    lspec = 'g:';
                    plot(20*super_suit,lspec);
                    plot(60*super_suit,lspec);
                    plot(100*super_suit,lspec);
                    plot(140*super_suit,lspec);
                    plot(200*super_suit,lspec);
                    
%                     td=0;
%                     if(sum(find(o.today==dayIdcs)) ~= 0)
%                         xline(dayLen-1,'k-','LineWidth',10)
%                         td = 1;
%                     end
                    
                    plttl = ...
                        [o.T_cln.WKD_Name(idcurr,:) ', ' datestr(o.T_cln.Date_Time_DTA(idcurr)) ': Acc = '...
                        num2str(acc) '%' ', dayLen = ' num2str(dayLen)];
                    title(plttl)
                    xlabel('observations (~1 per 30 minutes)')
                    ylabel('NEDOC Score')
%                     if td
%                         legend('Actual','Predictor','NEDOC Levels','Today')
%                     elseif i == 1
                        legend('Actual','Predictor','NEDOC Levels')
%                     end
                    axis([1, dayLen, 0, 200])
                    
                    hold off
                    
                    idcurr = dayIdcs(dayLen) + 1;
                    
                end
                
            elseif strcmp(figChoice,'week')
                
%                 [list,~] = o.getWeek(startday);
%                 idcurr = list(1);
%                 
%                 plotfig = figure('NumberTitle','off','Name',[figttl,' Weekly Plots']);
%                 
%                 for i = 1:nplots
%                     
%                     [weekIdcs,weekLen] = o.getWeek(idcurr);
%                     acc = 100 * sum(o.level(weekIdcs) == o.level_pred(weekIdcs)) / weekLen;
%                     accarr(i) = acc;
%                     
%                     subplot(ceil(sqrt(nplots)),ceil(sqrt(nplots)),i)
%                     hold on
%                     
%                     plot(o.score(weekIdcs), 'b-')
%                     
%                     plot(o.score_pred(weekIdcs), 'r--')
%                     
%                     super_suit = ones(size(o.score(weekIdcs)));
%                     lspec = 'g:';
%                     plot(20*super_suit,lspec);
%                     plot(60*super_suit,lspec);
%                     plot(100*super_suit,lspec);
%                     plot(140*super_suit,lspec);
%                     plot(200*super_suit,lspec);
%                     
%                     td=0;
%                     if(sum(find(o.today==weekIdcs)) ~= 0)
%                         xline(weekLen-1,'k-','LineWidth',10)
%                         td = 1;
%                     end
%                     
%                     plttl = ...
%                         [o.wkdayNames(idcurr,:) ', ' datestr(o.dtArr(idcurr)) ': Acc = '...
%                         num2str(acc) '%' ', weekLen = ' num2str(weekLen)];
%                     title(plttl)
%                     xlabel('observations (~1 per 30 minutes)')
%                     ylabel('NEDOC Score')
%                     if td
%                         legend('Actual','Predictor','NEDOC Levels','Today')
%                     else
%                         legend('Actual','Predictor','NEDOC Levels')
%                     end
%                     axis([1, weekLen, 0, 200])
%                     
%                     hold off
%                     
%                     idcurr = weekIdcs(weekLen) + 1;
%                     
%                 end
                
            end
            avg_acc = mean(accarr);
        end
        
    end
    
end