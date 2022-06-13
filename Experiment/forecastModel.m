

classdef forecastModel
    
    properties
        
        % ========== TITLE ==========
        
        % title
        ttl;
        
        % ========== DATA ==========
        
        % numeric features
        dtNum                           % numeric date/time value (decimal part corresponds to time)
        date                            % numeric date value
        time                            % numeric time value
        month                           % month
        wkday                           % wkday
        
        % non-numeric features
        dtArr                           % dateTimeArray (as datetime object)
        wkdayNames                      % weekday names
        
        % target responses
        score                           % NEDOC Score (0-200)
        level                           % NEDOC Level (1-5)
        
        % symbolic values
        X                               % collected numeric features
        y                               % response value (score, we're not using classifiers since response is ordinal)
        X_train                         % training split of X
        X_test                          % testing split of X
        y_train                         % training split of y
        y_test                          % testing split of y
        
        % predicted responses
        score_pred                      % NEDOC Score (0-200)
        level_pred                      % NEDOC Level (1-5)
        
        % other sets
        avg_score_daily                 % set of averages for each day
        avg_level_daily                 % avg level for day
        
        % ========== MODELING ============
        
        % forecast models
        mdl                             % model object
        trainFcn                        % training function
        predFcn                         % prediction function
        
        % split
        today                           % index of today (marks end of training set)
        tomorrow                        % index of tomorrow (marks beginning of testing set)
        
    end
    
    methods
        
        function this = forecastModel(TC,ttl)   % ctor
            this.dtNum = TC.Date_Time;
            this.date = TC.Date;
            this.time = TC.Time;
            this.month = TC.Month;
            this.wkday = TC.Weekday;
            
            this.dtArr = TC.Date_Time_DTA;
            this.wkdayNames = TC.WKD_Name;
            
            this.score = TC.Score;
            this.level = TC.Level;
            
            this.ttl = ttl;
            
            this.avg_score_daily = this.fillDailyAvgs();
            this.avg_level_daily = getLevel(this.avg_score_daily);
        end
        
        function this = rawModelInput(this,mdl)
            this.mdl = mdl;
        end
        
        function this = selectModelFunctions(this,trainingFcn,predictingFcn)
            if isa(trainingFcn,'function_handle')
                this.trainFcn = trainingFcn;
                this.predFcn = predictingFcn;
            else
                fprintf('inputs should be functions denoted with ''@''')
            end
        end
        
        function this = setSplit(this,td,bias)      % bias is a bool indicating whether to add bias feature
            [m,~] = size(this.date);
            if isa(td,'double')
                [idcs,s] = this.getDay(td);
                
                this.today = min(idcs)+s-1;
                this.tomorrow = this.today+1;
            elseif isa(td,'datetime')
                tdl = datenum(td);
                [idcs,s] = this.getDay(tdl);
                
                this.today = min(idcs)+s-1;
                this.tomorrow = this.today+1;
            end
            
            this.X = [this.date this.time this.wkday this.month];
            this.y = this.score;
            if bias
                this.X = [ones(size(this.date)) this.X];
            end
            this.X_train = this.X(1:this.today,:);
            this.y_train = this.y(1:this.today);
            this.X_test = this.X(this.tomorrow:m,:);
            this.y_test = this.y(this.tomorrow:m);
            
        end
        
        function [idxList,dSz] = getDay(this,idx)
            [m,~] = size(this.wkday);
            
            lowerBound = idx-60;
            upperBound = idx+60;
            
            if lowerBound < 1
                lowerBound = 1;
            end
            if upperBound > m
                upperBound = m;
            end
            
            wkd = this.wkday(idx);
            daySearchInterval = lowerBound:upperBound;
            
            idxList = find(this.wkday(daySearchInterval)==wkd);
            idxList = sort(idxList) + lowerBound - 1;
            [dSz,~] = size(idxList);
        end
        
        function this = train(this)
            if isempty([this.today this.tomorrow])
                fprintf('must select split before training')
            else
                clear this.mdl
                this.mdl = this.trainFcn(this.X_train, this.y_train);
            end
        end
        
        function this = pred(this)
            predictFcn = this.predFcn; %#ok<NASGU>
            this.score_pred = this.mdl.predictFcn(this.X);
            this.level_pred = getLevel(this.score_pred);
        end
        
        function avgScore = fillDailyAvgs(this)
            [m,~] = size(this.date);
            
            avgScore = [];
            idcurr = 1;
            while idcurr < m
                [dayIdcs,dayLen] = this.getDay(idcurr);
                avgScore = [avgScore ; mean(this.score(dayIdcs))];
                idcurr = idcurr + dayLen;
            end
        end
        
        function plotfig = generateRegPlots(this,startday,numdays,varargin)
            
            figChoice = 0;
            for i = 1:2:length(varargin)
                if strcmp('PlotType',varargin{i})
                    switch varargin{i+1}
                        case 'daily'
                            figChoice = 0;
                        case 'weekly'
                            figChoice = 1;
%                         case ''
                    end
                end
            end
            
            figttl = this.ttl;
            
            if isa(startday,'datetime')
                startday = datenum(startday);
            end
            
            if startday < this.tomorrow
                fprintf('WARNING: some value predicted may be part of training set')
                figttl = ['(WARNING! Training Data Present!) ',figttl];
            end
            
            if figChoice==0
                
                idcurr = startday;
                
                plotfig = figure('NumberTitle','off','Name',[figttl,' Daily Plots']);
                
                for i = 1:numdays
                    
                    [dayIdcs,dayLen] = this.getDay(idcurr);
                    acc = 100 * sum(this.level(dayIdcs) == this.level_pred(dayIdcs)) / dayLen;
                    
                    subplot(ceil(sqrt(numdays)),ceil(sqrt(numdays)),i)
                    hold on
                    
                    plot(this.score(dayIdcs), 'b-')
                    
                    plot(this.score_pred(dayIdcs), 'r--')
                    
                    super_suit = ones(size(this.score(dayIdcs)));
                    lspec = 'g:';
                    plot(20*super_suit,lspec);
                    plot(60*super_suit,lspec);
                    plot(100*super_suit,lspec);
                    plot(140*super_suit,lspec);
                    plot(200*super_suit,lspec);
                    
                    td=0;
                    if(sum(find(this.today==dayIdcs)) ~= 0)
                        xline(dayLen-1,'k-','LineWidth',10)
                        td = 1;
                    end
                    
                    plttl = ...
                        [this.wkdayNames(idcurr,:) ', ' datestr(this.dtArr(idcurr)) ': Acc = '...
                        num2str(acc) '%' ', dayLen = ' num2str(dayLen)];
                    title(plttl)
                    xlabel('observations (~1 per 30 minutes)')
                    ylabel('NEDOC Score')
                    if td
                        legend('Actual','Predictor','NEDOC Levels','Today')
                    else
                        legend('Actual','Predictor','NEDOC Levels')
                    end
                    axis([1, dayLen, 0, 200])
                    
                    hold off
                    
                    idcurr = dayIdcs(dayLen) + 1;
                    
                end
                
            end
            
        end
        
    end
    
end











