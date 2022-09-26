classdef NedocInterface < network_interface
    properties
        % raw data
        table
        date
        
        % params
        PPD
        nObs
        nDays
        trvl_date
        vlts_date
        lag
        
        % raw matrix forms
        Ym
        Ymp
        
        % transformations
        transforms
        centers
    end
    methods
        function obj = NedocInterface(table,NPPD)
            if nargin == 1
                NPPD = 48;
            end
            obj@network_interface();
            obj.setPPD(table,NPPD);
            
            obj.nObs = height(obj.table);
            obj.nDays = round(obj.nObs/obj.PPD);
            obj.date = DataStruct(unique(obj.table.Date_Time_DTA,'stable'));
            
            scorevect = obj.table.Score;
            dm = reshape(scorevect,[obj.PPD,obj.nDays])';
            
            obj.Ym = DataStruct(dm);
            obj.Ymp = DataStruct();
        end
        function obj = setsplits(obj,trvl,vlts)
            if isa(trvl,'datetime')
                obj.trvl_date = trvl;
                iTV = find(obj.date==trvl);
            elseif isa(trvl,'double')
                if 0 <= trvl && trvl <= 1
                    iTV = round(obj.nDays*trvl);
                    obj.trvl_date = obj.date(iTV);
                else
                    iTV = trvl;
                    obj.trvl_date = obj.date(iTV);
                end
            else
                error("Invalid datatype for trvl argument")
            end
            if isa(vlts,'datetime')
                obj.vlts_date = vlts;
                iVT = find(obj.date==vlts);
            elseif isa(vlts,'double')
                if 0 <= vlts && vlts <= 1
                    iVT = round(obj.nDays*vlts);
                    obj.vlts_date = obj.date(iVT);
                else
                    iVT = vlts;
                    obj.vlts_date = obj.date(vlts);
                end
            else
                error("Invalid datatype for vlts argument")
            end
            
            obj.Ym.setsplits(iTV,iVT);
            obj.date.setsplit(iTV,iVT);
        end
    end
    methods(Access=protected)
        function obj = setPPD(obj,tbl,NPPD)
            baseppd = 288;
            
            if NPPD < 1
                NPPD = 1;
            elseif NPPD > baseppd
                NPPD = baseppd;
            end
            
            rate = round(baseppd / NPPD);
            
            Date_Time_DTA = downsample( tbl.Date_Time_DTA, rate );
            Date_Time = downsample( tbl.Date_Time, rate );
            Date = downsample( tbl.Date, rate );
            Time = downsample( tbl.Time, rate );
            Month = downsample( tbl.Month, rate );
            WKD_Name = downsample( tbl.WKD_Name, rate );
            Weekday = downsample( tbl.Weekday, rate );
            Score = downsample( tbl.Score, rate );
            Level = downsample( tbl.Level, rate );
            
            obj.table = table(Date_Time_DTA, Date_Time, Date, Time,...
                Month, WKD_Name, Weekday, Score, Level); %#ok<CPROPLC>
            obj.PPD = NPPD;
        end
        function obj = preprocess(obj,varargin)
            % default args
            PCA_pcnt = [0.90 0.90];
            Lag = 1:14;
            % varargin proc
            for arg_idx = 1:2:length(varargin)
                if strcmp('PCA',varargin{arg_idx})
                    PCA_pcnt = varargin{arg_idx+1};
                    if length(PCA_pcnt) == 1
                        PCA_pcnt = [PCA_pcnt PCA_pcnt]; %#ok<AGROW>
                    end
                elseif strcmp('Lags',varargin{arg_idx})
                    Lag = varargin{arg_idx+1};
                    if length(Lag) == 1
                        Lag = 1:Lag;
                    end
                end
            end
            obj.lag = Lag;
            
            
        end
        function obj = postprocess(obj,varargin)
        end
    end
end