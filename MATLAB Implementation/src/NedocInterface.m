classdef NedocInterface < network_interface
    properties(Access=public)
        % raw data
        table
        date
        
        % params
        PPD
        nObs
        nDays
        trvl_date
        vlts_date
        lag_vector
        PCA_pcnts
        
        % raw matrix forms
        Ym
        Ymp
        
        % transformations
        transforms
        centers
    end
    methods(Access=public)
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
            
            obj.centers = {};
            obj.transforms = {};
            obj.lag_vector = 1:14;
            obj.PCA_pcnts = [0.9,0.9];
        end
        function obj = compile(obj,arch,opts,varargin)
            if nargin < 3
                desc = "Must specify network architecture and training options";
                id = "NedocInterface:NotEnoughInputs";
                throw(MException(id,desc))
            end
            
            trvl = 0.9;
            vlts = 0.9;
            
            for ii=1:2:numel(varargin), key=varargin{ii}; val=varargin{ii+1};
                switch(key)
                    case "Verbose"
                        opts.Verbose = val;
                    case "Plot"
                        if val
                            opts.Plots = "training-progress";
                        else
                            opts.Plots = "none";
                        end
                    case "TrainingSplit"
                        trvl = val;
                    case "ValidationSplit"
                        vlts = val;
                    case "Lags"
                        if numel(val)==1
                            obj.lag_vector = 1:val;
                        else
                            obj.lag_vector = val;
                        end
                    case "PCA"
                        if numel(val)==2
                            obj.PCA_pcnts = val;
                        else
                            error("Invalid entry for PCA")
                        end
                    otherwise
                        error("varargin key '" + key + "' is invalid")
                end
            end
            
            if gpuDeviceCount() > 0
                opts.ExecutionEnvironment = "gpu";
            else
                opts.ExecutionEnvironment = "cpu";
            end
            
            obj.architecture = arch;
            obj.options = opts;
            obj.setsplits(trvl,vlts)
            obj.preprocess();
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
            % this fcn fills Xr & Yr
            lag = obj.lag_vector;
            lastvalid = find(obj.date.all==obj.vlts_date);
            nontest = 1:lastvalid;
            omitlag = nontest(max(lag)+1:end);
            lasttrain = find(obj.date.all==obj.trvl_date);
            
            % Xr
            X_proto = lagmatrix(obj.Ym.all,obj.lag_vector);
            obj.centers{1} = struct('mu',mean(X_proto(omitlag,:)),'sig',std(X_proto(omitlag,:)));
            X_proto = (X_proto - obj.centers{1}.mu) ./ obj.centers{1}.sig;
            obj.transforms{1} = PCA(X_proto(omitlag,:),obj.PCA_pcnts(1));
            X_PCAd = X_proto * obj.transforms{1};
            obj.centers{3} = struct('mu',mean(X_PCAd(omitlag,:)),'sig',std(X_PCAd(omitlag,:)));
            X_PCAd = (X_PCAd - obj.centers{3}.mu) ./ obj.centers{3}.sig;
            
            obj.Xr.all = X_PCAd(lag(end)+1:end,:);
            obj.Xr.train = X_PCAd(lag(end)+1:lasttrain,:);
            obj.Xr.valid = X_PCAd(lasttrain+1:lastvalid,:);
            obj.Xr.test = X_PCAd(lastvalid+1:end,:);
            
            % Yr
            y_proto = obj.Ym.all;
            obj.centers{2} = struct('mu',mean(y_proto),'sig',std(y_proto));
            y_proto = (y_proto - obj.centers{2}.mu) ./ obj.centers{2}.sig;
            obj.transforms{2} = PCA(y_proto(nontest,:),obj.PCA_pcnts(2));
            y_PCAd = y_proto * obj.transforms{2};
            
            obj.Yr.all = y_PCAd(lag(end)+1:end,:);
            obj.Yr.train = y_PCAd(lag(end)+1:lasttrain,:);
            obj.Yr.valid = y_PCAd(lasttrain+1:lastvalid,:);
            obj.Yr.test = y_PCAd(lastvalid+1:end,:);
        end
        function obj = postprocess(obj,varargin)
        end
        function obj = setsplits(obj,trvl,vlts)
            if isa(trvl,'datetime')
                obj.trvl_date = trvl;
                iTV = find(obj.date.all==trvl);
            elseif isa(trvl,'double')
                if 0 <= trvl && trvl <= 1
                    iTV = round(obj.nDays*trvl);
                    obj.trvl_date = obj.date.all(iTV);
                else
                    iTV = trvl;
                    obj.trvl_date = obj.date.all(iTV);
                end
            else
                error("Invalid datatype for trvl argument")
            end
            if isa(vlts,'datetime')
                obj.vlts_date = vlts;
                iVT = find(obj.date.all==vlts);
            elseif isa(vlts,'double')
                if 0 <= vlts && vlts <= 1
                    iVT = round(obj.nDays*vlts);
                    obj.vlts_date = obj.date.all(iVT);
                else
                    iVT = vlts;
                    obj.vlts_date = obj.date.all(vlts);
                end
            else
                error("Invalid datatype for vlts argument")
            end
            
            obj.Ym.setsplits(iTV,iVT);
            obj.date.setsplits(iTV,iVT);
        end
    end
end