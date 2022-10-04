function [Xr,Yr,centers,transforms] = dataprep_shell(ds, varargin)

% process fcn args
PCA_pcnt = [0.90 0.90];
Lag = 1:14;
doValidation = false;
vpcnt = 0.8;
for arg_idx = 1:2:length(varargin)
    if strcmp('PCApcnt',varargin{arg_idx})
        PCA_pcnt = varargin{arg_idx+1};
        if length(PCA_pcnt) == 1
            PCA_pcnt = [PCA_pcnt PCA_pcnt]; %#ok<AGROW>
        end
    elseif strcmp('Lags',varargin{arg_idx})
        Lag = varargin{arg_idx+1};
        if length(Lag) == 1
            Lag = 1:Lag;
        end
    elseif strcmp('Validate',varargin{arg_idx})
        doValidation = varargin{arg_idx+1};
    elseif strcmp('ValidPcnt',varargin{arg_idx})
        vpcnt = varargin{arg_idx+1};
    end
end

% get data
MMM = ds.getmats('setspec','all','matspec','days');
day_matrix = MMM;
td = floor(ds.today.i / ds.PPD);
vd = floor(vpcnt * td);

% lag mat
X = lagmatrix(day_matrix,Lag);

% pre-PCA lag centering
mu_L = mean(X(Lag(end)+1:td,:));
sig_L = std(X(Lag(end)+1:td,:));
X = (X - mu_L) ./ sig_L;

% pre-PCA resp centering
mu_R = mean(day_matrix(1:td,:));
sig_R = std(day_matrix(1:td,:));
day_matrix = (day_matrix - mu_R) ./ sig_R;

% PCA
eigvecs_L = PCA(X(Lag(end)+1:td,:),PCA_pcnt(1));
eigvecs_R = PCA(day_matrix(1:td,:),PCA_pcnt(2));

% transform
X = X * eigvecs_L;
y = day_matrix * eigvecs_R;

% post-PCA centering
mu_P = mean(X(Lag(end)+1:end,:));
sig_P = std(X(Lag(end)+1:end,:));
X = (X - mu_P) ./ sig_P;

% split data
X_all = X(Lag(end)+1:end,:);
y_all = y(Lag(end)+1:end,:);
if doValidation
    X_train = X(Lag(end)+1:vd,:);
    y_train = y(Lag(end)+1:vd,:);
    X_valid = X(vd+1:td,:);
    y_valid = y(vd+1:td,:);
    X_test = X(td+1:end,:);
    y_test = y(td+1:end,:);
else
    X_train = X(Lag(end)+1:td,:);
    y_train = y(Lag(end)+1:td,:);
    X_valid = [];
    y_valid = [];
    X_test = X(td+1:end,:);
    y_test = y(td+1:end,:);
end

% generate cell arrays
Xc_all = mat2cellR(X_all);
Xc_train = mat2cellR(X_train);
Xc_valid = mat2cellR(X_valid);
Xc_test = mat2cellR(X_test);

% generate output structs
Xr = struct(    'all',{Xc_all} , 'train',{Xc_train} , ...
                'valid',{Xc_valid} , 'test',{Xc_test},...
                'all_num',{X_all} , 'train_num',{X_train} ,... 
                'valid_num',{X_valid} , 'test_num',{X_test} ...
);
Yr = struct( 'all',y_all , 'train',y_train , 'valid',y_valid , 'test',y_test );
centers = struct( 'muL',mu_L , 'sigL',sig_L , 'muR',mu_R , 'sigR',sig_R , 'muP',mu_P , 'sigP',sig_P );
transforms = struct( 'PCAL',eigvecs_L , 'PCAR',eigvecs_R , 'lag',Lag(end) );




