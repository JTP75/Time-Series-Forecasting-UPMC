import support_fcns as f
import numpy as np
import tensorflow as tf
import keras
import pandas

class Date:
    def __init__(self, m=None, d=None, y=None):
        self.month = m
        self.day = d
        self.year = y
    def datestring(self):
        if self.month == None or self.day == None or self.year == None:
            raise Exception("Error in Date.datestring(): date == uninitialized")
        return str(self.month) + "/" + str(self.day) + "/" + str(self.year)
    def __gt__(self,other):
        if self.year > other.year:
            return True
        elif self.year < other.year:
            return False
        if self.month > other.month:
            return True
        elif self.month < other.month:
            return False
        if self.day > other.day:
            return True
        elif self.day < other.day:
            return False
        else:
            return False
    def __ge__(self,other):
        if self.year > other.year:
            return True
        elif self.year < other.year:
            return False
        if self.month > other.month:
            return True
        elif self.month < other.month:
            return False
        if self.day > other.day:
            return True
        elif self.day < other.day:
            return False
        else:
            return True
    def __lt__(self,other):
        return not self >= other
    def __le__(self,other):
        return not self > other
    def __eq__(self,other):
        return self.year == other.year and self.month == other.month and self.day == other.day
    def __ne__(self,other):
        return not self == other
    def next(self):
        m = self.month
        d = self.day
        y = self.year
        
        jan = 31
        if y%4 == 0:
            feb = 29
        else:
            feb = 28
        mar = 31
        apr = 30
        may = 31
        jun = 30
        jul = 31
        aug = 31
        sep = 30
        oct = 31
        nov = 30
        dec = 31
        if m == 1 and d == jan:
            m += 1
            d = 1
        elif m == 2 and d == feb:
            m += 1
            d = 1
        elif m == 3 and d == mar:
            m += 1
            d = 1
        elif m == 4 and d == apr:
            m += 1
            d = 1
        elif m == 5 and d == may:
            m += 1
            d = 1
        elif m == 6 and d == jun:
            m += 1
            d = 1
        elif m == 7 and d == jul:
            m += 1
            d = 1
        elif m == 8 and d == aug:
            m += 1
            d = 1
        elif m == 9 and d == sep:
            m += 1
            d = 1
        elif m == 10 and d == oct:
            m += 1
            d = 1
        elif m == 11 and d == nov:
            m += 1
            d = 1
        elif m == 12 and d == dec:
            y += 1
            m = 1
            d = 1
        else:
            d += 1
            
        return Date(m,d,y)
    def prev(self):
        m = self.month
        d = self.day
        y = self.year
        jan = 31
        if y%4 == 0:
            feb = 29
        else:
            feb = 28
        mar = 31
        apr = 30
        may = 31
        jun = 30
        jul = 31
        aug = 31
        sep = 30
        oct = 31
        nov = 30
        dec = 31
        if d == 1:
            if m == 1:
                y -= 1
                m = 12
            else:
                m -= 1
            if m == 1:
                d = jan
            elif m == 2:
                d = feb
            elif m == 3:
                d = mar
            elif m == 4:
                d = apr
            elif m == 5:
                d = may
            elif m == 6:
                d = jun
            elif m == 7:
                d = jul
            elif m == 8:
                d = aug
            elif m == 9:
                d = sep
            elif m == 10:
                d = oct
            elif m == 11:
                d = nov
            elif m == 12:
                d = dec
                
        else:
            d -= 1
        return Date(m,d,y)
    def dtstr(self):
        return str(self.month) + "/" + str(self.day) + "/" + str(self.year)
        
            

class Forecaster:
    '''
    DataFrame   dataframe
    ndarray     score
    ndarray     daymat
    List<Date>  Dates
    int         ndays
    float       muL, sigL, muR, sigR, muP, sigP
    ndarray     X, y, Xtr, ytr, Xts, yts
    Date        td
    int         td_idx
    
    '''
    
    
    def __init__(self,dataframe=None):
        if dataframe is None:
            raise Exception("Error in Forecaster.__init__(): dataframe argument must not be empty")
        
        self.dataframe = dataframe
        
        score_arr_ = dataframe.Score.to_numpy()
        day_arr = 1
        score_arr_ = score_arr_.reshape(score_arr_.shape[0],1)

        score_arr = f.downsample(score_arr_,288,48)
        self.score = score_arr

        daymat = f.getdaymat(score_arr,48)
        self.daymat = daymat
        self.ndays = daymat.shape[0]
        dt = Date(6,27,2018)
        dates = []
        for i in range(self.ndays):
            dates.append(dt)
            dt = dt.next()
        self.Dates = dates
        
        self.prepdata()
        
    def prepdata(self,PCALpcnt=0.90,PCARpcnt=0.90,date_idx=1108):
        # lagmatrix 
        nlags = 14
        lagmat = f.lagmatrix(self.daymat,nlags)      # first 14 rows contain NaNs
        daymat = self.daymat
        lagmat_cropped = lagmat[nlags:,:]

        # pre-pca lag centering
        self.muL = np.mean(lagmat_cropped[:date_idx,:],0)
        self.sigL = np.std(lagmat_cropped[:date_idx,:],0)
        LMC_cent = (lagmat_cropped - self.muL) / self.sigL
        
        # pre-pca resp centering
        self.muR = np.mean(daymat[:date_idx,:],0)
        self.sigR = np.std(daymat[:date_idx,:],0)
        daymat_cent = (daymat - self.muR) / self.sigR

        # pca
        transformL = f.PCA(LMC_cent,PCALpcnt)
        transformR = f.PCA(daymat_cent,PCARpcnt)
        
        # network tensors
        X = np.dot(LMC_cent, transformL.T)
        y = np.dot(daymat_cent, transformR.T)
        muP = np.mean(X[:date_idx,:],0)
        sigP = np.std(X[:date_idx,:],0)
        X = (X - muP) / sigP

        # full network tensors
        self.X = np.reshape(X,(X.shape[0],X.shape[1],1,1))
        self.y = y[nlags:]

    def set_split(self,date):
        self.td = date
        training_days = 0
        found = False
        for dt in self.Dates:
            if dt == date:
                found = True
                break
            training_days += 1
        if not found:
            raise Exception("Error in Forecaster.set_split(): date argument entered is outside of known data set")
        self.td_idx = training_days
        self.Xtr = self.X[0:training_days]
        self.ytr = self.y[0:training_days]
        self.Xts = self.X[training_days+1:self.ndays]
        self.yts = self.y[training_days+1:self.ndays]
        
        

'''
print("Loading data set...")

try:
  mf = pandas.read_excel(r'NEDOC_DATA.xlsx')
except:
  print("Load failed!\n")
  exit()
  
print("Load successful!\n")


fc = Forecaster(mf)
'''