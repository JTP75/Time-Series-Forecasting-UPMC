from __netwrapper__ import __netwrapper__
from __wrapperbase__ import IO_Struct
import support_fcns as f
import numpy as np
from warnings import warn

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

class CenterStruct:
    def __init__(self,mu,sig):
        self.mu=mu
        self.sig=sig

class NEDOC_Wrapper(__netwrapper__):
    def __init__(self,**kwargs):
        super.__init__()
        self.perf_fcn = None
        self.training_pcnt = 0.95
        self.validation_pcnt = 0.00
        self.testing_pcnt = 0.05
        self.score_array = None
        self.daymat = None
        self.center_L = None
        self.center_R = None
        self.center_P = None
        self.transform_L = None
        self.transform_R = None
        self.today = None
        self.PPD = 48
        self.set(kwargs)
        
    def set(self, **kwargs):
        for key in ['X','Y','perf_fcn','training_pcnt','validation_pcnt','testing_pcnt','score_arr','daymat','today','PPD','setday1']:
            if key in kwargs:
                setattr(self,key,kwargs[key])

        if self.score_arr == None and self.daymat == None:
            warn("Warning from NEDOC_Wrapper.set(): the attributes score_arr and daymat are both unset. these must be set before calling prep()")
        elif self.score_arr == None:
            score_arr = np.reshape(self.daymat,(self.daymat.shape[0]*self.daymat.shape[1],1))
        elif self.daymat == None:
            daymat = np.reshape(self.score_arr,(self.score_arr[0]/self.PPD,self.PPD))
        self.today = Date()
        


        


    def prep(self, **kwargs):
        # process kwargs
        PCAL = 0.90
        PCAR = 0.90
        for key in kwargs:
            if key == 'PCA_pcnt_L':
                PCAL = kwargs[key]
            elif key == 'PCA_pcnt_R':
                PCAR = kwargs[key]
        
        
        
        
        
        

                
