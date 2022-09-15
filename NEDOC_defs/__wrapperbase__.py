import abc

import numpy as np

class IO_Struct:
    def __init__(self,all,train,test,valid):
        self.all=all
        self.train=train
        self.test=test
        self.valid=valid

class __wrapperbase__(abc.ABC):

    @abc.abstractmethod
    def __init__(self):
        self.X = None
        self.Y = None
        self.YP = None
        
    @abc.abstractmethod
    def set(self):
        pass

    @abc.abstractmethod
    def prep(self):
        pass

    @abc.abstractmethod
    def deprep(self):
        pass

    @abc.abstractmethod
    def assess(self):
        pass

