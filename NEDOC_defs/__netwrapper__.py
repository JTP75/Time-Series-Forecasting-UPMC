from __wrapperbase__ import __wrapperbase__, IO_Struct
import abc

class __netwrapper__(__wrapperbase__):

    @abc.abstractmethod
    def __init__(self):
        super.__init__()
        self.perf_fcn=None

    @abc.abstractmethod
    def print_network_stoolpigeon(self):
        pass

    