class mydecorator(object):
    def __init__(self):
        pass
    
    def asd2(self):
        a = 2 + 1
        return a

    def __call__(self, f):
        def wrapped_f(*args):
            print ("aaa")
            res = self.asd2(*args)
            return res
        return wrapped_f
        
        
@mydecorator()
def asd():
    a = 2 + 2
    return a
    
aaa = asd()

#=======================================================================================
# >>> aaa
# 3

# >>> asd
# <function mydecorator.__call__.<locals>.wrapped_f at 0x00000004970332F0>

# >>> type(asd)
# <class 'function'>

# >>> asd()
# aaa
# 3
#=======================================================================================
