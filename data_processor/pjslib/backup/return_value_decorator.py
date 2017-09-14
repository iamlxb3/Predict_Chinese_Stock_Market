import time
import inspect
import collections
# def time_elapse(func):
    # def d_func(*args, **kwargs):
        # t1 = time.time()
        # return_func =  func(*args,**kwargs)
        # t2 = time.time()
        # time_elapse = t2 - t1
        # print (func.__name__, time_elapse)
        # return return_func 
    # return d_func

# @time_elapse
# def loop1(num):
    # n = 0
    # for i in range(num):
        # n += i
    # return n

# #print (get_text("John"))
# a = loop1(100000)




#--------------class version---------------
class WriteReturnValue():
    return_value_dict = collections.defaultdict(lambda: [])
    path_list = []
    def __init__(self, mode = 'normal', path = 'return_result.txt', func_name = ''):
        #self.f = f
        self.mode = mode
        self.path = path
        self.run_function = set()
        self.func_name = func_name
        
    @staticmethod
    def clear():
        WriteReturnValue.run_function = set()

    @staticmethod
    def write():
        for path, tuple_list in WriteReturnValue.return_value_dict.items():
            with open(path, 'w') as f:
                # alphabat order
                tuple_list = sorted(tuple_list, key = lambda x:x[0])
                for tuple1 in tuple_list:
                    f.write(str(tuple1[0]) + '\n')
                    f.write(tuple1[1])
                    f.write('\n')
                
                
        
    def __call__(self,f):
    
        def get_class_that_defined_method(meth):
            '''copy from stackoverflow, '''
            '''http://stackoverflow.com/questions/3589311/get-defining-class-of-unbound-method-object-in-python-3/25959545#25959545'''
            if inspect.ismethod(meth):
                for cls in inspect.getmro(meth.__self__.__class__):
                    if cls.__dict__.get(meth.__name__) is meth:
                        return cls
                meth = meth # fallback to __qualname__ parsing
            if inspect.isfunction(meth):
                cls = getattr(inspect.getmodule(meth),
                              meth.__qualname__.split('.<locals>', 1)[0].rsplit('.', 1)[0])
                if isinstance(cls, type):
                    return cls
            return None # not required since None would have been implicitly returned anyway
        
        
        def unique_func_name():
            if not get_class_that_defined_method(f):
                class_name = ''
            else:
                class_name = get_class_that_defined_method(f).__name__
            func_name = class_name + '_' + f.__name__
            return func_name
        
        
        def get_return_value(*args,**kwargs):
            return_func =  f(*args,**kwargs)
            func_return_type = type(return_func)
            return_str = "------------------{}------------------\n".format(func_return_type)
            if func_return_type.__name__ == "dict":
                for key, value in return_func.items():
                    return_str += key + ':' + value + '\n'
                return_str += "----------------END------------------\n"
                    
            else:
                return_str += str(return_func)
                return_str += "\n----------------END------------------\n"
            return return_func, return_str
            
        
        def update_return_value_dict(path, f_name, return_str):
            WriteReturnValue.return_value_dict[path].append((f_name, return_str))
            
        
        def wrapped_f(*args,**kwargs):
            #TODO how to avoid run multiple times, it is ok within 200000 loops
            def run_once(*args,**kwargs):
                #print ("run_once!!!!!!!!!!!!!")
                # assign func_name
                if not self.func_name:
                    func_name = unique_func_name()
                else:
                    func_name = self.func_name
                if func_name in self.run_function:
                    print ("{} already exist!".format(func_name))
                    return 
                else:
                    return_func, return_str = get_return_value(*args,**kwargs)
                    self.run_function.add(func_name)
                    # update return_value_dict
                    f_name = func_name
                    path = self.path
                    update_return_value_dict(path, f_name, return_str)
                    return return_func
                
            def normal(*args,**kwargs):
                if not self.func_name:
                    func_name = unique_func_name()
                else:
                    func_name = self.func_name
                print(type(func_name))
                return_func, return_str = get_return_value(*args,**kwargs)
                # update return_value_dict
                f_name = func_name
                path = self.path
                update_return_value_dict(path, f_name, return_str)
                return return_func
                
            #print (self.mode)
            wrap_dict = {'normal': normal,
                         'once': run_once,
                         }
            return wrap_dict[self.mode](*args,**kwargs)
            
            
        return wrapped_f
    
    