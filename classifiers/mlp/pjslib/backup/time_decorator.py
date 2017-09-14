import time
import inspect
import collections






#--------------TimeElapse Decorator---------------
class TimeElapse():
    func_time_list = []
    def __init__(self, mode = 'normal'):
        #self.f = f
        # mode:('once', normal')
        self.mode = mode
        self.run_function = set()
        
    @staticmethod
    def clear(self):
        self.run_function = set()

    @staticmethod
    def write():
        path = 'time.txt'
        func_time_list  = sorted(TimeElapse.func_time_list, key = lambda x:x[1], reverse = True)
        with open(path, 'w') as f:
            for tuple1 in func_time_list:
                f.write(str(tuple1) + '\n')
        
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
        
        def compute_func_time_elapse(*args,**kwargs):
            t1 = time.time()
            return_func =  f(*args,**kwargs)
            t2 = time.time()
            time_elapse = t2 - t1
            #print (f.__doc__)
            print (f.__name__, time_elapse)
            return return_func, time_elapse
        
        def update_func_time_list(f_name, time_elapse):
            TimeElapse.func_time_list.append((f_name, time_elapse))
        
        def wrapped_f(*args,**kwargs):
            #TODO how to avoid run multiple times, it is ok within 200000 loops
            def run_once(*args,**kwargs):
                #print ("run_once!!!!!!!!!!!!!")
                func_name = unique_func_name()
                if func_name in self.run_function:
                    print ("{} already exist!".format(func_name))
                    return 
                else:
                    return_func, time_elapse = compute_func_time_elapse(*args,**kwargs)
                    self.run_function.add(func_name)
                    # update func_time_list
                    f_name = func_name
                    update_func_time_list(f_name, time_elapse)
                    return return_func
                
            def normal(*args,**kwargs):
                func_name = unique_func_name()
                print(type(func_name))
                return_func, time_elapse = compute_func_time_elapse(*args,**kwargs)
                # update func_time_list
                f_name = func_name
                update_func_time_list(f_name, time_elapse)
                return return_func
                
            #print (self.mode)
            wrap_dict = {'normal': normal,
                         'once': run_once,
                         }
            return wrap_dict[self.mode](*args,**kwargs)
            
            
        return wrapped_f
    
    