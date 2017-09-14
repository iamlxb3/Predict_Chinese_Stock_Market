import logging
import os

def get_upper_folder_path(num, path = ''):
    if not path:
        path = os.path.dirname(os.path.abspath(__file__))
    else:
        path = os.path.dirname(path)
    num -= 1
    if num > 0:
        return get_upper_folder_path(num, path = path)
    else:
        return path

code_folder_path = get_upper_folder_path(2)







# create formatter
#=====================Formatter==================================
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
formatter_t = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
#=====================Formatter==================================

#=====================Command Line===============================
# create console handler and set level to debug
ch = logging.StreamHandler()
# DEBUG, INFO, WARNING, ERROR, CRITICAL
#ch.setLevel(logging.DEBUG)
# add formatter to ch
ch.setFormatter(formatter)
#================================================================

# :::logger1
# create logger
logger1 = logging.getLogger('logger1')
# set level
logger1.setLevel(logging.ERROR)
# save to file
hdlr_1 = logging.FileHandler('logging.log')
hdlr_1.setFormatter(formatter)
# command line
# add ch to logger

if not logger1.handlers:
    logger1.addHandler(ch)
    logger1.addHandler(hdlr_1)

#