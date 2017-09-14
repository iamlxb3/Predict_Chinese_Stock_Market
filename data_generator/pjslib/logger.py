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
logger1.setLevel(logging.INFO)
# save to file
hdlr_1 = logging.FileHandler(os.path.join(code_folder_path,'logging.log'))
hdlr_1.setFormatter(formatter)
# command line
# add ch to logger

if not logger1.handlers:
    logger1.addHandler(ch)
    logger1.addHandler(hdlr_1)

#==================================================================
# :::logger1
# create logger
logger2 = logging.getLogger('logger2')
# set level
logger2.setLevel(logging.INFO)
# save to file
hdlr_2 = logging.FileHandler(os.path.join(code_folder_path,'mlp_logging.log'))
hdlr_2.setFormatter(formatter)
# command line
# add ch to logger
if not logger2.handlers:
    logger2.addHandler(ch)
    logger2.addHandler(hdlr_2)
#==========================================================
#
# # create logger
# logger_t = logging.getLogger('logger_temp')
# # set level
# logger_t.setLevel(logging.INFO)
# # save to file
#
# hdlr_t = logging.FileHandler(os.path.join(code_folder_path, 'temp_logging.log'))
# hdlr_t.setFormatter(formatter_t)
# # command line
# # add ch to logger
# if not logger_t.handlers:
#     logger_t.addHandler(ch)
#     logger_t.addHandler(hdlr_t)
#
#
# # # =========================================
# # # create logger
# # oanda_logger = logging.getLogger('oanda_logger')
# # # set level
# # oanda_logger.setLevel(logging.INFO)
# # # save to file
# # hdlr_1 = logging.FileHandler('logging/oanda_logging.log')
# # hdlr_1.setFormatter(formatter)
# # # command line
# # # add ch to logger
# # oanda_logger.addHandler(ch)
# # oanda_logger.addHandler(hdlr_1)