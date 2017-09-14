import os
import re
import sys
import collections


class Deleter:

    def __init__(self, delete_file_format_list):
        self.delete_file_format_list = delete_file_format_list
        self.delete_dict = collections.defaultdict(lambda: 0)


    def _get_all_paths_for_delete(self, delete_folder_list):
        for directory in delete_folder_list:
            file_name_list = os.listdir(directory)
            if not file_name_list:
                print ("Nothing in directory: {}".format(directory))
                continue
            else:
                file_path_list = [os.path.join(directory, x) for x in file_name_list]
                directory_list = []
                delete_list = []
                for i, file_path in enumerate(file_path_list):
                    ISDIR = os.path.isdir(file_path)
                    if ISDIR:
                        directory_list.append(file_path)
                    else:
                        file_name = file_name_list[i]
                        for delete_file_format in self.delete_file_format_list:
                            ISFORMAT = re.findall(delete_file_format, file_name)
                            if ISFORMAT:
                                delete_list.append(file_path)

                if not delete_list and not directory_list:
                    print("No more target files and directories in directory: {}".format(directory))
                    continue
                else:

                    # update delete_dict
                    if delete_list:
                        self.delete_dict[directory] = delete_list
                    #

                    # continue delete sub-folders
                    self._get_all_paths_for_delete(directory_list)
                    #

    def delete_all_files(self, delete_folder_list):
        self._get_all_paths_for_delete(delete_folder_list)

        if self.delete_dict:
            for directory, delete_list in self.delete_dict.items():
                count = 0
                for path in delete_list:
                    os.remove(path)
                    count += 1

                print ("Successfully remove {} {} files in directory {}".format(count, self.delete_file_format_list,
                                                                                  directory))
        else:
            print ("\nNothing to delete!")




parent_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

hyper_parameter_folder = 'hyper_parameter_test'

hyper_parameter_folder = os.path.join(parent_folder, hyper_parameter_folder)
hyper_parameter_folder_list = [hyper_parameter_folder]

delete_file_format_list = ['.csv']


input1 = input("Do you really want to delete all {} files in directory {} (y/n)?".
               format(delete_file_format_list, hyper_parameter_folder_list))
if input1 == 'y':
    deleter1 = Deleter(delete_file_format_list)
    deleter1.delete_all_files(hyper_parameter_folder_list)
else:
    sys.exit()

