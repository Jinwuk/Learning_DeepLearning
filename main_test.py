#!/usr/bin/python
# -*- coding: utf-8 -*-
###########################################################################
# O'Reilly Generating Deeplearning the 2nd ED.
# Working Directory : ..\work_2025
# 2025 02 09 by Jinwuk Seok
###########################################################################
_description = '''\
====================================================
main_test.py : main_test.py for Generating Deeplearning the 2nd ED 
                    Written by Jinwuk @ 2025-02-09
====================================================
Example :  There is no Operation instruction. 
'''
from configuration import configuration
from data_proc import Fashion_MNIST



# =================================================================
# Main Routine
# =================================================================
if __name__ == "__main__":
    L_param=[]
    c_conf = configuration(L_param=L_param, _intro_msg=_description)
    c_data = Fashion_MNIST(conf_data=c_conf)

    c_conf.set_data_transform()
    train_loader, test_loader = c_data.get_dataloaders(c_conf.transform)

    print("===================================================")
    print("Process Finished ")
    print("===================================================")