#!/usr/bin/python
# -*- coding: utf-8 -*-
###########################################################################
# O'Reilly Generating Deeplearning the 2nd ED.
# Working Directory : ..\work_2025
# 2025 02 09 by Jinwuk Seok
###########################################################################
_description = '''\
================================================================
confirm_gpu.py : confirm the usablity of the system  
                    Written by Jinwuk @ 2025-02-19
================================================================
Example :  There is no Operation instruction. 
'''
g_line      = "----------------------------------------------------------------"

import torch

# =================================================================
# Test Routine
# =================================================================
if __name__ == "__main__":

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"사용 가능한 GPU 개수: {device_count}")
        for i in range(device_count):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA를 사용할 수 없습니다.")

    print("===================================================")
    print("Process Finished ")
    print("===================================================")

