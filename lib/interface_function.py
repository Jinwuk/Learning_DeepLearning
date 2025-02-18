#!/usr/bin/python
# -*- coding: utf-8 -*-
###########################################################################
# O'Reilly Generating Deeplearning the 2nd ED.
# Working Directory : ..\work_2025
# 2025 02 09 by Jinwuk Seok
###########################################################################
_description = '''\
================================================================
interface_function.py : configuration.py for Generating Deeplearning the 2nd ED 
                    Written by Jinwuk @ 2025-02-09
================================================================
Example :  There is no Operation instruction. 
'''
import argparse, textwrap
import json, yaml
from pathlib import Path
import my_debug as DBG
def ArgumentParse(L_Param, _prog, _intro_msg=_description, bUseParam=False):
    parser = argparse.ArgumentParser(
        prog=_prog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(_intro_msg))
    #----------------------------------------------------------------
    # Help format
    #"Save graphics Not use plt.show (Default : False)"
    #
    # ----------------------------------------------------------------
    parser.add_argument('-fc', '--fundamental_configure_file',
                        help="fundamental_configure_file     (Default : fundamental_configure.yaml",
                        type=str, default='fundamental_configure.yaml')
    parser.add_argument('-dg', '--debug_mode', action='store_true',
                        help="Debug mode activation          (Default : False)",
                        default=False)
    parser.add_argument('-pm', '--processing_mode',
                        help='''\
                        Processing Mode [0] Developing [1] AutoEncoder [2] AutoEncoder+Classifier \n
                        [3] VAE [4] VAE+Classifier (Default : 0)"
                        ''', type=int, default=0)
    parser.add_argument('-lf', '--label_file',
                        help="Label File for Data            (Default : data_label.yaml)",
                        type=str, default='data_label.yaml')
    parser.add_argument('-dp', '--data_padding_size',
                        help="Data padding size              (Default : 2)",
                        type=int, default=2)
    parser.add_argument('-nw', '--number_of_workers',
                        help="Number_of_workers              (Default : 4)",
                        type=int, default=4)
    parser.add_argument('-gi', '--gpu_id',
                        help="GPU ID                         (Default : 0)",
                        type=int, default=0)
    parser.add_argument('-qm', '--quite_mode', action='store_true',
                        help="Quite mode                     (Default : False)",
                        default=False)
    parser.add_argument('-im', '--inference_mode', action='store_true',
                        help="Inference mode                 (Default : False)",
                        default=False)
    parser.add_argument('-ot', '--optimizer_spec',
                        help="Optimizer Specification        (Default : optimizer.yaml)",
                        type=str, default='optimizer.yaml')
    parser.add_argument('-sg', '--save_graphic', action='store_true',
                        help="Save graphics Not use plt.show (Default : False)",
                        default=False)

    args = parser.parse_args(L_Param) if bUseParam else parser.parse_args()

    print(_intro_msg)
    return args

def get_source(_work_fullpath):
    with open(_work_fullpath, 'r', encoding='utf-8') as _file:
        _contents = _file.read()
    return _contents
def put_result(_outfile, _contents, _mode='w'):
    with open(_outfile, _mode, encoding='utf-8') as _file:
         _file.write(_contents)
def read_json(_json_file):
    with open(_json_file, 'r', encoding='utf-8') as _file:
        _dict_data = json.load(_file)
    return _dict_data
def write_json(_json_file, _data):
    with open(_json_file, 'w', encoding='utf-8') as _file:
        json.dump(_data, _file, indent = 4)
def read_yaml(_yaml_file):
    with open(_yaml_file, 'r',  encoding='utf-8') as _file:
        _config = yaml.safe_load(_file)
    return _config
def write_yaml(_yaml_file, _data):
    with open(_yaml_file, 'w', encoding='utf-8') as _file:
        yaml.dump(_data, _file, default_flow_style=False)
# ----------------------------------------------------------------
# Modify YAML or Jason
# ----------------------------------------------------------------
def find_dict(d_target, _target_key, _new_value):
    for _key, _value in d_target.items():
        if isinstance(d_target.get(_key), dict):
            find_dict(_value, _target_key=_target_key, _new_value=_new_value)
        else:
            if _key == _target_key : d_target[_key] = _new_value
            else : pass
    return d_target

def modify_parameter_file(_file_name, _target_key, _new_value, b_quite=True):
    _yaml_or_jason = Path(_file_name).suffix

    if _yaml_or_jason == '.yaml':
        f_read_parameter_file = read_yaml
        f_write_parameter_file= write_yaml
    elif _yaml_or_jason == '.json':
        f_read_parameter_file = read_json
        f_write_parameter_file= write_json
    else:
        DBG.dbg("Unknown File Format (%s)!!! Program terminated!" %_yaml_or_jason)
        exit(0)

    my_dict = f_read_parameter_file(_file_name)

    if b_quite == False:
        print("Original Parameters")
        for _key, _value in my_dict.items():
            print("%16s : " %_key, _value)
    else : pass

    my_dict = find_dict(my_dict, _target_key=_target_key, _new_value=_new_value)

    if b_quite == False:
        print("Updated Parameters")
        for _key, _value in my_dict.items():
            print("%16s : " %_key, _value)
    else: pass

    f_write_parameter_file(_file_name, _data=my_dict)

# =================================================================
# Main Routine
# =================================================================
if __name__ == "__main__":

    args = ArgumentParse(L_Param=[], _prog=__file__, _intro_msg=_description)

    _config = modify_parameter_file(_file_name="fundamental_configure.yaml", _target_key="OPTIMIZER", _new_value="Adam", b_quite=False)

    print("===================================================")
    print("Process Finished ")
    print("===================================================")
