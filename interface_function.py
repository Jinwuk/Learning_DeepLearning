#!/usr/bin/python
# -*- coding: utf-8 -*-
###########################################################################
# O'Reilly Generating Deeplearning the 2nd ED.
# Working Directory : ..\work_2025
# 2025 02 09 by Jinwuk Seok
###########################################################################
_description = '''\
====================================================
interface_function.py : configuration.py for Generating Deeplearning the 2nd ED 
                    Written by Jinwuk @ 2025-02-09
====================================================
Example :  There is no Operation instruction. 
'''
import argparse, textwrap
import json, yaml

def ArgumentParse(L_Param, _prog, _intro_msg=_description, bUseParam=False):
    parser = argparse.ArgumentParser(
        prog=_prog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(_intro_msg))

    parser.add_argument('-fc', '--fundamental_configure_file', help="fundamental_configure_file",
                        type=str, default='fundamental_configure.yaml')
    parser.add_argument('-dp', '--data_padding_size', help="data padding size (Default:2)",
                        type=int, default=2)
    parser.add_argument('-nw', '--number_of_workers', help="number_of_workers (Default:4)",
                        type=int, default=4)

    parser.add_argument('-rs', '--report_style', action='store_true', help="[gemini_ai] Generate Report Style (Default : False)",
                        default=False)
    parser.add_argument('-mq', '--math_question', action='store_true', help="[gemini_ai] Is question a mathematical question (Default : False)",
                        default=False)

    args = parser.parse_args(L_Param) if bUseParam else parser.parse_args()

    print(_intro_msg)
    return args


def get_source(_work_fullpath):
    with open(_work_fullpath, 'r', encoding='utf-8') as _file:
        _contents = _file.read()
    return _contents
def put_result(_outfile, _contents):
    with open(_outfile, 'w', encoding='utf-8') as _file:
         _file.write(_contents)
def read_json(_json_file):
    with open(_json_file, 'r', encoding='utf-8') as _file:
        _dict_data = json.load(_file)
    return _dict_data
def read_yaml(_yaml_file):
    with open(_yaml_file, 'r',  encoding='utf-8') as _file:
        _config = yaml.safe_load(_file)
    return _config
def write_yaml(_yaml_file, _data):
    with open(_yaml_file, 'w', encoding='utf-8') as _file:
        yaml.dump(_data, _file, default_flow_style=False)

# =================================================================
# Main Routine
# =================================================================
if __name__ == "__main__":

    args = ArgumentParse(L_Param=[], _prog=__file__, _intro_msg=_description)
    _config = read_yaml(args.fundamental_configure_file)
    print(_config)

    print("===================================================")
    print("Process Finished ")
    print("===================================================")
