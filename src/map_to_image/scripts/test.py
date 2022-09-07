'''
Author: fengsc
Date: 2022-07-24 12:30:44
LastEditTime: 2022-07-24 12:30:46
'''
import os, signal, subprocess
 
 
def process():
    # 由用户输入进程名
    name = input("Enter process Name: ")
    try:
        # 通过进程名来过滤ps ax的结果
        for line in os.popen("ps ax | grep " + name + " | grep -v grep"):
            print('line: {}'.format(line))
            fields = line.split()
            print('fields: {}'.format(fields))
 
            # 解析出pid
            pid = fields[0]
            print('pid: {}'.format(pid))
 
            # 以发送信号的形式杀掉进程
            os.kill(int(pid), signal.SIGKILL)
        print("Process Successfully terminated")
 
    except:
        print("Error Encountered while running script")
 
 
if __name__ == '__main__':
    process()