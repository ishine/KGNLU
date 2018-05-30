#!/usr/bin/env python3
#coding=utf8
import json
import os,sys

if __name__=='__main__':

    infile='rules.txt.bak'
    outfile='rules.txt'

    with open(infile,'r') as inf,open(outfile,'w') as of:
        current_dict={}
        count=0
        for line in inf:
            line=line.strip()
            if line=='':
                of.write('\n')
                continue
            elif line.startswith('%'):
                if current_dict!={}:
                    for each_pattern in current_dict:
                        count+=1
                        of.write(current_dict[each_pattern]+'=>'+each_pattern+'\n')
                current_dict={}
            elif '=>' not in line:
                of.write(line+'\n')
            else:
                pattern,node=line.split('=>')
                node=node.strip()
                if node in current_dict:
                    current_dict[node]=current_dict[node]+'|'+pattern
                else:
                    current_dict[node]=pattern
        if current_dict!={}:
            for each_pattern in current_dict:
                count+=1
                of.write(current_dict[each_pattern]+'=>'+each_pattern+'\n')
    print(count)