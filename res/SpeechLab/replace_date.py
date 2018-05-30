#!/usr/bin/env python
#coding=utf8
import re,sys,os
replace_table={'一':'1','二':'2','三':'3','四':'4','五':'5','六':'6','七':'7','八':'8','九':'9','零':'0','两':'2'}
for i in range(10):
    replace_table[str(i)]=str(i)

infile='test.txt.questionnaire'
outfile='test.txt.questionnaire.new'

def replace_date(string):
    new=''
    for each in string:
        new+=replace_table[each]
    return new
pattern=re.compile(r'[\d一二三四五六七八九零两]{4}')
add_year=re.compile(r'([\d一二三四五六七八九零两]{4})([^年])')
with open(infile,'r') as inf,open(outfile,'w') as of:
    for line in inf:
        line=line.strip()
        if line=='':
            of.write('\n')
            continue

        l=pattern.findall(line)
        for each in l:
            line=line.replace(each,replace_date(each),1)

        of.write(re.sub(add_year,r'\1年\2',line)+'\n')