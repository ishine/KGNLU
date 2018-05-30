#!/usr/bin/env python3
#coding=utf8
import os,sys,json
import argparse

def main(argv):
    parser=argparse.ArgumentParser()
    parser.add_argument('-f','--infile',default='test.original.txt')
    parser.add_argument('-o','--outfile',default='eval.txt')
    args=parser.parse_args()
    with open(args.infile,'r') as inf,open(args.outfile,'w') as of:
        for line in inf:
            line=line.strip()
            if line=='':
                continue
            query,annotation=line.split('<=>')
            query,annotation=query.strip(),annotation.strip()
            annotation=json.loads(annotation)
            for each in annotation:
                if each.startswith('er'):
                    if not annotation[each]["ent_idx"]:
                        new={'expr_type':'entity'}
                        annotation[each]=dict(new,**annotation[each])
                    else:
                        new={'expr_type':'rel_attr'}
                        annotation[each]=dict(new,**annotation[each])
            of.write(' <=> '.join([query,json.dumps(annotation,ensure_ascii=False)])+'\n')

if __name__=='__main__':

    main(sys.argv)