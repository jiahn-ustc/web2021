import functools
import numpy as np

inpath = "./result/similarity.txt"
outpath = "./result/sort_similarity.txt"

def custom_sort(x,y):
    a = float(x.strip().split(',')[1])
    b = float(y.strip().split(',')[1])
    if a > b:
        return -1
    if a < b:
        return 1
    return 0

with open(inpath, 'r') as fr,open(outpath, 'w') as fw:
    for line in fr.readlines():
        id = line.strip().split('\t')[0]
        print('id=', id)
        fw.write(str(id)+'\t')
        score_list = line.strip().split('\t')[1].split(' ')
        #print(score_list)
        x = sorted(score_list,key=functools.cmp_to_key(custom_sort))
        for val in x:
            fw.write(val+' ')
        fw.write('\n')

