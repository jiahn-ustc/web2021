import numpy as np
from dataset import MusicDataset
from ordered_set import OrderedSet

path = "./result/sort_similarity.txt"
data = MusicDataset("./data/DoubanMusic.txt")


with open(path, 'r') as fr,open("./result/result.txt",'w')as fw:
    for line in fr.readlines():
        s = line.strip().split('\t')
        index = int(s[0])
        print('index=',index)
        fw.write(str(index)+'\t')
        a = OrderedSet()
        for value in s[1].strip().split(' '):
            if(len(a)>=100):
                break
            target = int(value.strip().split(',')[0])
            for j in data.id_Music[target]:
                if(len(a)<=99):
                    a.add(str(j))
                else:
                    break
        count=0
        for i in a:
            if(count<99):
                fw.write(i+',')
                count = count +1    
            else:
                fw.write(i)
                count = count +1
        fw.write('\n')
            

