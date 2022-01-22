from dataset import TestTwinsDataset,TextVecDataset,TrainTripletDataset
import math
import numpy as np
from train import train


test_vec = TestTwinsDataset('./lab2_dataset/test.txt')
entity_vec = TextVecDataset('./lab2_dataset/entity_vec.txt')
relation_vec = TextVecDataset('./lab2_dataset/relation_vec.txt')

def distance(a,b):
    n=100
    num_out = 0 
    for i in range(n):
        num_in = math.pow(a[i]-b[i],2)
        num_out += num_in
    num_out = math.sqrt(num_out)
    return num_out



if __name__ == '__main__':
    theta_h, theta_r=train()
    with open('./lab2_dataset/result.txt','w')as fw:
        num = test_vec.twins_num
        print('num = %d' % num)
        for i in range(num):
            print('正在写第%d行' %i)
            index_h = test_vec.twins['h'][i]
            index_r = test_vec.twins['r'][i]
            d ={}
            if index_h in entity_vec.id_text and index_r in relation_vec.id_text:
                t = theta_h * entity_vec.id_text[index_h]+ theta_r * relation_vec.id_text[index_r]
            else:
                t = np.zeros(100)
            for j in entity_vec.id_text:
                d[j]=distance(t,entity_vec.id_text[j])
            d=sorted(d.items(),key=lambda x:x[1])
            fw.write(str(d[0][0]))
            for k in range(1,5):
                fw.write(','+str(d[k][0]))
            fw.write('\n')
            
