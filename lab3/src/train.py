from dataset import MusicDataset
import numpy as np

outpath = "./result/similarity.txt"
data = MusicDataset("./data/DoubanMusic.txt")

def similarity(a,b):
    intersection = np.intersect1d(a,b)
    union = np.union1d(a,b)
    result = intersection.size / union.size
    return result

#compute the similarity each id
#result = {}
with open(outpath,'w') as fw: 
    for id in data.id_Music.keys():
        #store the similarity between id and others besides id itself
        print("id: ", id)
        fw.write(str(id)+'\t')
        #result[id]=[]
        for key in data.id_Music.keys():
            #print("key: ", key)
            similar = similarity(data.id_Music[id],data.id_Music[key])
            #result[id].append(float(similar))
            if(similar > 0):
                fw.write(str(key)+','+str(format(similar,'.4f'))+' ')
        fw.write('\n')
        