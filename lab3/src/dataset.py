import torch
import torch.utils.data
import numpy as np

class MusicDataset(torch.utils.data.Dataset):
    def __init__(self,datafile:str)-> None:
        super(MusicDataset,self).__init__()
        self.id_Music = {}
        with open(datafile,"r") as fr:
            for line in fr.readlines():
                if(line=='\n'):
                    break
                index = line.strip().split('\t')[0]
               # print('index=',index)
                self.id_Music[int(index)]=[]
                for s in line.strip().split('\t')[1:]:
                    #print('s=',s)
                    #print(float(s))
                    t = s.split(',')[0]
                    #print('t=',t)
                    self.id_Music[int(index)].append(int(t))
                    #print(self.id_Music[int(index)])
        for key in self.id_Music:
            self.id_Music[key] = np.array(self.id_Music[key])

if __name__ =='__main__':
    path = "./data/DoubanMusic.txt"
    music = MusicDataset(path)
    for key in music.id_Music.keys():
        print(music.id_Music[key])