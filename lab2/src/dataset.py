import torch
import torch.utils.data
import numpy as np

class TextVecDataset(torch.utils.data.Dataset):
    def __init__(self,datafile:str)-> None:
        super(TextVecDataset,self).__init__()
        self.id_text = {}
        self.num = 0
        with open(datafile,"r") as fr:
            for line in fr.readlines():
                index = line.strip().split('\t')[0]
                #print('index=',index)
                self.id_text[int(index)]=[]
                self.num += 1
                for s in line.strip().split('\t')[1].strip().split(','):
                   # print('s=',s)
                    #print(float(s))
                    self.id_text[int(index)].append(float(s))
                    #print(self.id_text[int(index)])
        for key in self.id_text:
            self.id_text[key] = np.array(self.id_text[key])

class TrainTripletDataset(torch.utils.data.Dataset):
    def __init__(self,datafile: str) -> None:
        super(TrainTripletDataset, self).__init__()
        self.triplet = {"h":[],"r":[],"t":[]}
        self.triplet_num = 0
        with open(datafile,"r") as fr:
            for line in fr.readlines():
                h,r,t = line.strip().split("\t")
                self.triplet["h"].append(int(h))
                self.triplet["r"].append(int(r))
                self.triplet["t"].append(int(t))
                self.triplet_num += 1
        self.triplet["h"]=np.array(self.triplet["h"])
        self.triplet["r"]=np.array(self.triplet["r"])
        self.triplet["t"]=np.array(self.triplet["t"])

class TestTwinsDataset(torch.utils.data.Dataset):
    def __init__(self,datafile: str) -> None:
        super(TestTwinsDataset,self).__init__()
        self.twins = {"h":[],"r":[],"t":[]}
        self.twins_num = 0
        with open(datafile,"r") as fr:
            for line in fr.readlines():
                h,r,t = line.strip().split("\t")
                self.twins["h"].append(int(h))
                self.twins["r"].append(int(r))
                self.twins_num += 1
        self.twins["h"]=np.array(self.twins["h"])
        self.twins["r"]=np.array(self.twins["r"])