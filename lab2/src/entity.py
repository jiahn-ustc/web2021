from gensim.models import word2vec
import numpy as np
import json

path_with = './lab2_dataset/entity_with_text.txt'
path_co = './lab2_dataset/entity_co.txt'
path_vec = './lab2_dataset/entity_vec.txt'


with open(path_with, 'r') as fr,open(path_co,'w') as fw:
    for line in fr.readlines():
        fw.write(line.strip().split('\t')[1])
        #print(line.strip().split('\t'))
        fw.write('\n')

sentences = word2vec.LineSentence(path_co);
model = word2vec.Word2Vec(sentences);


with open(path_with,'r') as fr,open(path_vec,'w') as fw:
    for line in fr.readlines():
        vec = 0
        entity_id = line.strip().split('\t')[0]
        words = line.strip().split('\t')[1].split(' ')
        for word in words:
            if word in model.wv:
                vec += model.wv[word]
        vec = vec/np.linalg.norm(vec)
        s = json.dumps(vec.tolist())
        s = s.strip("[]")
        fw.write(entity_id+'\t'+s+'\n')
        
