import numpy as np
import math
from dataset import TextVecDataset, TrainTripletDataset, TestTwinsDataset

entity_vec = TextVecDataset('./lab2_dataset/entity_vec.txt')
relation_vec = TextVecDataset('./lab2_dataset/relation_vec.txt')


def distance(a, b):
    #print("begin distance")
    n = 100
    num_out = 0
    for i in range(100):
        num_in = math.pow(a[i]-b[i], 2)
        num_out += num_in
    num_out = math.sqrt(num_out)
   # print("exit distance")
    return num_out


def molecular(theta_h, theta_r, h, r, t):
    num_h = 0
    num_r = 0
    n = 100
    for j in range(n):
        temp = theta_h*h[j]+theta_r*r[j]-t[j]
        num_h += h[j]*temp
        num_r += r[j]*temp
    return num_h,num_r


def derivative(theta_h, theta_r, h, r, t):
    #print("begin derivative_r")
    global entity_vec, relation_vec
    num_out_h = 0
    num_out_r = 0
    m = 272115
    entity_vec = TextVecDataset('./lab2_dataset/entity_vec.txt')
    relation_vec = TextVecDataset('./lab2_dataset/relation_vec.txt')
    for i in range(m):
        k = 0
        if h[i] not in entity_vec.id_text or r[i] not in relation_vec.id_text or t[i] not in entity_vec.id_text:
            num_in_h = 0
            num_in_r = 0
            if h[i] not in entity_vec.id_text and r[i] in relation_vec.id_text and t[i] in entity_vec.id_text:
                entity_vec.id_text[h[i]] = entity_vec.id_text[t[i]
                                                              ]-theta_r*relation_vec.id_text[r[i]]
            if r[i] not in relation_vec.id_text and h[i] in entity_vec.id_text and t[i] in entity_vec.id_text:
                relation_vec.id_text[r[i]] = (
                    entity_vec.id_text[t[i]]-theta_h*entity_vec.id_text[h[i]])/theta_r
            if t[i] not in entity_vec.id_text and h[i] in entity_vec.id_text and r[i] in relation_vec.id_text:
                entity_vec.id_text[t[i]] = theta_h * entity_vec.id_text[h[i]
                                                                        ]+theta_r*relation_vec.id_text[r[i]]
        else:
            d = distance(theta_h*entity_vec.id_text[h[i]]+theta_r *
                         relation_vec.id_text[r[i]], entity_vec.id_text[t[i]])
            k_h,k_r = molecular(theta_h, theta_r,  entity_vec.id_text[h[i]], relation_vec.id_text[r[i]], entity_vec.id_text[t[i]])
            num_in_r = k_r / d
            num_in_h = k_h / d
        num_out_r += num_in_r
        num_out_h += num_in_h
    num_out_r = num_out_r/(2*m)
    num_out_h = num_out_h / (2*m)
    print("num_out_r: ", num_out_r)
    print("num_out_h:", num_out_h)
    #print("exit derivative_r")
    return num_out_r, num_out_h


def train():
    #print("begin train")
    alpha = 0.025
    theta_h = 0.001
    theta_r = 0.001
    epochs = 100
    train_vec = TrainTripletDataset('./lab2_dataset/train.txt')
    for epoch in range(epochs):
        print("第%d次训练" % (epoch+1))
        temp_r, temp_h = derivative(
            theta_h, theta_r,  train_vec.triplet["h"], train_vec.triplet["r"], train_vec.triplet["t"])
        theta_h = theta_h - alpha*temp_h
        theta_r = theta_r - alpha*temp_r
        print("theta_h=%f,theta_r=%f" % (theta_h, theta_r))
   # print("exit train")
    return theta_h, theta_r


if __name__ == '__main__':
    print("begin")
    theta_h, theta_r = train()
    print("end")
