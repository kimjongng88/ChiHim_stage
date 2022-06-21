import sys
import os
from collections import Counter
import json

class segment_:
    def __init__(self, sum, seg_number, initials, label):
        self.sum = sum
        self.seg_number = seg_number
        self.label = label
        self.initials = initials
        self.cards = []

if __name__ == '__main__':
    print('begin')
    no_dubs = {}
    fh = open('ai_needle_emg-master/analyse/Correct_0-0299/double_0220_list.txt').readlines()
    #fh = open('ai_needle_emg-master/analyse/lijst.txt').readlines()
    sum = 0
    for line in fh:               
        row = line.split(',')
        seg_number, initials, label = [i.strip() for i in row]
        seg = segment_(sum, seg_number, initials, label)
        no_dubs[sum] = seg          
        #print(lijst[sum].seg_number)
        sum += 1 
    
    print('tellen')
    match_loop = 0
    length = len(no_dubs)
    rn = 0
    rc = 0
    nc = 0
    rr = 0
    cc = 0
    nn = 0
    while match_loop < length:
        a = no_dubs[match_loop].seg_number
        c = no_dubs[match_loop].initials
        e = no_dubs[match_loop].label
        for y in no_dubs:
            b = no_dubs[y].seg_number
            d = no_dubs[y].initials
            f = no_dubs[y].label
            #print(y)
            #print(no_dubs[y].seg_number, y)
            if a == b and c != d and 'rest' in f and 'needle' in e:
                rn += 1
            elif a == b and c != d and 'rest' in f and 'contraction' in e:
                rc += 1
            elif a == b and c != d and 'needle' in f and 'contraction' in e:
                nc += 1 
            elif a == b and c != d and 'rest' in f and 'rest' in e:
                rr += 1
            elif a == b and c != d and 'contraction' in f and 'contraction' in e:
                cc += 1
            elif a == b and c != d and 'needle' in f and 'needle' in e:
                nn += 1
            else:
                continue
        #match_loop += len(no_dubs)
        #no_dubs.pop(match_loop)
        match_loop += 1   
    if rr > 0:
        rr = rr/2
    if cc > 0:
        cc = cc/2
    if nn > 0:
        nn = nn/2
    print(rn,rc,nc,rr,cc,nn)