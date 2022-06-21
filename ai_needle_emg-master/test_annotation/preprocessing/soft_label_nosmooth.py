"""
Pre-process data to use for training, this one prepares data for model where we look at amount of reviewers.
"""
import json
from soft_label_class import segment2

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
    fh = open('ai_needle_emg-master/analyse/categorised_list/double_list_0900.txt').readlines()
    sum = 0
    for line in fh:               
        row = line.split(',')
        seg_number, initials, label = [i.strip() for i in row]
        seg = segment_(sum, seg_number, initials, label)
        no_dubs[sum] = seg
        sum += 1 

    print('tellen')
    tussen_lijst = { }
    dubbel = [ ]
    match_loop = 0
    sum = 0
    length = len(no_dubs)
    counter = 0
    while match_loop < length:
        #print(match_loop)
        a = no_dubs[match_loop].seg_number
        c = no_dubs[match_loop].initials
        e = no_dubs[match_loop].label
        for y in no_dubs:
            b = no_dubs[y].seg_number
            d = no_dubs[y].initials
            f = no_dubs[y].label
            if a in b and c not in d and 'rest' in f and 'needle' in e and 'aaa' not in b and 'hjb' not in a:
                seg = segment2(sum, no_dubs[match_loop].seg_number[1:], 0.5, 0, 0.5)
                a = 'aaa'
                no_dubs[y].seg_number = 'hjb'
                tussen_lijst[sum] = seg
                sum += 1
            elif a in b and c not in d and 'rest' in f and 'contraction' in e and 'aaa' not in b and 'hjb' not in a:
                seg = segment2(sum, no_dubs[match_loop].seg_number[1:], 0.5, 0.5, 0)
                a = 'aaa'
                no_dubs[y].seg_number = 'hjb'
                tussen_lijst[sum] = seg
                sum += 1
            elif a in b and c not in d and 'needle' in f and 'contraction' in e and 'aaa' not in b and 'hjb' not in a:
                seg = segment2(sum, no_dubs[match_loop].seg_number[1:], 0, 0.5, 0.5)
                a = 'aaa'
                no_dubs[y].seg_number = 'hjb'
                tussen_lijst[sum] = seg
                sum += 1
            elif a in b and c not in d and 'rest' in f and 'rest' in e and 'aaa' not in b and 'hjb' not in a:
                if counter == 1:
                    seg = segment2(sum, no_dubs[match_loop].seg_number[1:], 1, 0, 0)
                    a = 'aaa'
                    no_dubs[y].seg_number = 'hjb'
                    tussen_lijst[sum] = seg
                    sum += 1
                    counter = 0
                else:
                    counter += 1
            elif a in b and c not in d and 'contraction' in f and 'contraction' in e and 'aaa' not in b and 'hjb' not in a:
                if counter == 1:
                    seg = segment2(sum, no_dubs[match_loop].seg_number[1:], 0, 1, 0)
                    a = 'aaa'
                    no_dubs[y].seg_number = 'hjb'
                    tussen_lijst[sum] = seg
                    sum += 1
                    counter = 0
                else:
                    counter += 1
            elif a in b and c not in d and 'needle' in f and 'needle' in e and 'aaa' not in b and 'hjb' not in a:
                if counter == 1:
                    seg = segment2(sum, no_dubs[match_loop].seg_number[1:], 0, 0, 1)
                    a = 'aaa'
                    no_dubs[y].seg_number = 'hjb'
                    tussen_lijst[sum] = seg
                    sum += 1
                    counter = 0
                else:
                    counter += 1
        match_loop += 1   

    print('schrijven vier')
    x = 0
    file = open("ai_needle_emg-master/analyse/smoothed_labels/0900_soft.txt","w") 
    while x<len(tussen_lijst):
        file.write(str(tussen_lijst[x].seg_number)+','+str(tussen_lijst[x].needle)+','+str(tussen_lijst[x].contraction)+','+str(tussen_lijst[x].rest)+'\n')
        x+=1


    
    
