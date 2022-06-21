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
    lijst = {}
    fh = open('ai_needle_emg-master/analyse/Correct_0-0299/0_list.txt').readlines()
    #fh = open('ai_needle_emg-master/analyse/lijst.txt').readlines()
    sum = 0
    for line in fh:               
        row = line.split(',')
        seg_number, initials, label = [i.strip() for i in row]
        seg = segment_(sum, seg_number, initials, label)
        lijst[sum] = seg          
        #print(lijst[sum].seg_number)
        sum += 1 
    x = 0   
    match_number = 0
    match = {}
    while x < sum:
        for y in lijst:
            if lijst[x].seg_number == lijst[y].seg_number and lijst[x].initials != lijst[y].initials:
                #print(lijst[x].seg_number,lijst[y].seg_number,lijst[x].initials,lijst[y].initials + "   match") 
                match[match_number] = lijst[x] 
                match_number += 1
            else:
                #print(lijst[x].seg_number,lijst[y].seg_number + '   geen match')
                continue         
        x += 1
    
    print('dubbel check')
    no_dubs = {}
    key_value = 0
    for key, value in match.items():
        if value not in no_dubs.values():
            no_dubs[key] = value
            no_dubs[key_value] = no_dubs.pop(key)
            #no_dubs[key].sum = key_value
            key_value += 1            
            #print(no_dubs[key].sum)
            #print(no_dubs[key].seg_number, no_dubs[key].initials) 
    #print(no_dubs)

    print('drie dubbel')
    drie_dub = { }
    vier_dub = { }
    x = 0
    counter = 0
    sum = 0
    while x < len(no_dubs):
        for y in no_dubs:
            if no_dubs[x].seg_number == no_dubs[y].seg_number:
                if counter == 2:
                    seg = segment_(no_dubs[x].sum, no_dubs[x].seg_number, no_dubs[x].initials, no_dubs[x].label)
                    drie_dub[sum] = seg 
                    sum += 1
                    #print('drie', no_dubs[x].seg_number, no_dubs[x].initials)
                    counter = 0 
                    break
                else:
                    counter += 1
                    #print(counter,'hier counter', no_dubs[x].seg_number, no_dubs[x].initials)
            else: 
                continue
        x += 1
        counter = 0

    print('check vier')
    x = 0
    counter = 0
    sum = 0
    a = len(drie_dub)
    while x < a:
        for y in drie_dub:
            if drie_dub[x].seg_number == drie_dub[y].seg_number:
                if counter == 3:
                    seg = segment_(drie_dub[x].sum, drie_dub[x].seg_number, drie_dub[x].initials, drie_dub[x].label)
                    vier_dub[sum] = seg 
                    sum += 1
                    #print('vier', no_dubs[x].seg_number)
                    counter = 0
                else:
                    counter += 1
                    #print(counter,'hier counter', no_dubs[x].seg_number, no_dubs[x].initials)
            else: 
                continue
        x += 1
        counter = 0
  
    print('pop twee')
    x = 0
    a= len(no_dubs)
    while x < a:
        for y in vier_dub:
            if no_dubs[x].seg_number == vier_dub[y].seg_number:
                no_dubs.pop(x)
                break
            else:
                continue
        x += 1

    print('twee correct')
    twee_dub_tussenstap = { }
    key_value = 0
    for key, value in no_dubs.items():
        if value not in twee_dub_tussenstap.values():
            twee_dub_tussenstap[key] = value
            twee_dub_tussenstap[key_value] = twee_dub_tussenstap.pop(key)
            #print(drie_dub_correct[key_value].seg_number)
            key_value += 1 

    print('pop drie')
    x = 0
    a = len(drie_dub)
    while x < a:
        for y in vier_dub:
            if drie_dub[x].seg_number == vier_dub[y].seg_number:
                drie_dub.pop(x)
                break
            else:
                continue
        x += 1

    print('drie hernoemen')
    drie_dub_correct = { }
    key_value = 0
    for key, value in drie_dub.items():
        if value not in drie_dub_correct.values():
            drie_dub_correct[key] = value
            drie_dub_correct[key_value] = drie_dub_correct.pop(key)
            #print(drie_dub_correct[key_value].seg_number)
            key_value += 1  

    print('pop twee correct')
    x = 0
    while x<len(twee_dub_tussenstap):
        for y in drie_dub_correct:
            if twee_dub_tussenstap[x].seg_number == drie_dub_correct[y].seg_number:
                twee_dub_tussenstap.pop(x)
                break
            else:
                continue
        x += 1

    print('twee hernoemen')                                                  
    twee_dub_correct = { }
    key_value = 0
    for key, value in twee_dub_tussenstap.items():
        if value not in twee_dub_correct.values():
            twee_dub_correct[key] = value
            twee_dub_correct[key_value] = twee_dub_correct.pop(key)
            #print(drie_dub_correct[key_value].seg_number)
            key_value += 1  

    #print(twee_dub_correct)
    print(len(twee_dub_correct))
    print(len(drie_dub_correct))
    print(len(vier_dub))

    print('schrijven')
    x = 0
    file = open("ai_needle_emg-master/analyse/Correct_0-0299/double_0_list.txt","w") 
    while x<len(twee_dub_correct):
        file.write(str(twee_dub_correct[x].seg_number)+','+str(twee_dub_correct[x].initials)+','+str(twee_dub_correct[x].label)+'\n')
        x+=1

    print('schrijven drie')
    x = 0
    file = open("ai_needle_emg-master/analyse/Correct_0-0299/triple_0_list.txt","w") 
    while x<len(drie_dub_correct):
        file.write(str(drie_dub_correct[x].seg_number)+','+str(drie_dub_correct[x].initials)+','+str(drie_dub_correct[x].label)+'\n')
        x+=1

    print('schrijven vier')
    x = 0
    file = open("ai_needle_emg-master/analyse/Correct_0-0299/quadruple_0_list.txt","w") 
    while x<len(vier_dub):
        file.write(str(vier_dub[x].seg_number)+','+str(vier_dub[x].initials)+','+str(vier_dub[x].label)+'\n')
        x+=1
'''
    print('tellen')
    match_loop = 0
    length = len(no_dubs)
    rn = 0
    rc = 0
    re = 0
    nc = 0
    ne = 0
    ce = 0
    rr = 0
    cc = 0
    nn = 0
    ee = 0
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
            elif a == b and c != d and 'rest' in f and 'empty' in e:
                re += 1
            elif a == b and c != d and 'needle' in f and 'contraction' in e:
                nc += 1 
            elif a == b and c != d and 'needle' in f and 'empty' in e:
                ne += 1 
            elif a == b and c != d and 'contraction' in f and 'empty' in e:
                ce += 1
            elif a == b and c != d and 'rest' in f and 'rest' in e:
                rr += 1
            elif a == b and c != d and 'contraction' in f and 'contraction' in e:
                cc += 1
            elif a == b and c != d and 'needle' in f and 'needle' in e:
                nn += 1
            elif a == b and c != d and 'empty' in f and 'empty' in e:
                ee += 1
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
    if ee > 0:
        ee = ee/2
    print(rn,rc,re,nc,ne,ce,rr,cc,nn,ee)
'''
