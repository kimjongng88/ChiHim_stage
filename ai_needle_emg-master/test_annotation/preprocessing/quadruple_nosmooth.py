"""
Pre-process data to use for training, this one prepares data for model where we look at amount of reviewers.
"""
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
    fh = open('ai_needle_emg-master/analyse/Correct_0-0299/quadruple_0_list.txt').readlines()
    sum = 0
    for line in fh:               
        row = line.split(',')
        seg_number, initials, label = [i.strip() for i in row]
        seg = segment_(sum, seg_number, initials, label)
        no_dubs[sum] = seg
        sum += 1 

    tussen_lijst = { }
    match_loop = 0
    sum = 0
    counter = 0
    my_list = []
    while match_loop < len(no_dubs):
        #print(match_loop,'we zijn hier')
        a = no_dubs[match_loop].seg_number
        c = no_dubs[match_loop].initials
        e = no_dubs[match_loop].label
        print(a,match_loop)
        #string = '0107_R_gas_lat_1_1_21'
        string = a
        if string in a:
            if no_dubs[match_loop].seg_number in str(my_list):
                print('eerste is dubbel',a)
                match_loop += 1
            else:
                for y in no_dubs:
                    b = no_dubs[y].seg_number
                    d = no_dubs[y].initials
                    f = no_dubs[y].label
                    if string in b:
                        for z in no_dubs:
                            g = no_dubs[z].seg_number
                            h = no_dubs[z].initials
                            i = no_dubs[z].label
                            if string in g:
                                for da in no_dubs:
                                    aa = no_dubs[da].seg_number
                                    bb = no_dubs[da].initials
                                    cc = no_dubs[da].label
                                    if c not in d and c not in h and c not in bb and d not in h and d not in bb and h not in bb and string in aa:
                                        if a in b in g in aa and 'contraction' in f in i in cc and 'needle' in e:
                                            if a in str(my_list):
                                                break
                                            else:
                                                my_list.append(a)
                                                seg = segment2(sum, no_dubs[match_loop].seg_number[1:], 0, 3/4, 1/4)
                                                tussen_lijst[sum] = seg
                                                sum += 1
                                        if a in b in g in aa and 'needle' in f in i in cc and 'contraction' in e:
                                            if a in str(my_list):
                                                break
                                            else:
                                                my_list.append(a)
                                                seg = segment2(sum, no_dubs[match_loop].seg_number[1:], 0, 1/4, 3/4)
                                                tussen_lijst[sum] = seg
                                                sum += 1
                                        elif a in b in g in aa and 'contraction' in f in i in cc and 'rest' in e:
                                            if a in str(my_list):
                                                break
                                            else:
                                                my_list.append(a)
                                                seg = segment2(sum, no_dubs[match_loop].seg_number[1:], 1/4, 3/4, 0)
                                                tussen_lijst[sum] = seg
                                                sum += 1
                                        elif a in b in g in aa and 'rest' in f in i in cc and 'contraction' in e:
                                            if a in str(my_list):
                                                break
                                            else:
                                                my_list.append(a)
                                                seg = segment2(sum, no_dubs[match_loop].seg_number[1:], 3/4, 1/4, 0)
                                                tussen_lijst[sum] = seg
                                                sum += 1
                                        elif a in b in g in aa and 'rest' in f in i in cc and 'needle' in e:
                                            if a in str(my_list):
                                                break
                                            else:
                                                my_list.append(a)
                                                seg = segment2(sum, no_dubs[match_loop].seg_number[1:], 3/4, 0, 1/4)
                                                tussen_lijst[sum] = seg
                                                sum += 1
                                        elif a in b in g in aa and 'needle' in f in i in cc and 'rest' in e:
                                            if a in str(my_list):
                                                break
                                            else:
                                                my_list.append(a)
                                                seg = segment2(sum, no_dubs[match_loop].seg_number[1:], 1/4, 0, 3/4)
                                                tussen_lijst[sum] = seg
                                                sum += 1
                                        elif a in b in g in aa and 'rest' in f in i and 'needle' in e in cc:
                                            if a in str(my_list):
                                                break
                                            else:
                                                my_list.append(a)
                                                seg = segment2(sum, no_dubs[match_loop].seg_number[1:], 0.5, 0, 0.5)
                                                tussen_lijst[sum] = seg
                                                sum += 1
                                        elif a in b in g in aa and 'contraction' in f in i and 'needle' in e in cc:
                                            if a in str(my_list):
                                                break
                                            else:
                                                my_list.append(a)
                                                seg = segment2(sum, no_dubs[match_loop].seg_number[1:], 0, 0.5, 0.5)
                                                tussen_lijst[sum] = seg
                                                sum += 1
                                        elif a in b in g in aa and 'contraction' in f in i and 'rest' in e in cc:
                                            if a in str(my_list):
                                                break
                                            else:
                                                my_list.append(a)
                                                seg = segment2(sum, no_dubs[match_loop].seg_number[1:], 0.5, 0.5, 0)
                                                tussen_lijst[sum] = seg
                                                sum += 1
                                        elif a in b in g in aa and 'contraction' in f in i and 'rest' in e and 'needle' in cc:
                                            if a in str(my_list):
                                                break
                                            else:
                                                my_list.append(a)
                                                seg = segment2(sum, no_dubs[match_loop].seg_number[1:], 0.25, 0.5, 0.25)
                                                tussen_lijst[sum] = seg
                                                sum += 1
                                        elif a in b in g in aa and 'rest' in f in i and 'contraction' in e and 'needle' in cc:
                                            if a in str(my_list):
                                                break
                                            else:
                                                my_list.append(a)
                                                seg = segment2(sum, no_dubs[match_loop].seg_number[1:], 0.5, 0.25, 0.25)
                                                tussen_lijst[sum] = seg
                                                sum += 1
                                        elif a in b in g in aa and 'needle' in f in i and 'contraction' in e and 'rest' in cc:
                                            if a in str(my_list):
                                                break
                                            else:
                                                my_list.append(a)
                                                seg = segment2(sum, no_dubs[match_loop].seg_number[1:], 0.25, 0.25, 0.5)
                                                tussen_lijst[sum] = seg
                                                sum += 1
                            
                                        elif a in b in g in aa and 'contraction' in f in i in cc in e:
                                            if a in str(my_list):
                                                break
                                            else:
                                                my_list.append(a)
                                                seg = segment2(sum, no_dubs[match_loop].seg_number[1:], 0, 1, 0)
                                                tussen_lijst[sum] = seg
                                                sum += 1
                                                print('allemaal hetzelfde')
                                        elif a in b in g in aa and 'rest' in f in i in cc in e:
                                            if a in str(my_list):
                                                break
                                            else:
                                                my_list.append(a)
                                                seg = segment2(sum, no_dubs[match_loop].seg_number[1:], 1, 0, 0)
                                                tussen_lijst[sum] = seg
                                                sum += 1
                                                print('allemaal hetzelfde')
                                        elif a in b in g in aa and 'needle' in f in i in cc in e:
                                            if a in str(my_list):
                                                break
                                            else:
                                                my_list.append(a)
                                                seg = segment2(sum, no_dubs[match_loop].seg_number[1:], 0, 0, 1)
                                                tussen_lijst[sum] = seg
                                                sum += 1
                                                print('allemaal hetzelfde')
                                        else:
                                            continue
                                    else:
                                        continue
                            else:
                                continue
                    else:
                        continue
                match_loop += 1   
        else:
            print('nada')
            match_loop += 1
    print(len(tussen_lijst))

    print('schrijven vier')
    x = 0
    file = open("ai_needle_emg-master/analyse/quadruple/quadruple_soft.txt","a") 
    while x<len(tussen_lijst):
        file.write(str(tussen_lijst[x].seg_number)+','+str(tussen_lijst[x].needle)+','+str(tussen_lijst[x].contraction)+','+str(tussen_lijst[x].rest)+'\n')
        x+=1
   
