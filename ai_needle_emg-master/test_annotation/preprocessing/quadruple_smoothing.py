"""
Pre-process data to use for training, this one prepares data for model where we look at three factors.
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
    fh = open('Correct_0-0299/quadruple_0_list.txt').readlines()
    sum = 0
    for line in fh:               
        row = line.split(',')
        seg_number, initials, label = [i.strip() for i in row]
        seg = segment_(sum, seg_number, initials, label)
        no_dubs[sum] = seg
        sum += 1 

    tussen_lijst = { }
    x = 0
    while x<len(no_dubs):
        if 'CV' in no_dubs[x].initials:
            a = 0.95/4
        elif 'DH' in no_dubs[x].initials:
            a = 0.8/4
        elif 'LW' in no_dubs[x].initials:
            a = 0.9/4
        else:
            a = 0.75/4
        seg = segment_(x, no_dubs[x].seg_number, a, no_dubs[x].label)
        #print(x, no_dubs[x].seg_number, a, no_dubs[x].label)
        no_dubs[x] = seg   
        x+=1
        
    match_loop = 0
    sum = 0
    counter = 0
    my_list = []
    while match_loop < len(no_dubs):
        print(match_loop)
        #print(match_loop,'we zijn hier')
        a = no_dubs[match_loop].seg_number
        c = no_dubs[match_loop].initials
        e = no_dubs[match_loop].label
        string = a
        if string in a:
            if no_dubs[match_loop].seg_number in str(my_list):
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
                                    if string in aa:
                                        if a in b in g in aa and 'contraction' in f in i in cc and 'needle' in e:
                                            if a in str(my_list):
                                                #rcn
                                                break
                                            else:
                                                my_list.append(a)
                                                value_1 = c * 0.58
                                                value_2 = (d+h+bb) * 0.58
                                                y = (1 - (value_1+value_2))/3
                                                seg = segment2(sum, no_dubs[match_loop].seg_number[1:], y, value_2+y,value_1+y)
                                                tussen_lijst[sum] = seg
                                                sum += 1
                                        if a in b in g in aa and 'needle' in f in i in cc and 'contraction' in e:
                                            if a in str(my_list):
                                                break
                                            else:
                                                my_list.append(a)
                                                value_1 = c * 0.58
                                                value_2 = (d+h+bb) * 0.58
                                                y = (1 - (value_1+value_2))/3
                                                seg = segment2(sum, no_dubs[match_loop].seg_number[1:],value_1+y,y, value_2+y)
                                                tussen_lijst[sum] = seg
                                                sum += 1
                                        elif a in b in g in aa and 'contraction' in f in i in cc and 'rest' in e:
                                            if a in str(my_list):
                                                break
                                            else:
                                                my_list.append(a)
                                                value_1 = c * 0.69
                                                value_2 = (d+h+bb) * 0.69
                                                y = (1 - (value_1+value_2))/3
                                                seg = segment2(sum, no_dubs[match_loop].seg_number[1:], value_1+y,value_2+y,y)
                                                tussen_lijst[sum] = seg
                                                sum += 1
                                        elif a in b in g in aa and 'rest' in f in i in cc and 'contraction' in e:
                                            if a in str(my_list):
                                                break
                                            else:
                                                my_list.append(a)
                                                value_1 = c * 0.69
                                                value_2 = (d+h+bb) * 0.69
                                                y = (1 - (value_1+value_2))/3
                                                seg = segment2(sum, no_dubs[match_loop].seg_number[1:], value_2+y,value_1+y,y)
                                                tussen_lijst[sum] = seg
                                                sum += 1
                                        elif a in b in g in aa and 'rest' in f in i in cc and 'needle' in e:
                                            if a in str(my_list):
                                                break
                                            else:
                                                my_list.append(a)
                                                value_1 = c * 0.54
                                                value_2 = (d+h+bb) * 0.54
                                                y = (1 - (value_1+value_2))/3
                                                seg = segment2(sum, no_dubs[match_loop].seg_number[1:], value_2+y,y, value_1+y)
                                                tussen_lijst[sum] = seg
                                                sum += 1
                                        elif a in b in g in aa and 'needle' in f in i in cc and 'rest' in e:
                                            if a in str(my_list):
                                                break
                                            else:
                                                my_list.append(a)
                                                value_1 = c * 0.54
                                                value_2 = (d+h+bb) * 0.54
                                                y = (1 - (value_1+value_2))/3
                                                seg = segment2(sum, no_dubs[match_loop].seg_number[1:], value_1+y,y,value_2+y)
                                                tussen_lijst[sum] = seg
                                                sum += 1
                                        elif a in b in g in aa and 'rest' in f in i and 'needle' in e in cc:
                                            if a in str(my_list):
                                                break
                                            else:
                                                my_list.append(a)
                                                value_1 = (c+bb) * 0.54
                                                value_2 = (d+h) * 0.54
                                                y = (1 - (value_1+value_2))/3
                                                seg = segment2(sum, no_dubs[match_loop].seg_number[1:], value_2+y,y,value_1+y)
                                                tussen_lijst[sum] = seg
                                                sum += 1
                                        elif a in b in g in aa and 'contraction' in f in i and 'needle' in e in cc:
                                            if a in str(my_list):
                                                break
                                            else:
                                                my_list.append(a)
                                                value_1 = (c+bb) * 0.58
                                                value_2 = (d+h) * 0.58
                                                y = (1 - (value_1+value_2))/3
                                                seg = segment2(sum, no_dubs[match_loop].seg_number[1:], y, value_2+y,value_1+y)
                                                tussen_lijst[sum] = seg
                                                sum += 1
                                        elif a in b in g in aa and 'contraction' in f in i and 'rest' in e in cc:
                                            if a in str(my_list):
                                                break
                                            else:
                                                my_list.append(a)
                                                value_1 = (c+bb) * 0.69
                                                value_2 = (d+h) * 0.69
                                                y = (1 - (value_1+value_2))/3
                                                seg = segment2(sum, no_dubs[match_loop].seg_number[1:], value_1+y,value_2+y,y)
                                                tussen_lijst[sum] = seg
                                                sum += 1
                                        elif a in b in g in aa and 'contraction' in f in i and 'rest' in e and 'needle' in cc:
                                            if a in str(my_list):
                                                break
                                            else:
                                                my_list.append(a)
                                                value_1 = c * 0.46
                                                value_2 = (d+h) * 0.46
                                                value_3 = bb * 0.46
                                                y = (1 - (value_1+value_2+value_3))/3
                                                seg = segment2(sum, no_dubs[match_loop].seg_number[1:], value_1+y,value_2+y,value_3+y)
                                                tussen_lijst[sum] = seg
                                                sum += 1
                                        elif a in b in g in aa and 'rest' in f in i and 'contraction' in e and 'needle' in cc:
                                            if a in str(my_list):
                                                break
                                            else:
                                                my_list.append(a)
                                                value_1 = c * 0.46
                                                value_2 = (d+h) * 0.46
                                                value_3 = bb * 0.46
                                                y = (1 - (value_1+value_2+value_3))/3
                                                seg = segment2(sum, no_dubs[match_loop].seg_number[1:], value_2+y,value_1+y,value_3+y)
                                                tussen_lijst[sum] = seg
                                                sum += 1
                                        elif a in b in g in aa and 'needle' in f in i and 'contraction' in e and 'rest' in cc:
                                            if a in str(my_list):
                                                break
                                            else:
                                                my_list.append(a)
                                                value_1 = c * 0.46
                                                value_2 = (d+h) * 0.46
                                                value_3 = bb * 0.46
                                                y = (1 - (value_1+value_2+value_3))/3
                                                seg = segment2(sum, no_dubs[match_loop].seg_number[1:], value_3+y,value_1+y,value_2+y)
                                                tussen_lijst[sum] = seg
                                                sum += 1
                                        elif a in b in g in aa and 'contraction' in f in i in cc in e:
                                            if a in str(my_list):
                                                break
                                            else:
                                                my_list.append(a)
                                                value_1 = (c+d+h+bb) * 0.86
                                                y = (1 - (value_1))/3
                                                seg = segment2(sum, no_dubs[match_loop].seg_number[1:], y,value_1+y,y)
                                                tussen_lijst[sum] = seg
                                                sum += 1
                                        elif a in b in g in aa and 'rest' in f in i in cc in e:
                                            if a in str(my_list):
                                                break
                                            else:
                                                my_list.append(a)
                                                value_1 = (c+d+h+bb) * 0.8
                                                y = (1 - (value_1))/3
                                                seg = segment2(sum, no_dubs[match_loop].seg_number[1:], value_1+y,y,y)
                                                tussen_lijst[sum] = seg
                                                sum += 1
                                        elif a in b in g in aa and 'needle' in f in i in cc in e:
                                            if a in str(my_list):
                                                break
                                            else:
                                                my_list.append(a)
                                                value_1 = (c+d+h+bb) * 0.67
                                                y = (1 - (value_1))/3
                                                seg = segment2(sum, no_dubs[match_loop].seg_number[1:], y,y,value_1+y)
                                                tussen_lijst[sum] = seg
                                                sum += 1
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
    file = open("ai_needle_emg-master/analyse/smoothing/quadruple_smoothed.txt","a") 
    while x<len(tussen_lijst):
        file.write(str(tussen_lijst[x].seg_number)+','+str(tussen_lijst[x].needle)+','+str(tussen_lijst[x].contraction)+','+str(tussen_lijst[x].rest)+'\n')
        x+=1
   
