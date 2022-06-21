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
    fh = open('Correct_0-0299/triple_0220_list.txt').readlines()
    sum = 0
    for line in fh:               
        row = line.split(',')
        seg_number, initials, label = [i.strip() for i in row]
        seg = segment_(sum, seg_number, initials, label)
        no_dubs[sum] = seg          
        #print(lijst[sum].seg_number)
        sum += 1 
    
    tussen_lijst = { }
    x = 0
    while x<len(no_dubs):
        if 'CV' in no_dubs[x].initials:
            a = 0.95/3
        elif 'DH' in no_dubs[x].initials:
            a = 0.8/3
        elif 'LW' in no_dubs[x].initials:
            a = 0.9/3
        else:
            a = 0.75/3
        seg = segment_(x, no_dubs[x].seg_number, a, no_dubs[x].label)
        #print(x, no_dubs[x].seg_number, a, no_dubs[x].label)
        no_dubs[x] = seg   
        x+=1

    match_loop = 0
    sum = 0
    double_list = [ ]
    while match_loop < len(no_dubs):
        print(match_loop)
        a = no_dubs[match_loop].seg_number
        c = no_dubs[match_loop].initials
        e = no_dubs[match_loop].label
        naam = a
        if a in str(double_list):
            match_loop += 1
        else:
            for y in no_dubs:
                b = no_dubs[y].seg_number
                d = no_dubs[y].initials
                f = no_dubs[y].label
                if naam in b:
                    for z in no_dubs:
                        g = no_dubs[z].seg_number
                        h = no_dubs[z].initials
                        i = no_dubs[z].label
                        if naam in g:
                        #print(no_dubs[z].seg_number,no_dubs[z].initials,no_dubs[z].label,'dit is nu y', no_dubs[y].seg_number,no_dubs[y].initials,no_dubs[y].label,'dit is nu x', no_dubs[match_loop].seg_number,no_dubs[match_loop].initials,no_dubs[match_loop].label) 
                            if a in b in g  and 'rest' in e and 'needle' in f and 'needle' in i:
                                if a in str(double_list):
                                    break
                                else:
                                    double_list.append(a)
                                    value_1 = c * 0.54
                                    value_2 = (d + h) * 0.54
                                    y = (1 - (value_1+value_2))/3
                                    seg = segment2(sum, no_dubs[match_loop].seg_number[1:], value_1+y, y, value_2+y)
                                    #seg = {'seg_number':a,  'needle':value_2+y, 'contraction':y,'rest':value_1+y} 
                                    tussen_lijst[sum] = seg
                                    sum += 1
                            elif a in b in g and 'rest' in e and 'contraction' in f and "contraction" in i:
                                if a in str(double_list):
                                    break
                                else:   
                                    double_list.append(a)
                                    value_1 = c * 0.69
                                    value_2 = (d + h) * 0.69
                                    y = (1 - (value_1+value_2))/3
                                    seg = segment2(sum, no_dubs[match_loop].seg_number[1:], value_1+y, value_2+y, y)
                                    #seg = {'seg_number':a,  'needle':y, 'contraction':value_2+y,'rest':value_1+y,} 
                                    tussen_lijst[sum] = seg
                                    sum += 1
                            elif a in b in g and 'needle' in e and 'contraction' in f and "contraction" in i:
                                if a in str(double_list):
                                    break
                                else:   
                                    double_list.append(a)
                                    value_1 = c * 0.58
                                    value_2 = (d + h) * 0.58
                                    y = (1 - (value_1+value_2))/3
                                    seg = segment2(sum, no_dubs[match_loop].seg_number[1:], y, value_2+y, value_1+y)
                                    #seg = {'seg_number':a, 'needle': value_1 + y, 'contraction': value_2 + y, 'rest': y} 
                                    tussen_lijst[sum] = seg
                                    print(tussen_lijst)
                                    sum += 1
                            elif a in b in g and 'contraction' in e and 'needle' in f and 'needle' in i:
                                if a in str(double_list):
                                    break
                                else:   
                                    double_list.append(a)
                                    value_1 = c * 0.58
                                    value_2 = (d + h) * 0.58
                                    y = (1 - (value_1+value_2))/3
                                    seg = segment2(sum, no_dubs[match_loop].seg_number[1:], y, value_1+y, value_2+y)
                                    #seg = {'seg_number':a, 'needle': value_2 + y, 'contraction': value_1 + y, 'rest': y} 
                                    tussen_lijst[sum] = seg
                                    print(tussen_lijst)
                                    sum += 1
                            elif a in b in g  and 'needle' in e and 'contraction' in f and "rest" in i:
                                if a in str(double_list):
                                    break
                                else:   
                                    double_list.append(a)
                                    value_1 = c * 0.46
                                    value_2 = d * 0.46
                                    value_3 = h * 0.46
                                    y = (1 - (value_1+value_2+value_3))/3
                                    seg = segment2(sum, no_dubs[match_loop].seg_number[1:], value_1, value_2, value_3)
                                    #seg = {'seg_number':a, 'needle':value_3, 'contraction':value_2, 'rest':value_1} 
                                    tussen_lijst[sum] = seg
                                    sum += 1
                            elif a in b in g and 'needle' in e and 'rest' in f and "rest" in i:
                                if a in str(double_list):
                                    break
                                else:   
                                    double_list.append(a)
                                    value_1 = c * 0.54
                                    value_2 = (d + h) * 0.54
                                    y = (1 - (value_1+value_2))/3
                                    seg = segment2(sum, no_dubs[match_loop].seg_number[1:], value_1+y, y, value_2+y)
                                    #seg = {'seg_number':a, 'needle':value_2+y , 'contraction':y,'rest':value_1+y} 
                                    tussen_lijst[sum] = seg
                                    sum += 1
                            elif a in b in g and 'rest' in e and 'rest' in f and "rest" in i:
                                if a in str(double_list):
                                    break
                                else:   
                                    double_list.append(a)
                                    value_1 = (c * 0.8)
                                    value_2 = d * 0.8
                                    value_3 = h * 0.8
                                    y = (1 - (value_1*3))/3
                                    seg = segment2(sum, no_dubs[match_loop].seg_number[1:], value_1+value_2+value_3+y, y, y)
                                    #seg = {'seg_number':a, 'needle':y, 'contraction':y, 'rest':value_1+value_2+value_3+y} 
                                    tussen_lijst[sum] = seg
                                    sum += 1
                            elif a in b in g and 'contraction' in e and 'contraction' in f and 'contraction' in i:
                                if a in str(double_list):
                                    break
                                else:   
                                    double_list.append(a)
                                    value_1 = (c * 0.86)
                                    value_2 = d * 0.86
                                    value_3 = h * 0.86
                                    y = (1 - (value_1*3))/3
                                    seg = segment2(sum, no_dubs[match_loop].seg_number[1:], y, value_1+value_2+value_3+y, y)
                                    #seg = {'seg_number':a, 'needle':y, 'contraction':value_1+value_2+value_3+y, 'rest':y} 
                                    tussen_lijst[sum] = seg
                                    sum += 1
                            elif a in b in g and 'needle' in e and 'needle' in f and 'needle' in i:
                                if a in str(double_list):
                                    break
                                else:   
                                    double_list.append(a)
                                    value_1 = (c * 0.67)
                                    value_2 = c * 0.67
                                    value_3 = c * 0.67
                                    y = (1 - (value_1*3))/3
                                    seg = segment2(sum, no_dubs[match_loop].seg_number[1:],y , y, value_1+value_2+value_3+y)
                                    #seg = {'seg_number':a, 'needle':value_1+value_2+value_3+y, 'contraction':y, 'rest':y} 
                                    tussen_lijst[sum] = seg
                                    sum += 1
                            elif a in b in g and 'contraction' in e and 'rest' in f and 'rest' in i:
                                if a in str(double_list):
                                    break
                                else:   
                                    double_list.append(a)
                                    value_1 = c * 0.69
                                    value_2 = (d + h) * 0.69
                                    y = (1 - (value_1+value_2))/3
                                    seg = segment2(sum, no_dubs[match_loop].seg_number[1:], value_2+y, value_1+y, y)
                                    #seg = {'seg_number':a,  'needle':y, 'contraction':value_1+y,'rest':value_2+y} 
                                    tussen_lijst[sum] = seg
                                    sum += 1
                            else:
                                #print('vals') 
                                continue
                        else:
                            continue
                else:
                    continue
            match_loop += 1   

    print('schrijven vier')
    x = 0
    file = open("ai_needle_emg-master/analyse/smoothing/triple_smoothed.txt","a") 
    while x<len(tussen_lijst):
        file.write(str(tussen_lijst[x].seg_number)+','+str(tussen_lijst[x].needle)+','+str(tussen_lijst[x].contraction)+','+str(tussen_lijst[x].rest)+'\n')
        x+=1
   
    


