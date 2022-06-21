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
    fh = open('ai_needle_emg-master/analyse/categorised_list/triple_list_0900.txt').readlines()
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
        if no_dubs[match_loop].seg_number in str(my_list):
            print('eerste is dubbel',a)
            match_loop += 1
        else:
            for y in no_dubs:
                #print(y,'dit is y')
                b = no_dubs[y].seg_number
                d = no_dubs[y].initials
                f = no_dubs[y].label
                for z in no_dubs:
                    g = no_dubs[z].seg_number
                    h = no_dubs[z].initials
                    i = no_dubs[z].label
                        #print(no_dubs[z].seg_number,no_dubs[z].initials,no_dubs[z].label,'dit is nu y', no_dubs[y].seg_number,no_dubs[y].initials,no_dubs[y].label,'dit is nu x', no_dubs[match_loop].seg_number,no_dubs[match_loop].initials,no_dubs[match_loop].label) 
                    if a in b in g  and 'rest' in e and 'needle' in f and 'needle' in i and d not in h:
                        if a in str(my_list):
                            continue
                        else:
                            my_list.append(a)
                            seg = segment2(sum, no_dubs[match_loop].seg_number[1:], 1/3, 0, 2/3)
                            print('match')
                            tussen_lijst[sum] = seg
                            sum += 1
                            print(a,b,g)
                    elif a in b in g and 'rest' in e and 'contraction' in f and "contraction" in i and d not in h:
                        if a in str(my_list):
                            continue
                        else:
                            my_list.append(a)
                            seg = segment2(sum, no_dubs[match_loop].seg_number[1:], 1/3, 2/3, 0)
                            tussen_lijst[sum] = seg
                            sum += 1
                            print(a,b,g)
                    elif a in b in g and 'needle' in e and 'contraction' in f and "contraction" in i and d not in h:
                        if a in str(my_list):
                            continue
                        else:
                            my_list.append(a)
                            seg = segment2(sum, no_dubs[match_loop].seg_number[1:], 0, 2/3, 1/3)
                            tussen_lijst[sum] = seg
                            sum += 1
                            print(a,b,g)
                    elif a in b in g and 'contraction' in e and 'needle' in f and 'needle' in i and d not in h:
                        if a in str(my_list):
                            continue
                        else:
                            my_list.append(a)
                            seg = segment2(sum, no_dubs[match_loop].seg_number[1:], 0, 1/3, 2/3)
                            tussen_lijst[sum] = seg
                            sum += 1
                            print(a,b,g)
                    elif a in b in g  and 'needle' in e and 'contraction' in f and "rest" in i and d not in h:
                        if a in str(my_list):
                            continue
                        else:
                            my_list.append(a)
                            seg = segment2(sum, no_dubs[match_loop].seg_number[1:], 1/3, 1/3, 1/3)
                            tussen_lijst[sum] = seg
                            sum += 1
                            print(a,b,g)
                    elif a in b in g and 'needle' in e and 'rest' in f and "rest" in i and d not in h:
                        if a in str(my_list):
                            continue
                        else:
                            my_list.append(a)
                            print('match')
                            seg = segment2(sum, no_dubs[match_loop].seg_number[1:], 2/3, 0, 1/3)
                            tussen_lijst[sum] = seg
                            sum += 1
                            print(a,b,g)
                    elif a in b in g and 'rest' in e and 'rest' in f and "rest" in i and c not in d:
                        if a in str(my_list):
                                    continue
                        else:
                            my_list.append(a)
                            seg = segment2(sum, no_dubs[match_loop].seg_number[1:], 1, 0, 0)
                            tussen_lijst[sum] = seg
                            sum += 1
                            print(a,b,g)
                    elif a in b in g and 'contraction' in e and 'contraction' in f and 'contraction' in i and c not in d:
                        if a in str(my_list):
                            continue
                        else:
                            my_list.append(a)
                            seg = segment2(sum, no_dubs[match_loop].seg_number[1:], 0, 1, 0)
                            tussen_lijst[sum] = seg
                            sum += 1
                            print(a,b,g)
                    elif a in b in g and 'needle' in e and 'needle' in f and 'needle' in i and c not in d:
                        if a in str(my_list):
                            continue
                        else:
                            my_list.append(a)
                            seg = segment2(sum, no_dubs[match_loop].seg_number[1:], 0, 0, 1)
                            tussen_lijst[sum] = seg
                            sum += 1
                            print(a,b,g)
                    elif a in b in g and 'contraction' in e and 'rest' in f and 'rest' in i and d not in h:
                        if a in str(my_list):
                            continue
                        else:
                            my_list.append(a)
                            seg = segment2(sum, no_dubs[match_loop].seg_number[1:], 2/3, 1/3, 0)
                            tussen_lijst[sum] = seg
                            sum += 1
                            print(a,b,g)
            match_loop += 1   
    
    print(len(tussen_lijst))

    print('schrijven vier')
    x = 0
    file = open("ai_needle_emg-master/analyse/smoothed_labels/triple_0900_soft.txt","a") 
    while x<len(tussen_lijst):
        file.write(str(tussen_lijst[x].seg_number)+','+str(tussen_lijst[x].needle)+','+str(tussen_lijst[x].contraction)+','+str(tussen_lijst[x].rest)+'\n')
        x+=1
   
