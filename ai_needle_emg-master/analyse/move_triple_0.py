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
    counter = 0
    sum=0
    while match_loop < len(no_dubs):
        a = no_dubs[match_loop].seg_number
        c = no_dubs[match_loop].initials
        e = no_dubs[match_loop].label
        if '0136_R_ext_hal_1_1_11' in a:
            seg = segment_(sum,a,c,e)
            tussen_lijst[sum]=seg
            sum+=1
            counter+=1
            match_loop+=1
            '''
        elif '0136_R_ext_hal_1_1_47' in a:
            seg = segment_(sum,a,c,e)
            tussen_lijst[sum]=seg
            sum+=1
            counter+=1
            match_loop+=1
        elif '0136_R_ext_hal_1_1_48' in a:
            seg = segment_(sum,a,c,e)
            tussen_lijst[sum]=seg
            sum+=1
            counter+=1
            match_loop+=1
        elif '0136_R_ext_hal_1_1_49' in a:
            seg = segment_(sum,a,c,e)
            tussen_lijst[sum]=seg
            sum+=1
            counter+=1
            match_loop+=1
        elif '0136_R_ext_hal_1_1_45' in a:
            seg = segment_(sum,a,c,e)
            tussen_lijst[sum]=seg
            sum+=1
            counter+=1
            match_loop+=1
            '''
        else:
            match_loop+=1
    print(counter)

    print('twee hernoemen')                                                  
    twee_dub_correct = { }
    key_value = 0
    for key, value in tussen_lijst.items():
        if value not in twee_dub_correct.values():
            twee_dub_correct[key] = value
            twee_dub_correct[key_value] = twee_dub_correct.pop(key)
            #print(drie_dub_correct[key_value].seg_number)
            key_value += 1  

    print('schrijven')
    x = 0
    file = open("ai_needle_emg-master/analyse/quadruple/quadruple_0_single/0136/0136(1)/0136(11).txt","a") 
    while x<len(twee_dub_correct):
        file.write(str(twee_dub_correct[x].seg_number)+','+str(twee_dub_correct[x].initials)+','+str(twee_dub_correct[x].label)+'\n')
        x+=1