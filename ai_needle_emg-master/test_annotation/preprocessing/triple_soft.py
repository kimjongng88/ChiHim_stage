"""
This script defines all default values for the models. 
Included values:
Filepaths, data input parameters, flags and model parameters
"""
from matplotlib.pyplot import rc


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
    fh = open('ai_needle_emg-master/analyse/categorised_list/triple_list_0-0299.txt').readlines()
    #fh = open('ai_needle_emg-master/analyse/test_triple_list.txt').readlines()
    sum = 0
    for line in fh:               
        row = line.split(',')
        seg_number, initials, label = [i.strip() for i in row]
        seg = segment_(sum, seg_number, initials, label)
        no_dubs[sum] = seg
        sum += 1 


    print('tellen')
    match_loop = 0
    length = len(no_dubs)
    nnn = 0
    ccc = 0
    rrr = 0
    cnn = 0
    rnn = 0
    ncc = 0
    nrr = 0
    ncr = 0
    rcc = 0
    crr = 0
    vals = 0
    while match_loop < length:
        a = no_dubs[match_loop].seg_number
        c = no_dubs[match_loop].initials
        e = no_dubs[match_loop].label
        for y in no_dubs:
            b = no_dubs[y].seg_number
            d = no_dubs[y].initials
            f = no_dubs[y].label
            for z in no_dubs:
                g = no_dubs[z].seg_number
                h = no_dubs[z].initials
                i = no_dubs[z].label
                #print(no_dubs[z].seg_number,no_dubs[z].initials,no_dubs[z].label,'dit is nu y', no_dubs[y].seg_number,no_dubs[y].initials,no_dubs[y].label,'dit is nu x', no_dubs[match_loop].seg_number,no_dubs[match_loop].initials,no_dubs[match_loop].label) 
                if a in b in g and c != d and c != h and d!= h and 'rest' in e and 'needle' in f and 'needle' in i:
                    rnn += 1
                    print(no_dubs[z].seg_number,no_dubs[z].initials,no_dubs[z].label,'dit is nu y', no_dubs[y].seg_number,no_dubs[y].initials,no_dubs[y].label,'dit is nu x', no_dubs[match_loop].seg_number,no_dubs[match_loop].initials,no_dubs[match_loop].label) 
                elif a in b in g and c != d and c != h and d!= h and 'rest' in e and 'contraction' in f and "contraction" in i:
                    rcc += 1
                    print(no_dubs[z].seg_number,no_dubs[z].initials,no_dubs[z].label,'dit is nu y', no_dubs[y].seg_number,no_dubs[y].initials,no_dubs[y].label,'dit is nu x', no_dubs[match_loop].seg_number,no_dubs[match_loop].initials,no_dubs[match_loop].label) 
                elif a in b in g and c != d and c != h and d!= h and 'needle' in e and 'contraction' in f and "contraction" in i:
                    ncc += 1 
                elif a in b in g and c != d and c != h and d!= h and 'contraction' in e and 'needle' in f and 'needle' in i:
                    cnn += 1
                    print(no_dubs[z].seg_number,no_dubs[z].initials,no_dubs[z].label,'dit is nu y', no_dubs[y].seg_number,no_dubs[y].initials,no_dubs[y].label,'dit is nu x', no_dubs[match_loop].seg_number,no_dubs[match_loop].initials,no_dubs[match_loop].label) 
                elif a in b in g and c != d and c != h and d!=h and 'needle' in e and 'contraction' in f and "rest" in i:
                    ncr += 1
                elif a in b in g and c != d and c != h and d!=h and 'needle' in e and 'rest' in f and "rest" in i:
                    nrr += 1 
                elif a in b in g and c != d and c != h and d!=h and 'rest' in e and 'rest' in f and "rest" in i:
                    rrr += 1
                elif a in b in g and c != d and c != h and d!=h and 'contraction' in e and 'contraction' in f and 'contraction' in i:
                    ccc += 1
                elif a in b in g and c != d and c != h and d!=h and 'needle' in e and 'needle' in f and 'needle' in i:
                    nnn += 1
                elif a in b in g and c != d and c != h and d!=h and 'contraction' in e and 'rest' in f and 'rest' in i:
                    crr += 1
                else:
                    #print('vals') 
                    continue
        match_loop += 1   
    if cnn > 0:
        cnn = cnn/2
    if rnn > 0:
        rnn = rnn/2
    if ncc > 0:
        ncc = ncc/2
    if nrr > 0:
        nrr = nrr/2
    if rcc > 0:
        rcc = rcc/2
    if crr > 0:
        crr = crr/2
    if rrr > 0:
        rrr = rrr/6
    if ccc > 0:
        ccc = ccc/6
    if nnn > 0:
        nnn = nnn/6
    print(nnn,ccc,rrr,cnn,rnn,ncc,nrr,ncr,rcc,crr)