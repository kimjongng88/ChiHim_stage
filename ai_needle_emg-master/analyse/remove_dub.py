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
    fh = open('ai_needle_emg-master/analyse/Correct_0-0299/double_0_list.txt').readlines()
    sum = 0
    for line in fh:               
        row = line.split(',')
        seg_number, initials, label = [i.strip() for i in row]
        seg = segment_(sum, seg_number, initials, label)
        no_dubs[sum] = seg
        sum += 1 
    print(len(no_dubs))

    triple = { }
    fh = open('ai_needle_emg-master/analyse/Correct_0-0299/triple_0_list.txt').readlines()
    sum = 0
    for line in fh:               
        row = line.split(',')
        seg_number, initials, label = [i.strip() for i in row]
        seg = segment_(sum, seg_number, initials, label)
        triple[sum] = seg
        sum += 1 

    print('pop twee correct')
    x = 0
    while x<len(no_dubs):
        for y in triple:
            if no_dubs[x].seg_number == triple[y].seg_number:
                no_dubs.pop(x)
                break
            else:
                continue
        x += 1
    print(len(no_dubs))
    a=len(no_dubs)

    print('drie hernoemen')
    drie_dub_correct = { }
    key_value = 0
    for key, value in no_dubs.items():
        if value not in drie_dub_correct.values():
            drie_dub_correct[key] = value
            drie_dub_correct[key_value] = drie_dub_correct.pop(key)
            #print(drie_dub_correct[key_value].seg_number)
            key_value += 1  




    print('schrijven vier')
    x = 0
    file = open("ai_needle_emg-master/analyse/Correct_0-0299/double_0_list_correct.txt","w") 
    while x<a:
        file.write(str(drie_dub_correct[x].seg_number)+','+str(drie_dub_correct[x].initials)+','+str(drie_dub_correct[x].label)+'\n')
        x+=1

    
    
