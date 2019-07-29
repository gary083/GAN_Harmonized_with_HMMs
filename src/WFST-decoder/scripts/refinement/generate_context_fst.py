import sys
sys.path.append('scripts/')
import utils
import math

#48 phones
monophones_txt = sys.argv[1]
all_phones_txt = sys.argv[2]
self_loop_prob = sys.argv[3]

trans_prob = - math.log(1-float(self_loop_prob))
phones = utils.read_phone_txt(monophones_txt)
cd_phones = utils.read_phone_txt(all_phones_txt,0)
print("0 1 0 0")
for idx,phone in enumerate(cd_phones):
    if '_' in phone:
        x, y = phone.split('_')
        iy = phones.index(y)
        if x == y :
            print("1 {} {} {}".format(iy+2, idx, idx))
        else:
            print("1 {} {} {} {:.5f}".format(iy+2, idx, idx, trans_prob))


for idx,phone in enumerate(cd_phones):
    if '_' in phone:
        x, y = phone.split('_')
        iy = phones.index(y)
        ix = phones.index(x)
        if x == y :
            print("{} {} {} {}".format(ix+2 ,iy+2, idx, idx))
        else:
            print("{} {} {} {} {:.5f}".format(ix+2 ,iy+2, idx, idx, trans_prob))

disambig = []
for idx,phone in enumerate(cd_phones):
    if phone.startswith('#'):
        disambig.append(idx)

for idx,phone in enumerate(phones):
    for dis in disambig:
        print("{} {} {} {}".format(idx+2, idx+2, dis, dis))

print(len(phones)+2)

for idx,phone in enumerate(phones):
    print("{} {} {} {}".format(idx+2, len(phones)+2,0,0 ))
    
