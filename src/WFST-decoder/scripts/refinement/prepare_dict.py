import sys,os
sys.path.append('scripts/')
import utils 

monophone_list=sys.argv[1] # not include spn
dict_dir=sys.argv[2]
phone_list = utils.read_phone_txt(monophone_list)
with open(os.path.join(dict_dir,'extra_questions.txt'),'w') as f:
    f.write('')

with open(os.path.join(dict_dir,'optional_silence.txt'),'w') as f:
    f.write('sil' + '\n')

with open(os.path.join(dict_dir,'silence_phones.txt'),'w') as f:
    f.write('sil' + '\n')
    f.write('spn' + '\n')

with open(os.path.join(dict_dir,'nonsilence_phones.txt'),'w') as f:
    for phone1 in phone_list:
        for phone2 in phone_list:
            f.write(phone1 + '_' + phone2 + '\n')
    
with open(os.path.join(dict_dir,'lexicon.txt'),'w') as f:
    for phone1 in phone_list:
        for phone2 in phone_list:
            f.write("{} {} {}\n".format(phone2, phone1 + '_' + phone2, phone2 + '_' + phone2))
        f.write("{} {}\n".format(phone1, phone1 + '_' + phone1))
    f.write("<UNK> spn\n")



