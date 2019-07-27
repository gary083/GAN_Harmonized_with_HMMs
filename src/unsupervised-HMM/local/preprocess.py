import sys
import utils as utl
import pickle as pkl

phone_map_txt = sys.argv[1]
phone_list_txt = sys.argv[2]

phone_list = utl.read_phone_txt(phone_map_txt)
phone_list.append('spn') # for oov word, which is requried for kaldi
utl.write_phone_file(phone_list, phone_list_txt)


