import sys

sys.path.append('scripts/')
import utils as utl

phone_map_txt = sys.argv[1]
phone_list_txt = sys.argv[2]

phone_list = utl.read_phone_txt(phone_map_txt)
utl.write_phone_file(phone_list, phone_list_txt)
