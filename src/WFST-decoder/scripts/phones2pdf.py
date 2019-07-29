import utils
import sys
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("phone_table_path", type=str)
parser.add_argument("pdf_table_path", type=str)
parser.add_argument("--typ", type=str, default = 'mono', choices=['mono','biphone'])

args = parser.parse_args()


phones = utils.read_phone_txt(args.phone_table_path,0)
valid_phone = []
for x in phones:
    for s in ['<','#']:
        if x.startswith(s):
            break
    else:
        valid_phone.append(x)
if args.typ == 'mono':
    utils.write_phone_file(valid_phone, args.pdf_table_path, True)
else:
    L = []
    for i in range(len(valid_phone)):
        for j in range(len(valid_phone)+1):
            if j == 0:
                L.append('start' + '_' + valid_phone[i])
            else:
                L.append(valid_phone[j-1] + '_' + valid_phone[i])
    utils.write_phone_file(L, args.pdf_table_path, True)


