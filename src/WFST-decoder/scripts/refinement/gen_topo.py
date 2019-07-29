#!/usr/bin/env python2

# Copyright 2012  Johns Hopkins University (author: Daniel Povey)

# Generate a topology file.  This allows control of the number of states in the
# non-silence HMMs, and in the silence HMMs.  This is a modified version of
# 'utils/gen_topo.pl' that generates a different type of topology, one that we
# believe should be useful in the 'chain' model.  Note: right now it doesn't
# have any real options, and it treats silence and nonsilence the same.  The
# intention is that you write different versions of this script, or add options,
# if you experiment with it.

from __future__ import print_function
import argparse,sys
sys.path.append('scripts/')
import utils 


parser = argparse.ArgumentParser()
parser.add_argument("phone_list_txt", type=str)
parser.add_argument("--self_loop_prob", type=float, default=0.5,
                    help="Probabilty of staying in the same state. (1-self_loop_prob) is the probability of transition to other state")

args = parser.parse_args()
phone_list = utils.read_phone_txt(args.phone_list_txt,0)

same_phone = []
trans_phone = []
for idx, phone in enumerate(phone_list):
    if phone in ['sil','spn']:
        same_phone.append(idx)
    if '_' in phone:
        x, y = phone.split('_')
        if x == y :
            same_phone.append(idx)
        else:
            trans_phone.append(idx)

print("<Topology>")
print("<TopologyEntry>")
print("<ForPhones>")
print(" ".join([str(x) for x in same_phone]))
print("</ForPhones>")
print("<State> 0 <PdfClass> 0 <Transition> 0 {} <Transition> 1 {} </State>".format(args.self_loop_prob,1 - args.self_loop_prob))
print("<State> 1 </State>")
print("</TopologyEntry>")
print("<TopologyEntry>")
print("<ForPhones>")
print(" ".join([str(x) for x in trans_phone]))
print("</ForPhones>")
print("<State> 0 <PdfClass> 0 <Transition> 1 {} </State>".format(1))
print("<State> 1 </State>")
print("</TopologyEntry>")
print("</Topology>")

