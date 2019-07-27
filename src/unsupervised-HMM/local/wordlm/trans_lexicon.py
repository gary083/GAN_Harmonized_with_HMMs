import sys

raw_lexicon_file = sys.argv[1]
phone_map = sys.argv[2]
lexicon_output = sys.argv[3]

L = []
with open(raw_lexicon_file, 'r') as f:
    for line in f:
        tokens = line.rstrip().split(' ')
        word = tokens[0]
        phones = [x.strip('/').strip('1').strip('2') for x in tokens[1:] if len(x) > 0]
        L.append((word,phones))

d = {}

with open(phone_map,'r') as f:
    for line in f:
        tokens = line.rstrip().split()
        d[tokens[0]] = d[tokens[1]]
with open(lexicon_output,'w') as f:
    for word,phones in L:
        f.write(word)
        for phn in phones:
            f.write(' ' + d[phn])
        f.write('\n')




     


