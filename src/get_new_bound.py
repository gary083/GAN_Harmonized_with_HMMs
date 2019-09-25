import _pickle as pk
import argparse

from bnd_ops import *


def addParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bnd_type', type=str, default='orc', help='')
    parser.add_argument('--iteration', type=int, default=1, help='')
    parser.add_argument('--prefix', type=str, default='orc_iter1_match', help='')
    parser.add_argument('--data_path', type=str, default='/home/guanyu/guanyu/handoff/data', help='')
    return parser


if __name__ == '__main__':
    parser = addParser()
    args = parser.parse_args()

    # oracle information
    train_phn_path = f'{args.data_path}/timit_for_GAN/audio/timit-train-phn.pkl'
    phone_label = load_pickle(train_phn_path)

    # phone information
    lexicon_path = f'{args.data_path}/phones.60-48-39.map.txt'
    phn2idx, idx2phn, phn_mapping = read_phn_map(lexicon_path)

    # put your decoder output here!!
    decode_output_path = f'{args.data_path}/save/{args.prefix}/phones_ali.txt'

    new_bound, frame_output, phone_output = read_phn_boundary(decode_output_path)
    pk.dump(new_bound,
            open(f'{args.data_path}/timit_for_GAN/audio/timit-train-{args.bnd_type}{args.iteration + 1}-bnd.pkl', 'wb'))
