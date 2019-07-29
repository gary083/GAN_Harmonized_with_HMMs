import numpy as np
import ark
import utils 
import shutil
import argparse
import os,sys
import utils 
import pickle as pkl

EPS = 0.00000001
penalty=1.0
lmwt=20

class Decoder():
    def __init__(self, graph_dir, posterior_dir, decode_dir, output_file, nj = 50):
        self.graph_dir = graph_dir
        self.decode_dir = decode_dir
        self.posterior_dir = posterior_dir
        self.output_file = output_file
        self.nj = nj # number of jobs of decoding to be parrallel
        
        if os.path.isdir(self.posterior_dir):
            shutil.rmtree(self.posterior_dir)
        if not os.path.isdir(self.posterior_dir):
            os.makedirs(self.posterior_dir)
        if not os.path.isdir(self.decode_dir):
            os.makedirs(self.decode_dir)

    def decode(self, likelihood, lengths, trans = None):
        # Input: 
        # likelihood: N * L * P numpy array
        #             N : number of utterance
        #             L : maximum number of frames of one utterance
        #             P : dimension of phone posterior (after transform)
        #                 The order should follow lang/phones.txt excepts <eps>,
        #                 #0~#4
        # length : 1d numpy array of N lengths of utterances
        # trans : Transcription of N utterances, which is a list of N string. If
        #         None, skip scoring
        
        # Write posterior to feats.scp (In order to call split_data.sh)
        writer = ark.ArkWriter(os.path.join(self.posterior_dir,'feats.scp'), os.path.join(self.posterior_dir,'likelihoods.ark'))
        N = likelihood.shape[0]
        n_digits = len(str(N))

        for idx,(output, l) in enumerate(zip(likelihood, lengths)):
            output = output[:l,:]
            output = np.where(output == 0, np.finfo(float).eps, output)
            output = np.ascontiguousarray(output, dtype=np.float32)
            writer.write_next_utt(str(self._number2str(idx, n_digits)).encode('utf-8'), np.log(output))
        writer.close()

        self._gen_utt2spk(N)
        if trans is not None:
            scoring_cmd = 'false'
            self._write_trans(trans)
        else:
            scoring_cmd = 'true'

        os.system('%s/scripts/decode.sh --cmd run.pl --skip_scoring %s --nj %s %s %s %s | tee %s/decode.log || exit 1;' % ( os.getcwd(), scoring_cmd, self.nj, self.graph_dir, self.posterior_dir, self.decode_dir, self.decode_dir))
        
        # Get best WER and print it
        wer = os.popen('grep WER {}/wer_{}_{}'.format(self.decode_dir,lmwt,penalty)).read()

        output_path = os.path.join(self.decode_dir,'scoring_kaldi/penalty_{}/{}.txt'.format(penalty,lmwt))

        os.system("cat {} | sort > {}".format(output_path, output_file))
        print("The file of decoding results is in: {}\n".format(output_file))


    def transform_likelihood(self, trans_array, likelihood):
        # Kaldi reordered the phones and add
        # spn symbol for representing <UNK>, so this function will add one
        # dimension with zero prob and change the original order.
        # Input: 
        # trans_array : 1d np array of mapping dimension
        # likelihood: N * L * P numpy array
        #             N : number of utterance
        #             L : maximum number of frames of one utterance
        #             P : dimension of original phone posterior 
        N,L,P = likelihood.shape
        output = np.zeros((N,L,P+1))
        output[:,:,:P] = likelihood
        output = output[:,:,trans_array]
        return output 
        
    def _gen_utt2spk(self, N):
        # Input : 
        # N : number of utterance
        n_digits = len(str(N))
        utt2spk_path = os.path.join(self.posterior_dir,'utt2spk')
        with open(utt2spk_path,'w') as f:
            for idx in range(N):
                f.write('{0} {0}\n'.format(self._number2str(idx,n_digits).encode('utf-8')))

    def _write_trans(self, trans):
        n_digits = len(str(len(trans)))
        text_path = os.path.join(self.posterior_dir,'text')
        with open(text_path,'w') as f:
            for idx, tran in enumerate(trans):
                string = ""
                for x in tran:
                    string += x + ' '
                f.write('{} {}\n'.format(self._number2str(idx,n_digits).encode('utf-8'),string))

    def _number2str(self, idx, n_digits = 4):
        str_idx = str(idx)
        for _ in range(n_digits - len(str_idx)):
            str_idx = '0' + str_idx
        return str_idx


def get_trans_array(ori_phone_file, tgt_phone_file):
    #transform the orders of phone to kaldi format
    import utils
    ori = utils.read_phone_txt(ori_phone_file)
    ori.append('spn')
    tgt = utils.read_phone_txt(tgt_phone_file,0)
    new_tgt = []
    for x in tgt:
        for s in ['<','#']:
            if x.startswith(s):
                break
        else:
            new_tgt.append(x)
    trans = []
    for ix,x in enumerate(new_tgt):
        for iy,y in enumerate(ori):
            if x == y:
                trans.append(iy)
    return np.array(trans)

def get_trans(self, trans_pkl):
    phns = pkl.load(open(trans_pkl ,'rb'))
    trans = []
    for idx, phn in enumerate(phns):
        string=''
        for x in phn:
            string += x + ' '
        trans.append(string)
    return trans

def addParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set_type',   type=str, default='test', help='')
    parser.add_argument('--lm_type',    type=str, default='match', help='')
    parser.add_argument('--data_path',  type=str, default='/home/guanyu/guanyu/handoff/data', help='')
    parser.add_argument('--prefix',     type=str, default='orc_iter1_match', help='')
    parser.add_argument('--jobs',       type=int, default=8, help='')
    return parser


if __name__ == '__main__':
    parser = addParser()
    args = parser.parse_args()

    save_dir      = f'{args.data_path}/save/{args.prefix}'
    lengths       = pkl.load(open(f'{args.data_path}/timit_for_GAN/audio/timit-{args.set_type}-length.pkl', 'rb'))
    transcription = pkl.load(open(f'{args.data_path}/timit_for_GAN/audio/timit-{args.set_type}-phn.pkl', 'rb'))
    likelihood    = pkl.load(open(f'{save_dir}/{args.set_type}.pkl', 'rb'))
    
    # Experiment dir
    graph_dir     = f'data/{args.lm_type}/tree_sp0.95/graph_9gram'
    decode_dir    = f'{save_dir}/decode_{args.set_type}'
    posterior_dir = f'{save_dir}/posterior'
    output_file   = f'{save_dir}/{args.set_type}_output.txt'

    ## Start decode
    trans_array = get_trans_array(f'{args.data_path}/phones.60-48-39.map.txt', 'data/lang/phones.txt')
    decoder = Decoder(graph_dir, posterior_dir, decode_dir, output_file, nj = args.jobs)
    likelihood = decoder.transform_likelihood(trans_array, likelihood)
    decoder.decode(likelihood, lengths, transcription)
  
