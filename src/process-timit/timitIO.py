import _pickle as cPickle

class timit_path:

    train_meta = '/nfs/Caishen/givebirthday/timit/processed/timit-train-meta.pkl'
    train_mfcc = '/nfs/Caishen/givebirthday/timit/processed/timit-train-mfcc.pkl'
    train_fbank = '/nfs/Caishen/givebirthday/timit/processed/timit-train-fbank.pkl'
    train_spec = '/nfs/Caishen/givebirthday/timit/processed/timit-train-spec.pkl'
    train_tran = '/nfs/Caishen/givebirthday/timit/processed/timit-train-tran.txt'
    train_wrd  = '/nfs/Caishen/givebirthday/timit/processed/timit-train-wrd.pkl'
    train_phn  = '/nfs/Caishen/givebirthday/timit/processed/timit-train-phn.pkl'
    train_mfcc_nor = '/nfs/Caishen/givebirthday/timit/processed/timit-train-mfcc-nor.pkl'
    train_fbank_nor = '/nfs/Caishen/givebirthday/timit/processed/timit-train-fbank-nor.pkl'
    train_spec_nor = '/nfs/Caishen/givebirthday/timit/processed/timit-train-spec-nor.pkl'

    test_meta = '/nfs/Caishen/givebirthday/timit/processed/timit-test-meta.pkl'
    test_mfcc = '/nfs/Caishen/givebirthday/timit/processed/timit-test-mfcc.pkl'
    test_fbank = '/nfs/Caishen/givebirthday/timit/processed/timit-test-fbank.pkl'
    test_spec = '/nfs/Caishen/givebirthday/timit/processed/timit-test-spec.pkl'
    test_tran = '/nfs/Caishen/givebirthday/timit/processed/timit-test-tran.txt'
    test_wrd  = '/nfs/Caishen/givebirthday/timit/processed/timit-test-wrd.pkl'
    test_phn  = '/nfs/Caishen/givebirthday/timit/processed/timit-test-phn.pkl'
    test_mfcc_nor = '/nfs/Caishen/givebirthday/timit/processed/timit-test-mfcc-nor.pkl'
    test_fbank_nor = '/nfs/Caishen/givebirthday/timit/processed/timit-test-fbank-nor.pkl'
    test_spec_nor = '/nfs/Caishen/givebirthday/timit/processed/timit-test-spec-nor.pkl'

if __name__ == '__main__':
    train_wrd = cPickle.load(open(timit_path.train_wrd, 'rb'))
    train_phn = cPickle.load(open(timit_path.train_phn, 'rb'))

    with open('/nfs/Caishen/riviera1020/timit/timit-train.txt', 'r') as f:
        path = f.readlines()

    for i, (phn, wrd) in enumerate(zip(train_phn, train_wrd)):
        phn_error = False
        for l in phn:
            if l[1] > l[2]:
                print(f"Error in {path[i].rstrip()}")
                print(l)

        wrd_error = False
        for l in wrd:
            if l[1] > l[2]:
                print(f"Error in {path[i].rstrip()}")
                print(l)

    test_wrd = cPickle.load(open(timit_path.test_wrd, 'rb'))
    test_phn = cPickle.load(open(timit_path.test_phn, 'rb'))

    with open('/nfs/Caishen/riviera1020/timit/timit-test.txt', 'r') as f:
        path = f.readlines()

    for i, (phn, wrd) in enumerate(zip(test_phn, test_wrd)):
        phn_error = False
        for l in phn:
            if l[1] > l[2]:
                print(f"Error in {path[i].rstrip()}")
                print(l)

        wrd_error = False
        for l in wrd:
            if l[1] > l[2]:
                print(f"Error in {path[i].rstrip()}")
print(l)