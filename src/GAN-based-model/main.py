import numpy as np
import os
import json
import argparse
import yaml

from src.data.dataset import PickleDataset
from src.data.dataLoader import DataLoader
from src.models.uns_model import UnsModel
from src.models.sup_model import SupModel


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def read_config(path):
    return AttrDict(yaml.load(open(path, 'r')))

def addParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',           type=str, default='train', help='')
    parser.add_argument('--model_type',     type=str, default='uns', help='')
    parser.add_argument('--cuda_id',        type=str, default='0', help='')
    parser.add_argument('--bnd_type',       type=str, default='orc', help='')
    parser.add_argument('--setting',        type=str, default='match', help='')
    parser.add_argument('--iteration',      type=int, default=1, help='')
    parser.add_argument('--aug',            action='store_true', help='')
    parser.add_argument('--data_dir',       type=str, default=f'/home/r06942045/myProjects/GAN_Harmonized_with_HMMs/data') 
    parser.add_argument('--save_dir',       type=str, default=f'/home/r06942045/myProjects/GAN_Harmonized_with_HMMs/data/save/test_model') 
    parser.add_argument('--load_ckpt',       type=str, default=f'ckpt_9000.pth') 
    parser.add_argument('--config',         type=str, default=f'/home/r06942045/myProjects/GAN_Harmonized_with_HMMs/src/GAN-based-model/config.yaml') 
    return parser

def print_bar():
    print ('='*80)

def print_model_parameter(config):
    print ('Model Parameter:')
    print (f'   generator first layer:     {config.gen_hidden_size}')
    print (f'   frame temperature:         {config.frame_temp}')
    print (f'   dicriminator emb size:     {config.dis_emb_size}')
    print (f'   dicriminator first layer:  {config.dis_hidden_1_size}')
    print (f'   dicriminator second layer: {config.dis_hidden_2_size}')
    print (f'   intra-segment loss ratio:  {config.seg_loss_ratio}')
    print (f'   gradient penalty ratio:    {config.penalty_ratio}')
    print_bar()

def print_training_parameter(args, config):
    print ('Training Parameter:')
    print (f'   batch_size:             {config.batch_size}')
    if args.model_type == 'sup':
        print (f'   epoch:                  {config.epoch}')
        print (f'   learning rate(sup):     {config.sup_lr}')
    elif args.model_type == 'uns':
        print (f'   repeat:                 {config.repeat}')
        print (f'   step:                   {config.step}')
        print (f'   learning rate(gen):     {config.gen_lr}')
        print (f'   learning rate(dis):     {config.dis_lr}')
        print (f'   dis iteration:          {config.dis_iter}')
        print (f'   gen iteration:          {config.gen_iter}')
        print (f'   setting:                {args.setting}')
        print (f'   aug:                    {args.aug}')
        print (f'   bound type:             {args.bnd_type}')
        print (f'   data_dir:               {args.data_dir}')
        print (f'   save_dir:               {args.save_dir}')
        print (f'   config_path:            {args.config}')
    print_bar()     


if __name__ == "__main__":
    parser = addParser()
    args = parser.parse_args()
    config = read_config(args.config)

    ######################################################################
    # Environment & argument settings
    #
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_id

    train_bnd_path    = f'{args.data_dir}/timit_for_GAN/audio/timit-train-{args.bnd_type}{args.iteration}-bnd.pkl'
    output_path       = f'{args.save_dir}/train.pkl'
    phn_map_path      = f'{args.data_dir}/phones.60-48-39.map.txt'

    if args.setting == 'match': 
        data_length = None # Whole Source
        target_path = os.path.join(args.data_dir, config.match_target_path)
    elif args.setting == 'nonmatch':
        data_length = 3000
        target_path = os.path.join(args.data_dir, config.nonmatch_target_path)
    else: raise Exception("Invalid setting!", args.setting)
    print_bar()

    ######################################################################
    # Build dataset
    #
    if args.mode=='train' or args.mode=='load':
        # load train
        train_data_set = PickleDataset(config,
                                       os.path.join(args.data_dir, config.train_feat_path),
                                       os.path.join(args.data_dir, config.train_phn_path),
                                       os.path.join(args.data_dir, config.train_orc_bnd_path),
                                       train_bnd_path=train_bnd_path,
                                       target_path=target_path,
                                       data_length=data_length, 
                                       phn_map_path=phn_map_path,
                                       name='DATA LOADER(train)',
                                       random_batch=True,
                                       n_steps=config.step)
        train_data_set.print_parameter(True)
        # load dev
        dev_data_set = PickleDataset(config,
                                     os.path.join(args.data_dir, config.dev_feat_path),
                                     os.path.join(args.data_dir, config.dev_phn_path),
                                     os.path.join(args.data_dir, config.dev_orc_bnd_path),
                                     phn_map_path=phn_map_path,
                                     name='DATA LOADER(dev)',
                                     mode='dev')
        dev_data_set.print_parameter()
    else:
        # load train for evalution
        train_data_set = PickleDataset(config,
                                       os.path.join(args.data_dir, config.train_feat_path),
                                       os.path.join(args.data_dir, config.train_phn_path),
                                       os.path.join(args.data_dir, config.train_orc_bnd_path),
                                       phn_map_path=phn_map_path,
                                       name='DATA LOADER(evaluation train)',
                                       mode='dev')
        dev_data_set = PickleDataset(config,
                                     os.path.join(args.data_dir, config.dev_feat_path),
                                     os.path.join(args.data_dir, config.dev_phn_path),
                                     os.path.join(args.data_dir, config.dev_orc_bnd_path),
                                     phn_map_path=phn_map_path,
                                     name='DATA LOADER(evaluation test)',
                                     mode='dev')
        train_data_set.print_parameter()
        dev_data_set.print_parameter()
    config.feat_dim = train_data_set.feat_dim * config.concat_window
    config.phn_size = train_data_set.phn_size
    config.mfcc_dim = train_data_set.feat_dim
    config.save_path = f'{args.save_dir}/model'
    config.load_path = f'{config.save_path}/{args.load_ckpt}'


    ######################################################################
    # Build model
    #
    if args.model_type == 'sup':
        g = SupModel(config)
    else:
        g = UnsModel(config)
    print_bar()
    print_model_parameter(config)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    print ('Building Session...')
    print_bar()

    if args.mode == 'train':
        print_training_parameter(args, config)
        g.train(train_data_set, dev_data_set, args.aug)
        print_training_parameter(args, config)
        g.test(train_data_set, f'{args.save_dir}/train.pkl')
        g.test(dev_data_set, f'{args.save_dir}/test.pkl')

    elif args.mode == 'load':
        print_training_parameter(args, config)
        g.load_ckpt(config.load_path)
        g.train(train_data_set, dev_data_set, args.aug)
        print_training_parameter(args, config)
        g.test(train_data_set, f'{args.save_dir}/train.pkl')
        g.test(dev_data_set, f'{args.save_dir}/test.pkl')

    else:
        g.load_ckpt(config.load_path)
        g.test(train_data_set, f'{args.save_dir}/train.pkl')
        g.test(dev_data_set, f'{args.save_dir}/test.pkl')

