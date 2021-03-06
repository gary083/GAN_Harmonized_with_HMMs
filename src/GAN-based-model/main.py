import tensorflow as tf 
import numpy as np 
import os, json, argparse 

from evalution import output_framewise_prob
# from  lib.data_load import *
import lib.data_load
import uns_model
import sup_model

def addParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',           type=str, default='train', help='')
    parser.add_argument('--model_type',     type=str, default='uns', help='')
    parser.add_argument('--cuda_id',        type=str, default='0', help='')
    parser.add_argument('--bnd_type',       type=str, default='orc', help='')
    parser.add_argument('--setting',        type=str, default='match', help='')
    parser.add_argument('--iteration',      type=int, default=1, help='')
    parser.add_argument('--aug',            action='store_true', help='')
    parser.add_argument('--data_dir',       type=str, default=f'/home/guanyu/guanyu/handoff/data') 
    parser.add_argument('--save_dir',       type=str, default=f'/home/guanyu/guanyu/handoff/data/save/test_model') 
    parser.add_argument('--config',         type=str, default=f'/home/guanyu/guanyu/handoff/src/GAN-based-model/config.yaml') 
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

def main(args, config):
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
    # build dataloader
    if args.mode=='train' or args.mode=='load':
        # load train
        train_data_loader = lib.data_load.data_loader(config, 
                                                  os.path.join(args.data_dir, config.train_feat_path), 
                                                  os.path.join(args.data_dir, config.train_phn_path), 
                                                  os.path.join(args.data_dir, config.train_orc_bnd_path), 
                                                  train_bnd_path=train_bnd_path, 
                                                  target_path=target_path, 
                                                  data_length=data_length, 
                                                  phn_map_path=phn_map_path,
                                                  name='DATA LOADER(train)')
        train_data_loader.print_parameter(True)
        # load dev
        dev_data_loader = lib.data_load.data_loader(config, 
                                                os.path.join(args.data_dir, config.dev_feat_path), 
                                                os.path.join(args.data_dir, config.dev_phn_path), 
                                                os.path.join(args.data_dir, config.dev_orc_bnd_path), 
                                                phn_map_path=phn_map_path,
                                                name='DATA LOADER(dev)')
        dev_data_loader.print_parameter()
    else:
        # load train for evalution
        train_data_loader = lib.data_load.data_loader(config, 
                                                  os.path.join(args.data_dir, config.train_feat_path), 
                                                  os.path.join(args.data_dir, config.train_phn_path), 
                                                  os.path.join(args.data_dir, config.train_orc_bnd_path), 
                                                  phn_map_path=phn_map_path,
                                                  name='DATA LOADER(evalution train)')

        dev_data_loader  = lib.data_load.data_loader(config,
                                                 os.path.join(args.data_dir, config.dev_feat_path), 
                                                 os.path.join(args.data_dir, config.dev_phn_path), 
                                                 os.path.join(args.data_dir, config.dev_orc_bnd_path), 
                                                 phn_map_path=phn_map_path,
                                                 name='DATA LOADER(evalution test)')

        train_data_loader.print_parameter()
        dev_data_loader.print_parameter()
    config.feat_dim = train_data_loader.feat_dim * config.concat_window
    config.phn_size = train_data_loader.phn_size
    config.mfcc_dim = train_data_loader.feat_dim
    config.save_path = f'{args.save_dir}/model'

    # build model
    if args.model_type == 'sup': g = sup_model.model(config)
    else: g = uns_model.model(config)
    print_bar()
    print_model_parameter(config)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    with tf.Session(graph=g.graph) as sess:
        print ('Building Session...')
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=3)
        print_bar()

        if args.mode == 'train':
            print_training_parameter(args, config)
            g.train(config, sess, saver, train_data_loader, dev_data_loader, args.aug)
            print_training_parameter(args, config)
            output_framewise_prob(sess, g, f'{args.save_dir}/train.pkl', train_data_loader)
            output_framewise_prob(sess, g, f'{args.save_dir}/test.pkl', dev_data_loader)

        elif args.mode == 'load':
            print_training_parameter(args, config)
            saver.restore(sess, tf.train.latest_checkpoint(args.save_dir))
            g.train(config, sess, saver, train_data_loader, dev_data_loader, args.aug)
            print_training_parameter(args, config)
            output_framewise_prob(sess, g, f'{args.save_dir}/train.pkl', train_data_loader)
            output_framewise_prob(sess, g, f'{args.save_dir}/test.pkl', dev_data_loader)
        else:
            saver.restore(sess, tf.train.latest_checkpoint(args.save_dir))
            output_framewise_prob(sess, g, f'{args.save_dir}/train.pkl', train_data_loader)
            output_framewise_prob(sess, g, f'{args.save_dir}/test.pkl', dev_data_loader)

if __name__ == "__main__":
    parser = addParser()
    args = parser.parse_args()
    config = lib.data_load.read_config(args.config)
    main(args, config)
