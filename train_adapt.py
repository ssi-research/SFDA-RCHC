
"""
 Parts of this code are based on https://github.com/tim-learn/SHOT-plus/code/uda/image_target.py
 The license of the file is in: https://github.com/tim-learn/SHOT-plus/blob/master/LICENSE
"""


import torch
import torch.optim as optim
import numpy as np
import os
import networks as nets
import warnings
from adapt_trainer import Trainer
import argparse
import random

warnings.simplefilter("ignore", UserWarning)


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def main():
    print('\n Setting up data sources ...')

    settings={}

    parser = argparse.ArgumentParser(description='UDA')
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--load_exp_name', type=str, default='')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--classifier_train', type=int, default=0)
    parser.add_argument('--dataset_exp_name', type=str, default='')
    parser.add_argument('--source', type=str, default='')
    parser.add_argument('--target', type=str, default='')
    parser.add_argument('--FE_M_train', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--use_diversity_loss', type=int, default=1)
    parser.add_argument('--apply_wn', type=int, default=1)
    parser.add_argument('--dataset_name', type=str, default='')
    parser.add_argument('--num_C', type=int, default=31)
    parser.add_argument('--start_iter', type=int, default=1)
    parser.add_argument('--optimizer_type', type=str, default='SGD')
    parser.add_argument('--lr_value', type=float, default=1e-2)
    parser.add_argument('--pseudo_label_classification', type=int, default=1)
    parser.add_argument('--max_iter', type=int, default=5000)
    parser.add_argument('--CE_factor', type=float, default=0.3)
    parser.add_argument('--max_epoch', type=int, default=15)
    parser.add_argument('--type_bottleneck', type=str, default="bn", help="bn or bn_relu")
    parser.add_argument('--load_downloaded_weights', type=int, default=0)
    parser.add_argument('--load_downloaded_weights_path', type=str, default='')
    parser.add_argument('--PL_distance', type=str, default='cosine', choices=["euclidean", "cosine"])
    parser.add_argument('--PL_threshold', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--interval', type=int, default=15, help="")
    parser.add_argument('--cnn_to_use', type=str, default='resnet50', help="")
    parser.add_argument('--PL_CE_start', type=int, default=0, help="")
    parser.add_argument('--PL_select_start_epoch', type=int, default=0, help="")
    parser.add_argument('--PL_select_end_epoch', type=int, default=300, help="")
    parser.add_argument('--output_root_path', type=str, default='./output', help="")
    parser.add_argument('--data_root_path', type=str, default="/local_datasets/da_datasets", help="")
    parser.add_argument('--pretrained_path', type=str, default= '')
    parser.add_argument('--consistency_based_PL', type=int, default=0, help="")
    parser.add_argument('--short_exp_name', type=str, default='')
    parser.add_argument('--ssl', type=float, default=0.6)
    parser.add_argument('--use_rot', type=int, default=1)
    parser.add_argument('--per_class_acc', type=int, default=0)
    parser.add_argument('--img_list_root_path', type=str, default="./da_datasets", help="")
    parser.add_argument('--apply_max', type=int, default=1)
    parser.add_argument('--PL_pause', type=int, default=0)
    parser.add_argument('--E_dims', type=int, default=256)
    parser.add_argument('--softmax_temperature', type=float, default=1.0)
    parser.add_argument('--gamma', type=int, default=10)
    parser.add_argument('--consistency_pause', type=int, default=0)
    parser.add_argument('--use_dist_for_consistency', type=int, default=0)
    parser.add_argument('--dist_ratio_threshold', type=float, default=0.5)


    args = parser.parse_args()

    # Update Config
    settings['dist_ratio_threshold'] = args.dist_ratio_threshold
    settings['use_dist_for_consistency'] = args.use_dist_for_consistency == 1
    settings['consistency_pause'] = args.consistency_pause == 1
    settings['gamma'] = args.gamma
    settings['softmax_temperature'] = args.softmax_temperature
    settings['E_dims'] = args.E_dims
    settings['PL_pause'] = args.PL_pause == 1
    settings['apply_max'] = args.apply_max == 1
    settings['img_list_root_path'] = args.img_list_root_path
    settings['ssl'] = args.ssl
    settings['start_iter'] = args.start_iter
    settings['per_class_acc'] = args.per_class_acc == 1
    settings['use_rot'] = args.use_rot == 1
    settings['short_exp_name'] = args.short_exp_name
    settings['consistency_based_PL'] = args.consistency_based_PL == 1
    settings['PL_select_start_epoch'] = args.PL_select_start_epoch
    settings['PL_select_end_epoch'] = args.PL_select_end_epoch
    settings['pretrained_path'] = args.pretrained_path
    settings['data_root_path'] = args.data_root_path
    settings['output_root_path'] = args.output_root_path
    settings['PL_CE_start'] = args.PL_CE_start == 1
    settings['cnn_to_use'] = args.cnn_to_use
    settings['interval'] = args.interval
    settings['val_after'] = 100
    settings['PL_distance'] = args.PL_distance
    settings['PL_threshold'] = args.PL_threshold
    settings['load_downloaded_weights_path'] = args.load_downloaded_weights_path
    settings['load_downloaded_weights'] = args.load_downloaded_weights == 1
    settings['type_bottleneck'] = args.type_bottleneck
    settings['max_epoch'] = args.max_epoch
    settings['CE_factor'] = args.CE_factor
    settings['num_C'] = args.num_C
    settings['dataset_name'] = args.dataset_name
    settings['apply_wn'] = args.apply_wn == 1
    settings['use_diversity_loss'] = args.use_diversity_loss == 1
    settings['dataset_exp_name'] = args.dataset_exp_name
    settings['source'] = args.source
    settings['target'] = args.target
    settings['batch_size'] = args.batch_size
    settings['optimizer_type'] = args.optimizer_type

    settings['optimizer'] = ['M', 'E', 'G']

    settings['use_loss'] = {
        'adaptation': True,
        'pseudo_label_classification': args.pseudo_label_classification == 1,
    }

    settings['to_train'] = {
        'M': args.FE_M_train == 1,
        'E': True,
        'G': args.classifier_train == 1,
    }

    settings['gpu'] = args.gpu
    settings['device'] = 'cuda:' + str(settings['gpu'])
    torch.cuda.set_device(settings['gpu'])

    settings['load_exp_name'] = args.load_exp_name
    settings['exp_name'] = args.exp_name

    if settings['optimizer_type'] == 'Adm':
        print('Adm optimizer')
        lr_value = 1e-5
    else:
        print('SGD optimizer')
        lr_value = args.lr_value

    settings['enough_iters'] = 50
    settings['max_iter'] = args.max_iter
    settings['lr'] = {
        'M': lr_value * 0.1,
        'E': lr_value,
        'G': lr_value * 0.1,
    }

    print('######## SANITY CHECK ########')
    for key in sorted(settings.keys()):
        print('{}: {}'.format(key, settings[key]))

    torch.cuda.set_device(settings['gpu'])

    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    exp_name = settings['exp_name']
    settings['weights_path'] = os.path.join(settings['output_root_path'], 'weights')
    print(settings['weights_path'])
    os.makedirs(os.path.join(settings['weights_path'], exp_name), exist_ok=True)

    # Building network
    print('\n Building network ...')
    network = nets.netSHOT(settings['num_C'], cnn=settings['cnn_to_use'],
                                  E_dims=settings['E_dims'], apply_wn=settings['apply_wn'],
                                  type_bottleneck=settings['type_bottleneck'], pretrained_path=settings['pretrained_path']).cuda()

    if settings['load_downloaded_weights']:
        modelpath = settings['load_downloaded_weights_path'] + '/source_F.pt'
        network.M.load_state_dict(torch.load(modelpath))
        modelpath = settings['load_downloaded_weights_path'] + '/source_B.pt'
        network.E.load_state_dict(torch.load(modelpath))
        modelpath = settings['load_downloaded_weights_path'] + '/source_C.pt'
        network.G.load_state_dict(torch.load(modelpath))
        print('load_downloaded_weights from '  + settings['load_downloaded_weights_path'])
    else:
        modelpath = os.path.join(settings['output_root_path'], 'weights', settings['load_exp_name'],'source_F.pt')
        network.M.load_state_dict(torch.load(modelpath))
        modelpath = os.path.join(settings['output_root_path'], 'weights', settings['load_exp_name'],'source_B.pt')
        network.E.load_state_dict(torch.load(modelpath))
        modelpath = os.path.join(settings['output_root_path'], 'weights', settings['load_exp_name'],'source_C.pt')
        network.G.load_state_dict(torch.load(modelpath))

    # Setting up optimizers
    print('\n Setting up optimizers ...')

    to_train = []
    for comp in settings['optimizer']:
        if settings['to_train'][comp]:
            to_train.append(
                {'params': network.components[comp].parameters(), 'lr': settings['lr'][comp]})

    netR=[]
    if settings['use_rot']:
        netR = nets.feat_classifier(type='linear', class_num=4, bottleneck_dim=2 * settings['E_dims']).cuda()
        for k, v in netR.named_parameters():
            to_train.append({'params': v, 'lr': lr_value })

    if settings['optimizer_type'] == 'Adm':
        optimizer = optim.Adam(params=to_train)
    else:
        optimizer = optim.SGD(params=to_train)
        optimizer = op_copy(optimizer)


    # TRAINING AND VALIDATION
    train(network, netR, optimizer, exp_name, settings)


def train(network, netR, optimizer, exp_name, settings):
    global least_val_loss

    train_iter = settings['start_iter']
    trainer = Trainer(network, netR, optimizer, settings)
    max_iter = trainer.max_iter

    while True:

        trainer.set_mode_train()
        trainer.train()

        if train_iter % (trainer.interval_iter) == 0 or train_iter == max_iter:

            print("\n----------- train_iter " + str(train_iter) + ' -----------\n')

            print('validating')

            trainer.set_mode_val()
            test(trainer, settings)

            if train_iter == max_iter:
                dict_to_save = {component: network.components[component].cpu().state_dict() for component in
                                network.components}
                torch.save(dict_to_save, os.path.join(os.path.join(settings['weights_path'], exp_name) + '/',
                                                      'last_' + str(train_iter) + '.pth'))

            if train_iter >= max_iter:
                break

        train_iter += 1


def test(trainer, settings):

    if settings['dataset_name'] == 'VisDA-C':
        val_acc, val_acc_classes = trainer.validation()
    else:
        val_acc = trainer.validation()

    trainer.log_errors()

    print("\n--- val accuracy: " + str(val_acc) + ' ---\n')

    if settings['dataset_name'] == 'VisDA-C':
        class_names = ['Plane', 'Bcycle', 'Bus', 'Car', 'Horse', 'Knife', 'Mcycl', 'Person', 'Plant',
                       'Sktbrd', 'Train', 'Truck']
        for i in range(0, len(class_names)):
            curr_str = "val acc - " + class_names[i]
            print("\n ---" + curr_str + ": "+ str(val_acc_classes[i]) + ' ---\n')



if __name__ == '__main__':
    main()



