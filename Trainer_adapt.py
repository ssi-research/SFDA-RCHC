
"""
 Parts of this code are based on https://github.com/tim-learn/SHOT-plus
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm
from augmentations import SHOT_train_augment, SHOT_test_augment
from scipy.spatial.distance import cdist
from data_loader import ImageList, MultiEpochsDataLoader
import rotation
import networks


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


##############################
class TrainerG():

    def __init__(self, network, netR, optimizer, settings):

        # Set the network and optimizer
        self.network = network
        self.to_train = settings['to_train']

        # Optimizers to use
        self.optimizer = optimizer

        # Save the settings
        self.settings = settings

        # Get number of classes
        self.num_C = settings['num_C']

        # Extract commonly used settings
        self.batch_size = settings['batch_size']
        self.current_iteration = settings['start_iter']

        if settings["dataset_name"] == 'DomainNet':
            if settings["num_C"] == 40:
                image_list_file = settings["target"] + "_test_mini.txt"
                image_list_path = os.path.join('./Data', self.settings["dataset_name"], image_list_file)
        else:
            image_list_file = settings["target"] + "_list.txt"
            image_list_path = os.path.join('./Data', self.settings["dataset_name"], image_list_file)

        image_list = open(image_list_path).readlines()
        val_image_list = image_list
        print(len(image_list))

        test_transform = SHOT_test_augment()

        val_dataset = ImageList(val_image_list, img_root_dir=self.settings['data_root_path'], transform=test_transform)
        self.val_loader = MultiEpochsDataLoader(val_dataset, batch_size=self.batch_size, shuffle=False,
                                                num_workers=8)

        self.set_mode_val()
        self.val_over_val_set()
        self.log_errors()
        self.set_mode_train()

        if self.settings['use_rot']:
            self.netR = netR
            netR_dict, acc_rot = self.train_target_rot()
            self.netR.load_state_dict(netR_dict)
            netR.train()

        self.pseudo_label_train = {}

        # Initialize data loaders
        self.get_all_dataloaders()

        self.max_iter = self.settings["max_epoch"] * len(self.train_loader)
        self.interval_iter = self.max_iter // self.settings["interval"]

        self.PL_select_iter_start = self.settings["PL_select_start_epoch"] * len(self.train_loader)
        self.PL_select_iter_end = self.settings["PL_select_end_epoch"] * len(self.train_loader)


    ##############################
    def get_all_dataloaders(self):

        if self.settings["dataset_name"] == 'DomainNet':
            if self.settings["num_C"] == 40:
                image_list_file = self.settings["target"] + "_train_mini.txt"
                image_list_path = os.path.join('./Data', self.settings["dataset_name"], image_list_file)
        else:
            image_list_file = self.settings["target"] + "_list.txt"
            image_list_path = os.path.join('./Data', self.settings["dataset_name"], image_list_file)

        image_list = open(image_list_path).readlines()
        train_image_list = image_list

        train_transform = SHOT_train_augment()
        test_transform = SHOT_test_augment()

        train_dataset = ImageList(train_image_list, img_root_dir=self.settings['data_root_path'], transform=train_transform)
        self.train_loader = MultiEpochsDataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                                  num_workers=8)

        train_dataset_2 = ImageList(train_image_list, img_root_dir=self.settings['data_root_path'],
                                transform=test_transform)
        self.PL_loader = MultiEpochsDataLoader(train_dataset_2, batch_size=self.batch_size, shuffle=False,
                                               num_workers=8)


    ##################################################
    def get_loss(self):

        # entropy
        y_tilde_s = F.softmax(self.features['G'] / self.settings['softmax_temperature'], dim=-1)

        H_s = - torch.sum(y_tilde_s * torch.log(y_tilde_s), dim=-1)

        if self.settings['consistency_based_PL'] and \
            ((self.settings['dataset_name'] != "VisDA-C" and (not self.settings['PL_pause'])) or self.current_iteration>= self.interval_iter or self.settings['PL_CE_start']) \
            and (not self.settings['consistency_pause'] or self.current_iteration>= self.interval_iter):

            concat_outputs = self.features['G']
            concat_softmax = F.softmax(concat_outputs / self.settings['softmax_temperature'], dim=-1)
            pred = torch.argmax(concat_softmax, dim=-1)

            self.pseudo_label_train['pl_dist_sort'] = self.pseudo_label_train['pl_dist'].sort()
            self.pseudo_label_train['pl_dist_sort']=self.pseudo_label_train['pl_dist_sort'] [0]

            if self.settings['use_dist_for_consistency']:

               threshold = self.settings['dist_ratio_threshold']

               indices = [i for i in range(len(pred)) if  (self.pseudo_label_train['pseudo_label_sample'][i] != pred[i] and \
                        (self.pseudo_label_train['pl_dist'][i][self.pseudo_label_train['pseudo_label_pos'][i]]/self.pseudo_label_train['pl_dist_sort'][i][1]) < threshold )]

            else:
               indices = [i for i in range(len(pred)) if  (self.pseudo_label_train['pseudo_label_sample'][i] != pred[i])]


            self.consistency_factor = torch.ones(self.img_target.size()[0]).cuda()
            if self.settings['apply_max']:
                self.consistency_factor[indices] = -1
            else:
                self.consistency_factor[indices] = 0


            loss_over_batch = [self.consistency_factor[i] * H_s[i] for i in range(H_s.size()[0])]
            loss_over_batch = torch.stack(loss_over_batch)

        else:
            loss_over_batch = H_s

        loss = torch.mean(loss_over_batch, dim=0)

        # add diversity loss
        if self.settings['use_diversity_loss']:
            msoftmax = y_tilde_s.mean(dim=0)
            diversity_loss = torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))
            loss -= diversity_loss

        if self.settings['use_loss']['pseudo_label_classification'] == True and \
                (self.current_iteration>= self.interval_iter or (self.settings['dataset_name'] != "VisDA-C" and (not self.settings['PL_pause']) ) or \
                 self.settings['PL_CE_start']):

            loss_CE = nn.CrossEntropyLoss(reduction='mean')(self.features['G'],
                                                           self.pseudo_label_train['pseudo_label_sample'])

            loss += self.settings['CE_factor']*loss_CE


        # add rotation loss
        if self.settings['use_rot']:

            curr_img_list = self.img_target
            r_labels_target = np.random.randint(0, 4, len(curr_img_list))
            r_inputs_target = rotation.rotate_batch_with_labels(curr_img_list, r_labels_target)
            r_labels_target = torch.from_numpy(r_labels_target).cuda()
            r_inputs_target = r_inputs_target.cuda()

            f_outputs = self.network.E(self.network.M(curr_img_list))
            f_outputs = f_outputs.detach()
            f_r_outputs = self.network.E(self.network.M(r_inputs_target))
            r_outputs_target = self.netR(torch.cat((f_outputs, f_r_outputs), 1))

            rotation_loss = self.settings['ssl'] * nn.CrossEntropyLoss()(r_outputs_target, r_labels_target)
            loss += rotation_loss

        self.summary_dict['loss/'] = loss.data.cpu().numpy()
        return loss

    ##############################
    def loss(self):

        concat_outputs = self.features['G']
        concat_softmax = F.softmax(concat_outputs / self.settings['softmax_temperature'], dim=-1)

        pred = torch.argmax(concat_softmax, dim=-1)

        target_acc = (pred.float() == self.gt.float()).float().mean()
        self.summary_dict['acc/target_acc'] = target_acc

        if self.phase == 'train':

            # ====== BACKPROP LOSSES ======
            self.optimizer.zero_grad()
            loss = self.get_loss()
            loss.backward()
            self.optimizer.step()

        self.current_iteration += 1

    ###################################################
    def forward(self):

        # Used for evaluation purposes
        self.gt = self.data[1].cuda()
        self.img_target = self.data[0][:, :3, :, :].cuda()
        self.img_idx = self.data[3]
        self.features = {}

        if self.settings['use_loss']['pseudo_label_classification'] == True:
            if (self.current_iteration-1) % self.interval_iter == 0:
                self.set_mode_val()
                self.mem_label, self.dd, self.pred_pos, self.labelsetMap = self.obtain_label()
                self.mem_label = torch.from_numpy(self.mem_label).cuda()
                self.dd = torch.from_numpy(self.dd).cuda()
                self.set_mode_train()

            self.pseudo_label_train['pseudo_label_sample'] = self.mem_label[self.img_idx]
            self.pseudo_label_train['pseudo_label_pos'] = self.pred_pos[self.img_idx]
            self.pseudo_label_train['pl_dist'] = self.dd[self.img_idx]

        # Unlabeled Target data
        self.features['M'] = self.network.M(self.img_target)
        self.features['G'] = self.network.G(self.network.E(self.features['M']))

    ##############################
    def lr_scheduler(self, optimizer, iter_num, max_iter, gamma=10, power=0.75):
        decay = (1 + gamma * iter_num / max_iter) ** (-power)
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr0'] * decay
            param_group['weight_decay'] = 1e-3
            param_group['momentum'] = 0.9
            param_group['nesterov'] = True
        return optimizer

    ##############################
    def train(self):

        self.phase = 'train'
        self.summary_dict = {}

        try:
            self.data = self.dataloader_train.__next__()[1]
            img = self.data[0]
            if img.shape[0] == 1:
                self.dataloader_train = enumerate(self.train_loader)
                self.data = self.dataloader_train.__next__()[1]
        except:
            self.dataloader_train = enumerate(self.train_loader)
            self.data = self.dataloader_train.__next__()[1]

        if self.settings['optimizer_type']=='SGD':
            self.lr_scheduler(self.optimizer, iter_num=self.current_iteration,
                              max_iter=self.max_iter, gamma=self.settings['gamma'])


        self.forward()
        self.loss()

        return self.summary_dict['acc/target_acc']

    ##############################
    def log_errors(self):
        print(self.summary_dict)


    ##############################
    def set_mode_val(self):

        self.network.eval()
        self.backward = False
        for p in self.network.parameters():
            p.requires_grad = False

    ##############################
    def set_mode_train(self):

        self.network.train()
        self.backward = True
        for p in self.network.parameters():
            p.requires_grad = True

        for comp in self.settings['to_train']:
            if self.settings['to_train'][comp] == False:
                self.network.components[comp].eval()
                for p in self.network.components[comp].parameters():
                    p.requires_grad = False


    ##############################
    def val_over_val_set(self):

        with torch.no_grad():
            self.summary_dict = {}

            # ----------------------
            # Target validation Data
            # ----------------------

            print('\nValidating on target validation data')

            num_C = self.num_C

            classes = list(range(num_C))

            avg_acc = {c: 0 for c in classes}
            avg_count = {c: 0 for c in classes}

            idx = -1
            overall_acc = 0
            overall_acc_count = 0

            for data in tqdm(self.val_loader):
                idx += 1
                x = data[0][:, :3, :, :].to(self.settings['device'])
                labels_target = data[1].to(self.settings['device'])

                M = self.network.components['M'](x)
                E = self.network.components['E'](M)
                G = self.network.components['G'](E)

                concat_outputs = G
                concat_softmax = F.softmax(concat_outputs / self.settings['softmax_temperature'], dim=-1)

                max_act, pred = torch.max(concat_softmax, dim=-1)

                for c in classes:
                    avg_acc[c] += (pred[labels_target == c] == labels_target[labels_target == c]).float().sum()
                    avg_count[c] += pred[labels_target == c].shape[0]

                overall_acc += (pred == labels_target).float().sum()
                overall_acc_count += pred.shape[0]

            overall_acc = float(overall_acc) / float(overall_acc_count)

            # average accuracy
            avg = 0
            num_classes = num_C
            for c in classes:
                if avg_count[c] == 0:
                    avg += 0
                else:
                    avg += (float(avg_acc[c]) / float(avg_count[c]))
                    if self.settings['dataset_name'] == 'VisDA-C':
                        avg_acc[c] = (float(avg_acc[c]) / float(avg_count[c]))

            avg /= float(num_classes)
            self.summary_dict['acc/target_cs_avg'] = avg

            self.summary_dict['overall_acc/target_cs_avg'] = overall_acc

            if self.settings['dataset_name'] == 'VisDA-C':
                self.summary_dict['class_acc/target_cs_avg'] = avg_acc

            if self.settings['dataset_name'] == 'VisDA-C':
                return avg,avg_acc

            if self.settings['per_class_acc'] or self.settings['dataset_name'] == 'DomainNet':
                return avg

            return overall_acc


    #######################################
    def obtain_label(self):
        start_test = True
        with torch.no_grad():

            for data in tqdm(self.PL_loader):
                x = data[0][:, :3, :, :].to(self.settings['device'])
                labels_target = data[1]

                M = self.network.components['M'](x)
                feas = self.network.components['E'](M)
                outputs = self.network.components['G'](feas)

                if start_test:
                    all_fea = feas.float().cpu()
                    all_output = outputs.float().cpu()
                    all_label = labels_target.float()
                    start_test = False
                else:
                    all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels_target.float()), 0)

        all_output = nn.Softmax(dim=1)(all_output)
        _, predict = torch.max(all_output, 1)

        accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
        if self.settings['PL_distance'] == 'cosine':
            all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
            all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

        ###
        all_fea = all_fea.float().cpu().numpy()
        K = all_output.size(1)
        aff = all_output.float().cpu().numpy()

        for _ in range(2):
            initc = aff.transpose().dot(all_fea)
            initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
            cls_count = np.eye(K)[predict].sum(axis=0)
            labelset = np.where(cls_count>self.settings['PL_threshold'])
            labelset = labelset[0]

            dd = cdist(all_fea, initc[labelset],  self.settings['PL_distance'])
            pred_label = dd.argmin(axis=1)
            predict = labelset[pred_label]

            aff = np.eye(K)[predict]

        acc = np.sum(predict == all_label.float().numpy()) / len(all_fea)
        log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)

        print(log_str+'\n')

        labelsetMap = -1*np.ones(self.num_C)
        for i in range(len(labelset)):
            labelsetMap[labelset[i]] = i

        return predict.astype('int'), dd, pred_label.astype('int'), labelsetMap.astype('int')


    #######################
    # rotation classifier
    def train_target_rot(self):

        if self.settings["dataset_name"] == 'DomainNet':
         if self.settings["num_C"] == 40:
             image_list_file = self.settings["target"] + "_train_mini.txt"
             image_list_path = os.path.join('./Data', self.settings["dataset_name"], image_list_file)

        else:
            image_list_file = self.settings["target"] + "_list.txt"
            image_list_path = os.path.join('./Data', self.settings["dataset_name"], image_list_file)

        image_list = open(image_list_path).readlines()
        train_image_list = image_list
        train_transform = SHOT_train_augment()
        train_dataset = ImageList(train_image_list, img_root_dir=self.settings['data_root_path'],
                                  transform=train_transform)
        train_rot_loader = MultiEpochsDataLoader(train_dataset, batch_size=self.batch_size,
                                                  shuffle=True, num_workers=8)

        ## set base network
        if self.settings['cnn_to_use'][0:3] == 'res':
            netF = networks.ResBase(res_name=self.settings['cnn_to_use']).cuda()
        elif self.settings['cnn_to_use'] == 'convnextTiny':
            netF = networks.ConvnextTiny().cuda()
        elif self.settings['cnn_to_use'] == 'convnextTiny22':
            netF = networks.ConvnextTiny2(in_22k=True).cuda()
        elif self.settings['cnn_to_use'] == 'convnextSmall':
            netF = networks.ConvnextSmall().cuda()
        elif self.settings['cnn_to_use'] == 'convnextSmall22':
            netF = networks.ConvnextSmall(in_22k=True).cuda()

        netB = networks.feat_bootleneck(type="bn", feature_dim=netF.in_features,
                                       bottleneck_dim=self.settings['E_dims']).cuda()
        netR = networks.feat_classifier(type='linear', class_num=4, bottleneck_dim=2 * self.settings['E_dims']).cuda()

        if self.settings['load_downloaded_weights']:
            modelpath = self.settings['load_downloaded_weights_path']
        else:
            modelpath = os.path.join(self.settings['output_root_path'], 'weights', self.settings['load_exp_name'])

        netF.load_state_dict(torch.load(modelpath + '/source_F.pt'))
        netF.eval()
        for k, v in netF.named_parameters():
            v.requires_grad = False

        netB.load_state_dict(torch.load(modelpath + '/source_B.pt'))
        netB.eval()
        for k, v in netB.named_parameters():
            v.requires_grad = False

        param_group = []
        for k, v in netR.named_parameters():
            param_group += [{'params': v, 'lr': self.settings['lr']['E'] * 1}]
        netR.train()
        optimizer = optim.SGD(param_group)
        optimizer = op_copy(optimizer)

        max_epoch=15
        max_iter = max_epoch * len(train_rot_loader)
        interval_iter = max_iter // 10
        iter_num = 0

        rot_acc = 0
        while iter_num < max_iter:
            optimizer.zero_grad()
            try:
                data = iter_test.__next__()[1]
            except:
                iter_test = enumerate(train_rot_loader)
                data = iter_test.__next__()[1]

            inputs_test = data[0]

            if inputs_test.shape[0] == 1:
                continue

            inputs_test = inputs_test.cuda()

            iter_num += 1
            self.lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

            r_labels_target = np.random.randint(0, 4, len(inputs_test))
            r_inputs_target = rotation.rotate_batch_with_labels(inputs_test, r_labels_target)
            r_labels_target = torch.from_numpy(r_labels_target).cuda()
            r_inputs_target = r_inputs_target.cuda()

            f_outputs = netB(netF(inputs_test))
            f_r_outputs = netB(netF(r_inputs_target))
            r_outputs_target = netR(torch.cat((f_outputs, f_r_outputs), 1))

            rotation_loss = nn.CrossEntropyLoss()(r_outputs_target, r_labels_target)
            rotation_loss.backward()

            optimizer.step()

            if iter_num % interval_iter == 0 or iter_num == max_iter:
                netR.eval()
                acc_rot = self.cal_acc_rot(train_rot_loader, netF, netB, netR)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(self.settings['exp_name'], iter_num, max_iter, acc_rot)
                print(log_str + '\n')
                netR.train()

                if rot_acc < acc_rot:
                    rot_acc = acc_rot
                    best_netR = netR.state_dict()

        log_str = 'Best Accuracy = {:.2f}%'.format(rot_acc)
        print(log_str + '\n')

        return best_netR, rot_acc

    ##############################
    def cal_acc_rot(self, loader, netF, netB, netR):
        start_test = True
        with torch.no_grad():
            iter_test = enumerate(loader)
            for i in range(len(loader)):
                data = iter_test.__next__()[1]
                inputs = data[0].cuda()
                r_labels = np.random.randint(0, 4, len(inputs))
                r_inputs = rotation.rotate_batch_with_labels(inputs, r_labels)
                r_labels = torch.from_numpy(r_labels)
                r_inputs = r_inputs.cuda()

                f_outputs = netB(netF(inputs))
                f_r_outputs = netB(netF(r_inputs))

                r_outputs = netR(torch.cat((f_outputs, f_r_outputs), 1))
                if start_test:
                    all_output = r_outputs.float().cpu()
                    all_label = r_labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, r_outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, r_labels.float()), 0)
        _, predict = torch.max(all_output, 1)
        accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])

        return accuracy * 100
