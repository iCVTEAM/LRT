from .base import Trainer
import os.path as osp
import torch.nn as nn
from copy import deepcopy

from .helper import *
from utils import *
from dataloader.data_utils import *


class FSCILTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.set_save_path()
        self.args = set_up_datasets(self.args)

        self.model = MYNET(self.args, mode=self.args.base_mode)
        self.model = nn.DataParallel(self.model, list(range(self.args.num_gpu)))
        self.model = self.model.cuda()

        if self.args.model_dir is not None:
            print('Loading init parameters from: %s' % self.args.model_dir)
            self.best_model_dict = torch.load(self.args.model_dir)['params']
        else:
            print('random init params')
            if args.start_session > 0:
                print('WARING: Random init weights for new sessions!')
            self.best_model_dict = deepcopy(self.model.state_dict())

    def get_optimizer_base(self):

        """ Set base optimization parameters """
        
        params = []
        for name, param in self.model.named_parameters():
            if "transformer" in name or "token" in name or "positional_embedding" in name or "text_projection" in name or "logit_scale" in name:
                param.requires_grad = False
            elif "prompt_text" in name or "gamma" in name or "proj_prompt" in name:
                param.requires_grad = True
                print(name)
                params.append({"params": param,'weight_decay': self.args.decay, 'lr': self.args.lr_base})
            elif "visual" in name: 
              #print(name)
              param.requires_grad = True
              params.append({"params": param,'weight_decay': self.args.decay, 'lr': self.args.lr_base*0.1})
            elif "scale_mm" in name: 
              #print(name)
              param.requires_grad = True
            else:
              print(name)
              params.append({"params": param,'weight_decay': self.args.decay, 'lr': self.args.lr_base})
        #optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), self.args.lr_base, momentum=0.9, nesterov=True,
        #                            weight_decay=self.args.decay)
        optimizer = torch.optim.SGD(params, self.args.lr_base, momentum=0.9, nesterov=True,
                                    weight_decay=self.args.decay)
        if self.args.schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step, gamma=self.args.gamma)
        elif self.args.schedule == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.milestones,
                                                             gamma=self.args.gamma)

        return optimizer, scheduler
        
    def get_optimizer_prompt(self):
        params = []
        lr = 0.01
        for name, param in self.model.named_parameters():
            if "prompt_text" in name or "gamma" in name or "proj_prompt" in name:
                param.requires_grad = True
                print(name)
                # set training parameters, the prompt parameters are enabled.
                params.append({"params": param,'weight_decay': self.args.decay, 'lr': lr})
            #elif "visual" in name: 
            #    print(name)
            #    param.requires_grad = True
            #    params.append({"params": param,'weight_decay': self.args.decay, 'lr': lr*0.1})
            else:
                param.requires_grad = False
                #print(name+"not used")
        optimizer = torch.optim.AdamW(params,lr=0.01,betas=(0.9,0.999),eps=1e-08,weight_decay=0.0001,amsgrad=False)
        #optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), self.args.lr_base*0.1, momentum=0.9, nesterov=True,
        #                            weight_decay=self.args.decay)
        if self.args.schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step, gamma=self.args.gamma)
        elif self.args.schedule == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20,30],
                                                             gamma=self.args.gamma)

        return optimizer, scheduler
    def get_dataloader(self, session):
        if session == 0:
            trainset, trainloader, testloader = get_base_dataloader(self.args)
        else:
            trainset, trainloader, testloader = get_new_dataloader(self.args, session)
        return trainset, trainloader, testloader

    def train(self):
        args = self.args
        t_start_time = time.time()

        # init train statistics
        result_list = [args]
        
        for session in range(args.start_session, args.sessions):

            train_set, trainloader, testloader = self.get_dataloader(session)
            self.model.module.text_dict = train_set.text_dict

            self.model.load_state_dict(self.best_model_dict,strict = False)

            if session == 0:  # load base class train img label

                print('new classes for this session:\n', np.unique(train_set.targets))
                optimizer, scheduler = self.get_optimizer_base()
                print('initialization optimizer \n')
                #self.model = update_textproto(train_set, self.model, args)
                for epoch in range(0,args.epochs_base):
                    if epoch == 90:
                        optimizer, scheduler = self.get_optimizer_prompt()
                    start_time = time.time()
                    # train base sess
                    tl, ta = base_train(self.model, trainloader, optimizer, scheduler, epoch, args)
                    # test model with all seen class
                    tsl, tsa, va_base, va_new = test(self.model, testloader, epoch, args, session)

                    # save better model
                    if (tsa * 100) >= self.trlog['max_acc'][session]:
                        self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                        self.trlog['max_acc_epoch'] = epoch
                        save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                        torch.save(dict(params=self.model.state_dict()), save_model_dir)
                        torch.save(optimizer.state_dict(), os.path.join(args.save_path, 'optimizer_best.pth'))
                        self.best_model_dict = deepcopy(self.model.state_dict())
                        print('********A better model is found!!**********')
                        print('Saving model to :%s' % save_model_dir)
                    print('best epoch {}, best test acc={:.3f}'.format(self.trlog['max_acc_epoch'],
                                                                       self.trlog['max_acc'][session]))

                    self.trlog['train_loss'].append(tl)
                    self.trlog['train_acc'].append(ta)
                    self.trlog['test_loss'].append(tsl)
                    self.trlog['test_acc'].append(tsa)
                    lrc = scheduler.get_last_lr()[0]
                    result_list.append(
                        'epoch:%03d,lr:%.4f,training_loss:%.5f,training_acc:%.5f,test_loss:%.5f,test_acc:%.5f,base_acc:%.5f,inc_acc:%.5f' % (
                            epoch, lrc, tl, ta, tsl, tsa,va_base,va_new))
                    print('This epoch takes %d seconds' % (time.time() - start_time),
                          '\nstill need around %.2f mins to finish this session' % (
                                  (time.time() - start_time) * (args.epochs_base - epoch) / 60))
                    scheduler.step()



                if not args.not_data_init:
                        
                    self.model.load_state_dict(self.best_model_dict,strict = False)
                    #self.model = update_textproto(train_set, self.model, args)
                    self.model = replace_base_fc(train_set, testloader.dataset.transform, self.model, args)
                    best_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                    print('Replace the fc with average embedding, and save it to :%s' % best_model_dir)
                    self.best_model_dict = deepcopy(self.model.state_dict())
                    torch.save(dict(params=self.model.state_dict()), best_model_dir)
    
                    self.model.module.mode = 'avg_cos'
                    tsl, tsa,va_base, va_new = test(self.model, testloader, 0, args, session)
                    if (tsa * 100) >= self.trlog['max_acc'][session]:
                        self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                        print('The new best test acc of base session={:.3f}'.format(self.trlog['max_acc'][session]))
                result_list.append('Session {}, Test Best Epoch {},\nbest test Acc {:.4f}\n'.format(
                    session, self.trlog['max_acc_epoch'], self.trlog['max_acc'][session], ))

            else:  # incremental learning sessions
                print("training session: [%d]" % session)

                self.model.module.mode = self.args.new_mode
                self.model.eval()
                transform2 = trainloader.dataset.transform
                trainloader.dataset.transform = testloader.dataset.transform
                self.model.module.update_fc(trainloader,transform2, np.unique(train_set.targets), session)

                tsl, tsa,va_base, va_new = test(self.model, testloader, 0, args, session)

                # save model
                self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                self.trlog['base_acc'][session] = float('%.3f' % (va_base * 100))
                self.trlog['new_acc'][session] = float('%.3f' % (va_new * 100))
                save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                torch.save(dict(params=self.model.state_dict()), save_model_dir)
                self.best_model_dict = deepcopy(self.model.state_dict())
                print('Saving model to :%s' % save_model_dir)
                print('  test acc={:.3f}'.format(self.trlog['max_acc'][session]))

                result_list.append('Session {}, test Acc {:.3f}\n'.format(session, self.trlog['max_acc'][session]))
                result_list.append('Session {}, test Acc {:.3f}\n'.format(session, self.trlog['base_acc'][session]))
                result_list.append('Session {}, test Acc {:.3f}\n'.format(session, self.trlog['new_acc'][session]))

        result_list.append('Base Session Best Epoch {}\n'.format(self.trlog['max_acc_epoch']))
        result_list.append(self.trlog['max_acc'])
        print(self.trlog['max_acc'])
        save_list_to_txt(os.path.join(args.save_path, 'results.txt'), result_list)

        t_end_time = time.time()
        total_time = (t_end_time - t_start_time) / 60
        print('Base Session Best epoch:', self.trlog['max_acc_epoch'])
        print('Total time used %.2f mins' % total_time)

    def set_save_path(self):
        mode = self.args.base_mode + '-' + self.args.new_mode
        if not self.args.not_data_init:
            mode = mode + '-' + 'data_init'

        self.args.save_path = '%s/' % self.args.dataset
        self.args.save_path = self.args.save_path + '%s/' % self.args.project

        self.args.save_path = self.args.save_path + '%s-start_%d/' % (mode, self.args.start_session)
        if self.args.schedule == 'Milestone':
            mile_stone = str(self.args.milestones).replace(" ", "").replace(',', '_')[1:-1]
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr_%.4f-MS_%s-Gam_%.2f-Bs_%d-Mom_%.2f' % (
                self.args.epochs_base, self.args.lr_base, mile_stone, self.args.gamma, self.args.batch_size_base,
                self.args.momentum)
        elif self.args.schedule == 'Step':
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr_%.4f-Step_%d-Gam_%.2f-Bs_%d-Mom_%.2f' % (
                self.args.epochs_base, self.args.lr_base, self.args.step, self.args.gamma, self.args.batch_size_base,
                self.args.momentum)
        if 'cos' in mode:
            self.args.save_path = self.args.save_path + '-T_%.2f' % (self.args.temperature)

        if 'ft' in self.args.new_mode:
            self.args.save_path = self.args.save_path + '-ftLR_%.3f-ftEpoch_%d' % (
                self.args.lr_new, self.args.epochs_new)

        if self.args.debug:
            self.args.save_path = os.path.join('debug', self.args.save_path)

        self.args.save_path = os.path.join('checkpoint', self.args.save_path)
        ensure_path(self.args.save_path)
        return None
