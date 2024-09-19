#!/usr/bin/python
#-*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy, math, pdb, sys, random
import time, os, itertools, shutil, importlib
from tuneThreshold import tuneThresholdfromScore
from DatasetLoader import test_dataset_loader, loadWAV
import pickle
import numpy as np
import time
from tqdm import tqdm
import soundfile


class WrappedModel(nn.Module):

    ## The purpose of this wrapper is to make the model structure consistent between single and multi-GPU

    def __init__(self, model):
        super(WrappedModel, self).__init__()
        self.module = model

    def forward(self, x, x_clean=None, label=None,l2_reg_dict=None, epoch=-1):
        return self.module(x, x_clean, label, epoch=epoch)


class SpeakerNet(nn.Module):

    def __init__(self, model, optimizer, trainfunc, nPerSpeaker, **kwargs):
        super(SpeakerNet, self).__init__()

        SpeakerNetModel = importlib.import_module('models.'+model).__getattribute__('MainModel')
        self.__S__ = SpeakerNetModel(**kwargs);

        LossFunction = importlib.import_module('loss.'+trainfunc).__getattribute__('LossFunction')
        self.__L__ = LossFunction(**kwargs);

        self.nPerSpeaker = nPerSpeaker
        self.weight_finetuning_reg = kwargs['weight_finetuning_reg']


    def forward(self, data, data_clean=None, label=None, l2_reg_dict=None, epoch=-1):
        if label is None:
            data_reshape = data[0].cuda()
            outp = self.__S__.forward([data_reshape, data[1]])
            return outp
        elif len(data) == 3 and data[2] == "gen_ps":
            data_reshape = data[0].reshape(-1,data[0].size()[-1]).cuda()
            outp = self.__S__.forward([data_reshape, data[1]])
            pseudo_labels = self.__L__.get_pseudo_labels(outp, label)
            return pseudo_labels
        else:
            data_reshape = data[0].reshape(-1,data[0].size()[-1]).cuda()
            data_clean_reshape = data_clean.reshape(-1,data_clean.size()[-1]).cuda()
            outp = self.__S__.forward([data_reshape, data[1]])
            outp_clean = self.__S__.forward([data_clean_reshape, data[1]])
            nloss, prec1, ce = self.__L__.forward(outp, outp_clean, label, epoch)

            if l2_reg_dict is not None:
                Learned_dict = l2_reg_dict
                l2_reg = 0
                for name,param in self.__S__.model.named_parameters():
                    if name in Learned_dict:
                        l2_reg = l2_reg + torch.norm(param-Learned_dict[name].cuda(),2)
                tloss = nloss/nloss.detach() + self.weight_finetuning_reg*l2_reg/(l2_reg.detach()+1e-5)
            else:
                tloss = nloss
                print("Without L2 Reg")

            return tloss, prec1, nloss, ce




class ModelTrainer(object):

    def __init__(self, speaker_model, optimizer, scheduler, gpu, mixedprec, **kwargs):

        self.__model__  = speaker_model

        WavLM_params = list(map(id, self.__model__.module.__S__.model.parameters()))
        Backend_params = filter(lambda p: id(p) not in WavLM_params, self.__model__.module.parameters())   
        self.path = kwargs['pretrained_model_path']

        Optimizer = importlib.import_module('optimizer.'+optimizer).__getattribute__('Optimizer')

        # Define the initial param groups
        param_groups = [{'params': Backend_params, 'lr': kwargs['LR_MHFA']}]

        # Extract the encoder layers
        encoder_layers = self.__model__.module.__S__.model.encoder.layers

        # Iterate over the encoder layers to create param groups
        for i in range(12):  # Assuming 12 layers from 0 to 11 (for BASE model, when it comes to LARGE model, 12->24)
            lr = kwargs['LR_Transformer'] * (kwargs['LLRD_factor'] ** i)
            param_groups.append({'params': encoder_layers[i].parameters(), 'lr': lr})

        # Initialize the optimizer with these param groups
        self.__optimizer__ = Optimizer(param_groups, **kwargs)

        # self.__optimizer__ = Optimizer(self.__model__.parameters(), **kwargs)
        # print('scheduler.'+scheduler)
        Scheduler = importlib.import_module('scheduler.'+scheduler).__getattribute__('Scheduler')
        # print(kwargs)
        try:
            self.__scheduler__, self.lr_step = Scheduler(self.__optimizer__, **kwargs)
        except:
            self.__scheduler__, self.lr_step = Scheduler(self.__optimizer__, lr_decay=0.9, **kwargs)

        # self.scaler = GradScaler() 

        self.gpu = gpu

        self.mixedprec = mixedprec
        print("Mix prec: %s"%(self.mixedprec))

        assert self.lr_step in ['epoch', 'iteration']

    # ## ===== ===== ===== ===== ===== ===== ===== =====
    # ## Train network
    # ## ===== ===== ===== ===== ===== ===== ===== =====
    
    def update_lgl_threshold(self, lgl_threshold):
        self.__model__.module.__L__.lgl_threshold = lgl_threshold
    
    # """
    def train_network(self, loader, loss_vals_path, epoch, verbose):
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            unique_loss_vals_path = f"{loss_vals_path.split('.')[0]}_rank{rank}.txt"
        else:
            unique_loss_vals_path = loss_vals_path
        
        self.__model__.train();

        stepsize = loader.batch_size;

        counter = 0;
        index   = 0;
        loss    = 0;
        top1    = 0     # EER or accuracy

        tstart = time.time()
        Learned_dict = {}
        checkpoint = torch.load(self.path)
        for name, param in checkpoint['model'].items():
            if 'w2v_encoder.w2v_model.' in name:
                newname = name.replace('w2v_encoder.w2v_model.', '')
            else:
                newname = name
            Learned_dict[newname] = param;

        # for data_clean, data, data_label, data_path in loader:
        #     telapsed = time.time() - tstart
        #     tstart = time.time()
        #     counter += 1;
        #     index   += stepsize
        #     sys.stdout.write("\rProcessing (%d) "%(index));
        #     sys.stdout.write("Loss %f TEER/TAcc %2.3f%% - %.2f Hz "%(loss/counter, top1/counter, stepsize/telapsed));
        #     if counter % 100 == 0:
        #         sys.stdout.flush()
            
        with open(unique_loss_vals_path, 'w') as loss_vals_file:
            for data_clean, data, data_label, data_path in loader:
                data_clean = data_clean.transpose(1,0)
                data = data.transpose(1,0)
                self.__model__.zero_grad()
                label   = torch.LongTensor(data_label).cuda()

                nloss, prec1, spkloss, ce = self.__model__([data,"train"], data_clean, label, Learned_dict, epoch=epoch)
                
                for ce_val, path in zip(ce.detach().cpu().numpy(), data_path):
                    loss_vals_file.write(f'{ce_val} {"/".join(path.split("/")[5:])}\n')
                
                nloss.backward()

                self.__optimizer__.step();

                loss    += spkloss.detach().cpu()
                top1    += prec1.detach().cpu()
                

                counter += 1;
                index   += stepsize;

            

                telapsed = time.time() - tstart
                tstart = time.time()

                if verbose:
                    sys.stdout.write("\rProcessing (%d) "%(index));
                    sys.stdout.write("Loss %f TEER/TAcc %2.3f%% - %.2f Hz "%(loss/counter, top1/counter, stepsize/telapsed));
                    sys.stdout.flush();

                if self.lr_step == 'iteration': self.__scheduler__.step()
            
        if self.lr_step == 'epoch': self.__scheduler__.step()

        sys.stdout.write("\n");
        return (loss/counter, top1/counter);
    # """

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Evaluate from list
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def evaluateFromList(self, test_list, test_path, nDataLoaderThread, print_interval=10, num_eval=15, **kwargs):
        
        self.__model__.eval();
        
        lines       = []
        files       = []
        feats       = {}
        tstart      = time.time()

        ## Read all lines
        with open(test_list) as f:
            lines = f.readlines()
        
        ## Get a list of unique file names
        files = sum([x.strip().split()[-2:] for x in lines],[])
        setfiles = list(set(files))
        setfiles.sort()

        ## Define test data loader
        test_dataset = test_dataset_loader(setfiles, test_path, num_eval=num_eval, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=nDataLoaderThread,
            drop_last=False,
        )
        ref_feat_list = []
        ref_feat_2_list = []
        max_len = 0
        forward = 0
        ## Extract features for every image
        for idx, data in enumerate(test_loader):
            

            inp1                = data[0][0].cuda()
            inp2                = data[1][0].cuda()
            telapsed_2 = time.time() 
            b,utt_l = inp2.shape
            if utt_l > max_len:
                max_len = utt_l
            ref_feat            = self.__model__([inp1, "test"]).cuda()
            ref_feat = ref_feat.detach().cpu()
            ref_feat_2            = self.__model__([inp2[:,:700000], "test"]).cuda() # The reason why here is set to 700000 is due to GPU memory size.
            ref_feat_2 = ref_feat_2.detach().cpu()

            feats[data[2][0]]   = [ref_feat,ref_feat_2]
            
            ref_feat_list.extend(ref_feat.numpy())
            ref_feat_2_list.extend(ref_feat_2.numpy())

            telapsed = time.time() - tstart
            forward = forward + time.time() - telapsed_2

            if idx % print_interval == 0:
                sys.stdout.write("\rReading %d of %d: %.2f Hz, forward speed: %.2f Hz, embedding size %d, max_len %d"%(idx,len(setfiles),idx/telapsed,idx/forward, ref_feat.size()[-1],max_len));

        print('')
        all_scores = [];
        all_labels = [];
        all_trials = [];
        all_scores_1 = [];        
        all_scores_2 = [];

        tstart = time.time()

        ref_feat_list = numpy.array(ref_feat_list)
        ref_feat_2_list = numpy.array(ref_feat_2_list)

        ref_feat_list_mean = 0
        ref_feat_2_list_mean  = 0


        ## Read files and compute all scores
        for idx, line in enumerate(lines):

            data = line.split();

            ## Append random label if missing
            if len(data) == 2: data = [random.randint(0,1)] + data

            ref_feat,ref_feat_2 = feats[data[1]]
            com_feat,com_feat_2 = feats[data[2]]

            # if self.__model__.module.__L__.test_normalize:
            ref_feat = F.normalize(ref_feat-ref_feat_list_mean, p=2, dim=1) # B, D
            com_feat = F.normalize(com_feat-ref_feat_list_mean, p=2, dim=1)
            ref_feat_2 = F.normalize(ref_feat_2-ref_feat_2_list_mean, p=2, dim=1) # B, D
            com_feat_2 = F.normalize(com_feat_2-ref_feat_2_list_mean, p=2, dim=1)

            score_1 = torch.mean(torch.matmul(ref_feat, com_feat.T)) # higher is positive
            score_2 = torch.mean(torch.matmul(ref_feat_2, com_feat_2.T))
            score = (score_1 + score_2) / 2
            score = score.detach().cpu().numpy()

            all_scores.append(score);  
            all_scores_1.append(score_1);
            all_scores_2.append(score_2);

            all_labels.append(int(data[0]));
            all_trials.append(data[1]+" "+data[2])

            if idx % (10*print_interval) == 0:
                telapsed = time.time() - tstart
                sys.stdout.write("\rComputing %d of %d: %.2f Hz"%(idx,len(lines),idx/telapsed));
                sys.stdout.flush();

        print('')

        return (all_scores, all_labels, all_trials,all_scores_1,all_scores_2);
    
    def generate_embeddings(self, wav_files, output, device):
        res = {}

        for file in tqdm(wav_files):
            wav, sr = soundfile.read(file)
            wav = torch.from_numpy(wav).float().to(device)

            with torch.no_grad():
                embedding = self.__model__([wav.unsqueeze(0), "test"]).detach().cpu()
            
            key = '/'.join(file.split('/')[-3:])
            res[key] = embedding

        torch.save(res, output)

    def saveParameters(self, path):
        torch.save(self.__model__.module.state_dict(), path);


    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Load parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def loadParameters(self, path):

        self_state = self.__model__.module.state_dict();
        loaded_state = torch.load(path, map_location="cuda:%d"%self.gpu);
        # loaded_state = torch.load(path, map_location="cpu");



        for name, param in loaded_state.items():
            origname = name;

            if name not in self_state:
                name = name.replace("module.", "");

                if name not in self_state:
                    print("%s is not in the model."%origname);
                    continue;

            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()));
                continue;

            self_state[name].copy_(param);
            




