import sys
import numpy as np
import inspect
import importlib
import random
import pandas as pd

#---->
from MyOptimizer import create_optimizer
from MyLoss import create_loss
from MyLoss.infonce import InfoNCELoss
from utils.utils import cross_entropy_torch

#---->
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

#---->
import pytorch_lightning as pl



# mem
class  ModelInterface(pl.LightningModule):
    '''with mems'''

    #---->init
    def __init__(self, model, loss, optimizer, settings, **kargs):
        super(ModelInterface, self).__init__()
        self.save_hyperparameters()
        self.settings = settings
        self.load_model()
        self.loss = create_loss(loss)
        self.optimizer = optimizer
        self.n_classes = model.n_classes
        self.log_path = kargs['log']
        self.ce_loss = nn.CrossEntropyLoss()

        #---->acc
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]

        #---->Metrics
        if self.n_classes > 2:
            self.AUROC = torchmetrics.AUROC(num_classes = self.n_classes, average = 'macro')
            metrics = torchmetrics.MetricCollection([torchmetrics.Accuracy(num_classes = self.n_classes,
                                                                        #    average='micro'),
                                                                        #    average = None,
                                                                           average='macro',
                                                                           task='MULTICLASS'),
                                                    # torchmetrics.AUROC(num_classes = self.n_classes, average = 'macro'),
                                                    torchmetrics.AUROC(num_classes=self.n_classes, average = 'macro', task='MULTICLASS'),

                                                    # torchmetrics.Accuracy(num_classes = self.n_classes,
                                                    #                     #    average = 'micro',
                                                    #                        average = 'macro',
                                                    #                     #    average = None,
                                                    #                        task='MULTICLASS'),
                                                     torchmetrics.CohenKappa(num_classes = self.n_classes),
                                                     torchmetrics.F1(num_classes = self.n_classes,
                                                                     average = 'macro'),
                                                     torchmetrics.Recall(average = 'macro',
                                                                         num_classes = self.n_classes),
                                                     torchmetrics.Precision(average = 'macro',
                                                                            num_classes = self.n_classes),
                                                     torchmetrics.Specificity(average = 'macro',
                                                                            num_classes = self.n_classes)])
        else :
            self.AUROC = torchmetrics.AUROC(num_classes=2, average = 'macro', task='MULTICLASS')
            metrics = torchmetrics.MetricCollection([torchmetrics.Accuracy(num_classes = 2,
                                                                        #    average = 'micro',
                                                                           average = 'macro',
                                                                        #    average = None,
                                                                           task='MULTICLASS'),

                                                    # torchmetrics.AUROC(num_classes = 2, average = 'macro'),
                                                    torchmetrics.AUROC(num_classes=2, average = 'macro', task='MULTICLASS'),
                                                    # torchmetrics.Accuracy(num_classes = 2,
                                                    #                     #    average = 'micro',
                                                    #                        average = 'macro',
                                                    #                     #    average = None,
                                                    #                        task='MULTICLASS'),
                                                     torchmetrics.CohenKappa(num_classes = 2, task='MULTICLASS'),
                                                     torchmetrics.F1Score(num_classes = 2,
                                                                     average = 'macro', task='MULTICLASS'),
                                                     torchmetrics.Recall(average = 'macro',
                                                                         num_classes = 2, task='MULTICLASS'),
                                                     torchmetrics.Precision(average = 'macro',
                                                                            num_classes = 2, task='MULTICLASS')])
        self.valid_metrics = metrics.clone(prefix = 'val_')
        self.test_metrics = metrics.clone(prefix = 'test_')

        # self.valid_metrics.update(torch.rand(1, 2), torch.tensor([1]))
        # print(torch.rand(1, 2).shape)
        # print(torch.tensor([1]).shape)
        # print('hello')
        # import sys;sys.exit()
        #--->random
        self.shuffle = kargs['data'].data_shuffle
        self.count = 0




        self.val_probs = {}
        self.train_probs = {}

        self.val_labels = {}
        self.train_labels = {}


        # added
        self.training_step_outputs = [
            {
                "count": 0,
                "correct": 0,
                # 'f_count': 0,
                # 'f_correct': [],
                # 'f_prob': [],
                # 'label': [],

            } for i in range(self.n_classes)]

        self.validation_step_outputs = []

        # self.valid_results_old = {}
        # self.valid_results_new = {}
        self.mems = None
        # self.mems = []

        self.ignore_label = settings.ignore_label



        # self.queues = []
        # self.seen_examples = []
        # for cls_id in range(settings.num_classes):
        #     # self.queues[cls_id] = None
        #     self.queues.append(None)
        #     self.seen_examples.append(0)
        self._init_queue()


        self.num_classes = settings.num_classes


        # self.bg_min = None
        # self.bg_cont = None
        # self.bg_queue = None
        self.bg_cls = settings.num_classes
        self.queue_len = 200

        self.infonce_loss = InfoNCELoss()

    def _init_queue(self):
        self.bg_queue = None

        self.queues = []
        self.seen_examples = []
        for cls_id in range(self.settings.num_classes):
            # self.queues[cls_id] = None
            self.queues.append(None)
            self.seen_examples.append(0)




    def reservoir(num_seen_examples: int, buffer_size: int) -> int:
        """
        Reservoir sampling algorithm.
        :param num_seen_examples: the number of seen examples
        :param buffer_size: the maximum buffer size
        :return: the target index if the current image is sampled, else -1
        """
        if num_seen_examples < buffer_size:
            return num_seen_examples

        rand = np.random.randint(0, num_seen_examples + 1)
        if rand < buffer_size:
            return rand
        else:
            return -1



    def update_prob_dict(self, prob_dict, prob, slide_id):
        # if prob_dict[slide_id] is None:
            # prob
        for prob, slide_id in zip(prob, slide_id):
            if slide_id not in prob_dict.keys():
                prob_dict[slide_id] = []

            prob_dict[slide_id].append(prob)

    def update_labels(self, slide_id, label, label_dict):
        for s_id, la in zip(slide_id, label):
            # self.labels[s_id] = la
            label_dict[s_id] = la

    #---->remove v_num
    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items



    def cal_bg_loss(self, logits, bg_feats, labels):
        loss = 0

        # predict_labels = torch.zeros(labels.shape, dtype=labels.dtype, device=labels.device)

        # for key, value in self.queues.items():
        # for cls_id in range(self.num_classes)

        real_bg_feats = []
        for cls_id, cls_feat in enumerate(self.queues):
            # if key == self.ignore_label:
            if cls_id == self.ignore_label:
                continue
            # if key == self.bg_cls:
                # continue

            # if memory queue is empty
            if cls_feat is None:
                continue

            # mask = labels == key
            mask = labels == cls_id
            print('1111111', labels, cls_id, mask)

            # feats do not contain cls with label key
            if mask.sum() == 0:
                continue

            # cls_feats = bg_feats[mask]
            sub_bg_feats = bg_feats[mask]
            B, N, dim = cls_feat.shape
            cls_feat = cls_feat.view(B * N, dim)
            sub_bg_logits = logits[mask]

            # compute similarity between bg and cls
            sim = torch.einsum('id, jd->ij', sub_bg_feats, cls_feat)
            sim_prob = torch.sigmoid(sim, dim=1).mean(dim=1)
            # predict_labels[mask]
            soft_prob = torch.stack([sim_prob, 1 - sim_prob], dim=1)

            real_bg_mask = sim_prob < 0.5
            if real_bg_mask.sum() > 0:
                real_bg_feats.append(sub_bg_feats[real_bg_mask])
            # soft_prob = 1 - prob
            # print(prob, soft_prob)
            loss += self.ce_loss(sub_bg_logits, soft_prob)


        if real_bg_feats:
            real_bg_feats = torch.cat(real_bg_feats)
        else:
            real_bg_feats = None
        # return loss, bg_labels
        # import sys; sys.exit()
        # return loss, torch.cat(real_bg_feats)
        return loss, real_bg_feats



    # def bg_enqueue(self, bg_feats, labels):
    def bg_enqueue(self, bg_feats):

        # if queue is empty
        if self.bg_queue is None:
            self.bg_queue = bg_feats
            self.bg_queue = bg_feats[-self.queue_len:].detach().clone()
            return

        # if queue is not full
        if self.bg_enqueue.shape[0] <= self.queue_len:
            self.bg_queue = torch.cat([bg_feats, self.bg_queue], dim=0)
            self.bg_queue = self.bg_queue[-self.queue_len:].detach().clone()
            return



        # get all cls_feats
        cls_feats, _ = self.get_queue_feats_and_labels()


        # if cls_feats is not empty
        if cls_feats:

            # compute the lowest value
            with torch.no_grad():
                all_bg_feats = torch.cat([bg_feats, self.bg_queue], dim=0)
                sim = torch.einsum('bd,nd->bn', all_bg_feats, cls_feats)
                score = torch.softmax(sim, dim=1).mean(dim=1) # b, n

                # in here, the self.bg_queue is full, the code above can process
                # the situation where the queue is not full
                top_indices = torch.topk(score, k=score.shape[0]).indices
                mask = torch.zeros(score.shape, dtype=torch.bool, device=score.device)
                mask[top_indices] = True
                self.bg_queue = all_bg_feats[mask].detach().clone()

                if self.bg_queue.shape[0] != self.queue_len:
                    assert ValueError('something is wrong')


        else:
            # no cls features, we can not compute the similarity
            # of bg and cls features
            # directly push all bg feats into queue
            # perform a FIFO routine

            self.bg_queue = torch.cat([bg_feats, self.bg_queue], dim=0)
            self.bg_queue = self.bg_queue[-self.queue_len:].detach().clone()





    def get_queue_feats_and_labels(self):
        cls_feats = []
        cls_labels = []
        for cls_id, cls_feat in enumerate(self.queues):
            if cls_feat is None:
                break

            if cls_id == self.ignore_label:
                break

            cls_feats.append(cls_feat)
            labels = torch.zeros(cls_feat.shape[0], dtype=torch.long, device=cls_feats.device) + cls_id
            cls_labels.append(labels)

        if cls_feats:
            cls_feats = torch.cat(cls_feats, dim=0)
            cls_labels = torch.cat(cls_labels, dim=0)
            return cls_feats, cls_labels
        else:
            return None, None


    def add_data(self, feat, cls_id):

        index = self.reservoir(self.seen_examples[cls_id], self.queue_len)
        if index >= 0:
            # queue is full
            if index != self.seen_examples[cls_id]:
                self.queues[cls_id][index] = feat

            # queue is not full
            else:
                self.queues[cls_id] = torch.cat([
                        self.queues[cls_id], feat
                    ], dim=0)

        self.seen_examples[cls_id] += 1



    def cls_feats_enqueue(self, cls_feats, cls_labels):
        for cls_id in range(self.num_classes):
            if cls_id == self.ignore_label:
                continue

            feats = cls_feats[cls_labels == cls_id]

            for feat in feats:
                self.add_data(feat, cls_id)






    def con_loss(self, logits, feats, labels):
        """
            logits: bg logits whether belongs to
        """

        # filter out ignored classes
        ignore_mask = labels != self.ignore_label
        print(feats.shape, 'before', labels)
        feats = feats[ignore_mask] # feats are with gradients
        labels = labels[ignore_mask]
        logits = logits[ignore_mask]
        print(feats.shape, 'after', self.ignore_label, labels.shape)


        # current batch contains no avilable samples
        # print(labels.shape)
        if labels.numel() == 0:
            # print('000000000000000000000000000000000')
            return 0

        # return 0
        total_loss = 0
        infonce_loss = 0
        # first compute bg class ce loss
        # if is background class
        print('heihei', self.bg_queue)
        if self.mems[0].shape[1] < self.model.mem_length:
            bg_loss, real_bg_feats = self.cal_bg_loss(logits, feats, labels)


            if real_bg_feats is not None:
                real_bg_labels = torch.zeros(real_bg_labels.shape[0], dtype=labels.dtype, device=labels.device)
                cls_feats, cls_label = self.get_queue_feats_and_labels()
                if cls_feats is not None:
                    all_feats = torch.cat([cls_feats, real_bg_feats], dim=0)
                    all_labels = torch.cat([cls_label, real_bg_labels], dim=0)

                    all_feats = all_feats.unsqueeze(dim=1)
                    infonce_loss += self.infonce(all_feats, all_labels)

                self.bg_enqueue(real_bg_feats)

        else:
            cls_feats, cls_label = self.get_queue_feats_and_labels()
            bg_feats = self.bg_queue
            bg_labels = torch.zeros((bg_feats.shape[0]), device=bg_feats.device, dtype=cls_label.detype)
            all_feats = torch.cat([cls_feats, feats, bg_feats], dim=0)
            all_labels = torch.cat([cls_label, labels, bg_labels], dim=0)
            all_feats = all_feats.unsqueeze(dim=1)
            infonce_loss += self.infonce(all_feats, all_labels)

            self.cls_feats_enqueue(feats, labels)
#
#
        total_loss = infonce_loss + bg_loss

        return total_loss

    def training_step(self, batch, batch_idx):
        #---->inference
        # data, label, slide_id = batch
        data, label, slide_id, is_last = batch
        # print('train_slide', slide_id, 'label', label, 'is_last', is_last)
        # print(label)

        results_dict = self.model(data=data, label=label, mems=self.mems)
        self.mems = results_dict['mems']
        # seen_example = self.mems[0].shape[1]
        # sample = self.sample_feat(feat)
        # if sample = None




        if is_last.sum() > 0:
            self.mems = None
            # self.mems = []

        if isinstance(self.mems, list):
            for mem in self.mems:
                print(mem.shape)
        elif isinstance(self.mems, torch.Tensor):
            print(self.mems.shape)
        else:
            print(self.mems)


        logits = results_dict['logits']
        Y_prob = results_dict['Y_prob']
        Y_hat = results_dict['Y_hat']

        feat = results_dict['feat']
        logits_bg = results_dict['logits_bg']
        # mems = results_dict['mems']
        con_loss = self.con_loss(logits_bg, feat, label)






        print('train_slide', slide_id, 'is_last', is_last, 'Y_prob', Y_prob, 'label', label,)
        #---->loss
        # print(self.loss)
        # import sys; sys.exit()
        loss = self.loss(logits, label) + con_loss

        #---->acc log
        #Y_hat = int(Y_hat)
        #Y = int(label)

        Y_hat = Y_hat
        Y = label

        # print(Y_prob.shape,  label.shape)

        # added
        for y_hat, y in zip(Y_hat, Y):
            self.training_step_outputs[y]["count"] += 1
            self.training_step_outputs[y]["correct"] += (y_hat == y)


        #################################
        self.update_prob_dict(self.train_probs, Y_prob.detach(), slide_id)
        self.update_labels(slide_id, label, self.train_labels)
        #################################
        # print(self.train_probs)
        print('end of training step')

        return {'loss': loss}

    def get_predict_all_avg(self, slide_id, all_probs, last_half):
        # print(self.get_predict_all_avg)
        # print(slide_id)
        # print(all_probs)
        probs = all_probs[slide_id]
        # print(probs.shape)
        # 'test_001': [tensor([0.5319, 0.4681], device='cuda:0')],
        # print(torch.stack(probs, dim=0).shape)
        # print(torch.stack(probs, dim=0))
        if last_half:
            prob_len = len(probs)
            assert prob_len % 2 == 0
            probs = probs[-prob_len:]

        probs = torch.stack(probs, dim=0)
        total_prob = torch.mean(probs, dim=0, keepdim=True)
        # print('total_prob', total_prob)
                # print('total_prob', total_prob / (self.model.mem_length * 2), total_prob.shape)
        # last_prob = torch.cat(probs[-self.model.mem_length:]).sum(dim=0)
            # print('last_prob', last_prob / self.model.mem_length, last_prob.shape)
            # last_correct = torch.sum()
                # print(f_corrects, sum(f_corrects) / len(f_corrects), \
                    #   res_data[c]['label'][count:count + self.model.mem_length * 2])

        # all_c = torch.argmax(total_prob, dim=0)
        # total_prob = total_prob / len(probs)
        # return all_c
        # print('total_prob', total_prob)
        return total_prob

            # last_c = torch.argmax(last_prob, dim=0)


    # def shuffle_dataset(self, dataloader):
    #     dataset = dataloader.dataset
    #     data_len = len(dataset.data)
    #     index = list(range(data_len))
    #     print('shuffle index of training dataset, before shuffle', index)
    #     random.shuffle(index)
    #     print('shuffle index of training dataset, after shuffle', index)
    #     dataset.data.iloc[list(range(data_len))] = dataset.data.loc[index]
    #     dataset.label.iloc[list(range(data_len))] = dataset.label.loc[index]
        # print()

    # def training_epoch_end(self, training_step_outputs):

    def wsi_level_acc(self, probs, labels, metrics, last_half=False):
        # acc = {}
        for slide_id, label in labels.items():
            # print(slide_id, label, self.wsi_level_acc)
            # print(slide_id)
            # print(probs.keys(), slide_id)
            # print(probs)
            # print(labels)
            prob = self.get_predict_all_avg(slide_id, probs, last_half=last_half)
            # self.valid_metrics.update(prob, label.unsqueeze(dim=0))
            metrics.update(prob, label.unsqueeze(dim=0))

            # self.valid_results_new[slide_id] = prob.argmax(dim=1).item()
            # print(prob, slide_id)




            # print(prob)
            # print(label)
            # print(prob.shape, label.unsqueeze(dim=0).shape)
            # print(labels.keys())
            # print('cccccccccccccccc')
            # label = label.unsqueeze(dim=0)

            # self.valid_metrics.update(prob, label)


        # res = self.valid_metrics.compute()
        res = metrics.compute()
        # print(self.valid_metrics.)
        # print('', res)
        # print(res)
        # self.valid_metrics.reset()
        metrics.reset()
        return res




    def on_train_epoch_end(self):

        total_avg = 0
        for c in range(self.n_classes):
            #count = self.data[c]["count"]
            #correct = self.data[c]["correct"]

            count = self.training_step_outputs[c]["count"]
            correct = self.training_step_outputs[c]["correct"]

            if count == 0:
                acc = None
            else:
                acc = float(correct) / count

            total_avg += acc
            print('train epoch class {}: acc {}, correct {}/{}'.format(c, acc, correct, count))

        print('train avg acc', total_avg / self.n_classes)
        # self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
        self.training_step_outputs = [
            {
                "count": 0,
                "correct": 0,
                'f_count':0,
                'f_correct': [],
                'f_prob':[],
                'label':[]
            } for i in range(self.n_classes)]


        # print(self.train_probs)
        res = self.wsi_level_acc(self.train_probs, self.train_labels, self.valid_metrics, last_half=False)
        print('train performance, all {} '.format(res))

        # res = self.wsi_level_acc(self.train_probs, self.train_labels, self.valid_metrics, last_half=True)
        # print('train performance, last half {} '.format(res))

        # clear probs
        self.train_probs = {}

        self._init_queue()


    def validation_step(self, batch, batch_idx):
        # data, label, slide_id = batch
        data, label, slide_id, is_last = batch
        # print('val_slide', slide_id, label)

        with torch.no_grad():
            results_dict = self.model(data=data, label=label, mems=self.mems)
            self.mems = results_dict['mems']
            if is_last.sum() > 0:
                self.mems = None
                # self.mems = []

            if isinstance(self.mems, list):
                for mem in self.mems:
                     print(mem.shape)
            elif isinstance(self.mems, torch.Tensor):
                print(self.mems.shape)
            else:
                print(self.mems)


        logits = results_dict['logits']
        Y_prob = results_dict['Y_prob']
        Y_hat = results_dict['Y_hat']

        # print('val_slide', slide_id, 'is_last', is_last, 'Y_prob', Y_prob, 'label', label,)
        print('val_slide', slide_id, 'is_last', is_last, 'Y_prob', Y_prob, 'label', label,)


        #---->acc log
        Y = label
        for y_hat, y in zip(Y_hat, Y):
            self.data[y]["count"] += 1
            self.data[y]["correct"] += (y_hat.item() == y)

        self.validation_step_outputs.append(
            {'logits' : logits, 'Y_prob' : Y_prob, 'Y_hat' : Y_hat, 'label' : label}
        )

        # self.valid_results_old[slide_id[0]] = Y_hat.item()


        # self.data = [
        #      {
        #          "count": 0,
        #          "correct": 0,
        #         #  'f_count':0,
        #         #  'f_correct': [],
        #         #  'f_prob':[],
        #         #  'label':[]
        #      } for i in range(self.n_classes)]

        self.update_prob_dict(self.val_probs, Y_prob, slide_id)
        self.update_labels(slide_id, label, self.val_labels)

        # print(self.val_probs)

        # print(self.v)

        return {'logits' : logits, 'Y_prob' : Y_prob, 'Y_hat' : Y_hat, 'label' : label}

    # def validation_epoch_end(self, val_step_outputs):
    def on_validation_epoch_end(self):

        val_step_outputs = self.validation_step_outputs

        logits = torch.cat([x['logits'] for x in val_step_outputs], dim = 0)
        probs = torch.cat([x['Y_prob'] for x in val_step_outputs], dim = 0)
        # max_probs = torch.stack([x['Y_hat'] for x in val_step_outputs])
        target = torch.cat([x['label'] for x in val_step_outputs], dim = 0)

        #---->
        self.log('val_loss', cross_entropy_torch(logits, target), prog_bar=True, on_epoch=True, logger=True)
        self.log('auc', self.AUROC(probs, target.squeeze()), prog_bar=True, on_epoch=True, logger=True)
        # self.log_dict(self.valid_metrics(max_probs.squeeze() , target.squeeze()),
        #                   on_epoch = True, logger = True)

        #---->acc log
        total_acc = 0
        for c in range(self.n_classes):
            count = self.data[c]["count"]
            correct = self.data[c]["correct"]
            if count == 0:
                acc = None
            else:
                acc = float(correct) / count

            total_acc += acc
            print('val class {}: acc {}, correct {}/{}'.format(c, acc, correct, count))

        print('avg acc', total_acc / self.n_classes)
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]

        #---->random, if shuffle data, change seed
        if self.shuffle == True:
            self.count = self.count+1
            random.seed(self.count*50)


        self.validation_step_outputs.clear()

        # print(self.val_probs)

        res = self.wsi_level_acc(self.val_probs, self.val_labels, self.test_metrics)
        print('val performance, all {} '.format(res))

        # res = self.wsi_level_acc(self.val_probs, self.val_labels, self.test_metrics, last_half=True)
        # print('val performance, last half {} '.format(res))
        # print('valid_old')
        # for k, v in self.valid_results_old.items():
            # print(k, v, self.valid_results_new[k])
        # print('valid_new', self.valid_results_new)
        # for t1, t2 in zip(self.valid_results_old.items(), self.valid_results_new.items()):
            # print(t1, t2)
        self.val_probs = {}

        # self.valid_results_new = {}
        # self.valid_results_old = {}

    def configure_optimizers(self):
        optimizer = create_optimizer(self.optimizer, self.model)
        return [optimizer]

    def test_step(self, batch, batch_idx):
        data, label = batch
        results_dict = self.model(data=data, label=label)
        logits = results_dict['logits']
        Y_prob = results_dict['Y_prob']
        Y_hat = results_dict['Y_hat']

        #---->acc log
        Y = int(label)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat.item() == Y)

        return {'logits' : logits, 'Y_prob' : Y_prob, 'Y_hat' : Y_hat, 'label' : label}

    def test_epoch_end(self, output_results):
        probs = torch.cat([x['Y_prob'] for x in output_results], dim = 0)
        max_probs = torch.stack([x['Y_hat'] for x in output_results])
        target = torch.stack([x['label'] for x in output_results], dim = 0)

        #---->
        auc = self.AUROC(probs, target.squeeze())
        metrics = self.test_metrics(max_probs.squeeze() , target.squeeze())
        metrics['auc'] = auc
        for keys, values in metrics.items():
            print(f'{keys} = {values}')
            metrics[keys] = values.cpu().numpy()
        print()
        #---->acc log
        for c in range(self.n_classes):
            count = self.data[c]["count"]
            correct = self.data[c]["correct"]
            if count == 0:
                acc = None
            else:
                acc = float(correct) / count
            print('class {}: acc {}, correct {}/{}'.format(c, acc, correct, count))
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
        #---->
        result = pd.DataFrame([metrics])
        result.to_csv(self.log_path / 'result.csv')


    def load_model(self):
        name = self.hparams.model.name
        # Change the `trans_unet.py` file name to `TransUnet` class name.
        # Please always name your model file name as `trans_unet.py` and
        # class name or funciton name corresponding `TransUnet`.
        if '_' in name:
            camel_name = ''.join([i.capitalize() for i in name.split('_')])
        else:
            camel_name = name
        try:
            Model = getattr(importlib.import_module(
                f'models.{name}'), camel_name)
        except:
            raise ValueError('Invalid Module File Name or Invalid Class Name!')
        self.model = self.instancialize(Model, max_len=self.settings.max_len)
        print(self.model)
        pass

    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getargspec(Model.__init__).args[1:]
        inkeys = self.hparams.model.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams.model, arg)
        args1.update(other_args)
        # print('ccccccc')
        print(args1, other_args)
        # import sys; sys.exit()
        return Model(**args1)







# no mem
class  ModelInterface1(pl.LightningModule):
    '''without mem'''

    #---->init
    def __init__(self, model, loss, optimizer, **kargs):
        super(ModelInterface, self).__init__()
        self.save_hyperparameters()
        self.load_model()
        self.loss = create_loss(loss)
        self.optimizer = optimizer
        self.n_classes = model.n_classes
        self.log_path = kargs['log']

        #---->acc
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]

        #---->Metrics
        if self.n_classes > 2:
            self.AUROC = torchmetrics.AUROC(num_classes = self.n_classes, average = 'macro')
            metrics = torchmetrics.MetricCollection([torchmetrics.Accuracy(num_classes = self.n_classes,
                                                                        #    average='micro'),
                                                                        #    average = None,
                                                                           average='macro',
                                                                           task='MULTICLASS'),

                                                    # torchmetrics.Accuracy(num_classes = self.n_classes,
                                                    #                     #    average = 'micro',
                                                    #                        average = 'macro',
                                                    #                     #    average = None,
                                                    #                        task='MULTICLASS'),
                                                     torchmetrics.CohenKappa(num_classes = self.n_classes),
                                                     torchmetrics.F1(num_classes = self.n_classes,
                                                                     average = 'macro'),
                                                     torchmetrics.Recall(average = 'macro',
                                                                         num_classes = self.n_classes),
                                                     torchmetrics.Precision(average = 'macro',
                                                                            num_classes = self.n_classes),
                                                     torchmetrics.Specificity(average = 'macro',
                                                                            num_classes = self.n_classes)])
        else :
            self.AUROC = torchmetrics.AUROC(num_classes=2, average = 'macro', task='MULTICLASS')
            metrics = torchmetrics.MetricCollection([torchmetrics.Accuracy(num_classes = 2,
                                                                        #    average = 'micro',
                                                                           average = 'macro',
                                                                        #    average = None,
                                                                           task='MULTICLASS'),

                                                    # torchmetrics.Accuracy(num_classes = 2,
                                                    #                     #    average = 'micro',
                                                    #                        average = 'macro',
                                                    #                     #    average = None,
                                                    #                        task='MULTICLASS'),
                                                     torchmetrics.CohenKappa(num_classes = 2, task='MULTICLASS'),
                                                     torchmetrics.F1Score(num_classes = 2,
                                                                     average = 'macro', task='MULTICLASS'),
                                                     torchmetrics.Recall(average = 'macro',
                                                                         num_classes = 2, task='MULTICLASS'),
                                                     torchmetrics.Precision(average = 'macro',
                                                                            num_classes = 2, task='MULTICLASS')])
        self.valid_metrics = metrics.clone(prefix = 'val_')
        self.test_metrics = metrics.clone(prefix = 'test_')

        # self.valid_metrics.update(torch.rand(1, 2), torch.tensor([1]))
        # print(torch.rand(1, 2).shape)
        # print(torch.tensor([1]).shape)
        # print('hello')
        # import sys;sys.exit()
        #--->random
        self.shuffle = kargs['data'].data_shuffle
        self.count = 0




        self.val_probs = {}
        self.train_probs = {}

        self.val_labels = {}
        self.train_labels = {}


        # added
        self.training_step_outputs = [
            {
                "count": 0,
                "correct": 0,
                # 'f_count': 0,
                # 'f_correct': [],
                # 'f_prob': [],
                # 'label': [],

            } for i in range(self.n_classes)]

        self.validation_step_outputs = []

        # self.valid_results_old = {}
        # self.valid_results_new = {}
        self.mems = None

        # self.target_class = {
        #         'normal':0,
        #         'cls':1,
        #     }
        # self.queue = {'0': 1}


    def update_prob_dict(self, prob_dict, prob, slide_id):
        # if prob_dict[slide_id] is None:
            # prob
        for prob, slide_id in zip(prob, slide_id):
            if slide_id not in prob_dict.keys():
                prob_dict[slide_id] = []

            prob_dict[slide_id].append(prob)

    def update_labels(self, slide_id, label, label_dict):
        for s_id, la in zip(slide_id, label):
            # self.labels[s_id] = la
            label_dict[s_id] = la

    #---->remove v_num
    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def training_step(self, batch, batch_idx):
        #---->inference
        data, label, slide_id = batch
        # data, label, slide_id, is_last = batch
        # print('train_slide', slide_id, 'label', label, 'is_last', is_last)
        # print(label)

        results_dict = self.model(data=data, label=label, mems=self.mems)
        # self.mems = results_dict['mems']
        # if is_last.sum() > 0:
            # self.mems = None

        # if self.mems is not None:
        #     print('mems', self.mems.shape)
        # else:
        #     print('mems', self.mems)

        logits = results_dict['logits']
        Y_prob = results_dict['Y_prob']
        Y_hat = results_dict['Y_hat']
        # mems = results_dict['mems']

        # print('train_slide', slide_id, 'is_last', is_last, 'Y_prob', Y_prob, 'label', label,)
        print('train_slide', slide_id, 'is_last', 'Y_prob', Y_prob, 'label', label,)
        #---->loss
        loss = self.loss(logits, label)

        #---->acc log
        #Y_hat = int(Y_hat)
        #Y = int(label)

        Y_hat = Y_hat
        Y = label

        # print(Y_prob.shape,  label.shape)

        # added
        for y_hat, y in zip(Y_hat, Y):
            self.training_step_outputs[y]["count"] += 1
            self.training_step_outputs[y]["correct"] += (y_hat == y)


        #################################
        self.update_prob_dict(self.train_probs, Y_prob.detach(), slide_id)
        self.update_labels(slide_id, label, self.train_labels)
        #################################
        # print(self.train_probs)
        print('end of training step')

        return {'loss': loss}

    def get_predict_all_avg(self, slide_id, all_probs, last_half):
        # print(self.get_predict_all_avg)
        # print(slide_id)
        # print(all_probs)
        probs = all_probs[slide_id]
        # print(probs.shape)
        # 'test_001': [tensor([0.5319, 0.4681], device='cuda:0')],
        # print(torch.stack(probs, dim=0).shape)
        # print(torch.stack(probs, dim=0))
        if last_half:
            prob_len = len(probs)
            assert prob_len % 2 == 0
            probs = probs[-prob_len:]

        probs = torch.stack(probs, dim=0)
        total_prob = torch.mean(probs, dim=0, keepdim=True)
        # print('total_prob', total_prob)
                # print('total_prob', total_prob / (self.model.mem_length * 2), total_prob.shape)
        # last_prob = torch.cat(probs[-self.model.mem_length:]).sum(dim=0)
            # print('last_prob', last_prob / self.model.mem_length, last_prob.shape)
            # last_correct = torch.sum()
                # print(f_corrects, sum(f_corrects) / len(f_corrects), \
                    #   res_data[c]['label'][count:count + self.model.mem_length * 2])

        # all_c = torch.argmax(total_prob, dim=0)
        # total_prob = total_prob / len(probs)
        # return all_c
        # print('total_prob', total_prob)
        return total_prob

            # last_c = torch.argmax(last_prob, dim=0)


    # def shuffle_dataset(self, dataloader):
    #     dataset = dataloader.dataset
    #     data_len = len(dataset.data)
    #     index = list(range(data_len))
    #     print('shuffle index of training dataset, before shuffle', index)
    #     random.shuffle(index)
    #     print('shuffle index of training dataset, after shuffle', index)
    #     dataset.data.iloc[list(range(data_len))] = dataset.data.loc[index]
    #     dataset.label.iloc[list(range(data_len))] = dataset.label.loc[index]
        # print()

    # def training_epoch_end(self, training_step_outputs):

    def wsi_level_acc(self, probs, labels, metrics, last_half=False):
        # acc = {}
        for slide_id, label in labels.items():
            # print(slide_id, label, self.wsi_level_acc)
            # print(slide_id)
            # print(probs.keys(), slide_id)
            # print(probs)
            # print(labels)
            prob = self.get_predict_all_avg(slide_id, probs, last_half=last_half)
            # self.valid_metrics.update(prob, label.unsqueeze(dim=0))
            metrics.update(prob, label.unsqueeze(dim=0))

            # self.valid_results_new[slide_id] = prob.argmax(dim=1).item()
            # print(prob, slide_id)




            # print(prob)
            # print(label)
            # print(prob.shape, label.unsqueeze(dim=0).shape)
            # print(labels.keys())
            # print('cccccccccccccccc')
            # label = label.unsqueeze(dim=0)

            # self.valid_metrics.update(prob, label)


        # res = self.valid_metrics.compute()
        res = metrics.compute()
        # print(self.valid_metrics.)
        # print('', res)
        # print(res)
        # self.valid_metrics.reset()
        metrics.reset()
        return res




    def on_train_epoch_end(self):

        total_avg = 0
        for c in range(self.n_classes):
            #count = self.data[c]["count"]
            #correct = self.data[c]["correct"]

            count = self.training_step_outputs[c]["count"]
            correct = self.training_step_outputs[c]["correct"]

            if count == 0:
                acc = None
            else:
                acc = float(correct) / count

            total_avg += acc
            print('train epoch class {}: acc {}, correct {}/{}'.format(c, acc, correct, count))

        print('train avg acc', total_avg / self.n_classes)
        # self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
        self.training_step_outputs = [
            {
                "count": 0,
                "correct": 0,
                'f_count':0,
                'f_correct': [],
                'f_prob':[],
                'label':[]
            } for i in range(self.n_classes)]


        # print(self.train_probs)
        res = self.wsi_level_acc(self.train_probs, self.train_labels, self.valid_metrics, last_half=False)
        print('train performance, all {} '.format(res))

        # res = self.wsi_level_acc(self.train_probs, self.train_labels, self.valid_metrics, last_half=True)
        # print('train performance, last half {} '.format(res))

        # clear probs
        self.train_probs = {}


    def validation_step(self, batch, batch_idx):
        data, label, slide_id = batch
        # data, label, slide_id, is_last = batch
        # print('val_slide', slide_id, label)

        with torch.no_grad():
            results_dict = self.model(data=data, label=label, mems=self.mems)
            # self.mems = results_dict['mems']
            #if is_last.sum() > 0:
            #    self.mems = None

            #if self.mems is not None:
            #    print('mems', self.mems.shape)
            #else:
            #    print('mems', self.mems)

        logits = results_dict['logits']
        Y_prob = results_dict['Y_prob']
        Y_hat = results_dict['Y_hat']

        # print('val_slide', slide_id, 'is_last', is_last, 'Y_prob', Y_prob, 'label', label,)
        # print('val_slide', slide_id, 'is_last', is_last, 'Y_prob', Y_prob, 'label', label,)
        print('val_slide', slide_id, 'Y_prob', Y_prob, 'label', label)


        #---->acc log
        Y = label
        for y_hat, y in zip(Y_hat, Y):
            self.data[y]["count"] += 1
            self.data[y]["correct"] += (y_hat.item() == y)




        self.validation_step_outputs.append(
            {'logits' : logits, 'Y_prob' : Y_prob, 'Y_hat' : Y_hat, 'label' : label}
        )

        # self.valid_results_old[slide_id[0]] = Y_hat.item()


        # self.data = [
        #      {
        #          "count": 0,
        #          "correct": 0,
        #         #  'f_count':0,
        #         #  'f_correct': [],
        #         #  'f_prob':[],
        #         #  'label':[]
        #      } for i in range(self.n_classes)]

        self.update_prob_dict(self.val_probs, Y_prob, slide_id)
        self.update_labels(slide_id, label, self.val_labels)

        # print(self.val_probs)

        # print(self.v)

        return {'logits' : logits, 'Y_prob' : Y_prob, 'Y_hat' : Y_hat, 'label' : label}

    # def validation_epoch_end(self, val_step_outputs):
    def on_validation_epoch_end(self):

        val_step_outputs = self.validation_step_outputs

        logits = torch.cat([x['logits'] for x in val_step_outputs], dim = 0)
        probs = torch.cat([x['Y_prob'] for x in val_step_outputs], dim = 0)
        # max_probs = torch.stack([x['Y_hat'] for x in val_step_outputs])
        target = torch.cat([x['label'] for x in val_step_outputs], dim = 0)

        #---->
        self.log('val_loss', cross_entropy_torch(logits, target), prog_bar=True, on_epoch=True, logger=True)
        self.log('auc', self.AUROC(probs, target.squeeze()), prog_bar=True, on_epoch=True, logger=True)
        # self.log_dict(self.valid_metrics(max_probs.squeeze() , target.squeeze()),
        #                   on_epoch = True, logger = True)

        #---->acc log
        total_acc = 0
        for c in range(self.n_classes):
            count = self.data[c]["count"]
            correct = self.data[c]["correct"]
            if count == 0:
                acc = None
            else:
                acc = float(correct) / count

            total_acc += acc
            print('val class {}: acc {}, correct {}/{}'.format(c, acc, correct, count))

        print('avg acc', total_acc / self.n_classes)
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]

        #---->random, if shuffle data, change seed
        if self.shuffle == True:
            self.count = self.count+1
            random.seed(self.count*50)


        self.validation_step_outputs.clear()

        # print(self.val_probs)

        res = self.wsi_level_acc(self.val_probs, self.val_labels, self.test_metrics)
        print('val performance, all {} '.format(res))

        # res = self.wsi_level_acc(self.val_probs, self.val_labels, self.test_metrics, last_half=True)
        # print('val performance, last half {} '.format(res))
        # print('valid_old')
        # for k, v in self.valid_results_old.items():
            # print(k, v, self.valid_results_new[k])
        # print('valid_new', self.valid_results_new)
        # for t1, t2 in zip(self.valid_results_old.items(), self.valid_results_new.items()):
            # print(t1, t2)
        self.val_probs = {}

        # self.valid_results_new = {}
        # self.valid_results_old = {}

    def configure_optimizers(self):
        optimizer = create_optimizer(self.optimizer, self.model)
        return [optimizer]

    def test_step(self, batch, batch_idx):
        data, label = batch
        results_dict = self.model(data=data, label=label)
        logits = results_dict['logits']
        Y_prob = results_dict['Y_prob']
        Y_hat = results_dict['Y_hat']

        #---->acc log
        Y = int(label)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat.item() == Y)

        return {'logits' : logits, 'Y_prob' : Y_prob, 'Y_hat' : Y_hat, 'label' : label}

    def test_epoch_end(self, output_results):
        probs = torch.cat([x['Y_prob'] for x in output_results], dim = 0)
        max_probs = torch.stack([x['Y_hat'] for x in output_results])
        target = torch.stack([x['label'] for x in output_results], dim = 0)

        #---->
        auc = self.AUROC(probs, target.squeeze())
        metrics = self.test_metrics(max_probs.squeeze() , target.squeeze())
        metrics['auc'] = auc
        for keys, values in metrics.items():
            print(f'{keys} = {values}')
            metrics[keys] = values.cpu().numpy()
        print()
        #---->acc log
        for c in range(self.n_classes):
            count = self.data[c]["count"]
            correct = self.data[c]["correct"]
            if count == 0:
                acc = None
            else:
                acc = float(correct) / count
            print('class {}: acc {}, correct {}/{}'.format(c, acc, correct, count))
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
        #---->
        result = pd.DataFrame([metrics])
        result.to_csv(self.log_path / 'result.csv')


    def load_model(self):
        name = self.hparams.model.name
        # Change the `trans_unet.py` file name to `TransUnet` class name.
        # Please always name your model file name as `trans_unet.py` and
        # class name or funciton name corresponding `TransUnet`.
        if '_' in name:
            camel_name = ''.join([i.capitalize() for i in name.split('_')])
        else:
            camel_name = name
        try:
            Model = getattr(importlib.import_module(
                f'models.{name}'), camel_name)
        except:
            raise ValueError('Invalid Module File Name or Invalid Class Name!')
        self.model = self.instancialize(Model, max_len=self.settings.max_len)
        pass

    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getargspec(Model.__init__).args[1:]
        inkeys = self.hparams.model.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams.model, arg)
        args1.update(other_args)
        return Model(**args1)