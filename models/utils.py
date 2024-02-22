import torch
import torch.nn as nn

import os
import sys
sys.path.append(os.getcwd())
from MyLoss.infonce import InfoNCELoss

def get_queue_feats_and_labels(queues):
    # print(self.get_queue_feats_and_labels)
    cls_feats = []
    cls_labels = []
    for cls_id, cls_feat in enumerate(queues):
        if cls_feat is None:
            continue

        # if cls_id == self.ignore_label:
            # continue

        cls_feats.append(cls_feat)
        # labels = torch.zeros(cls_feat.shape[0], dtype=torch.long, device=cls_feat.device) + cls_id
        q_len = cls_feat.shape[0]
        labels = torch.full((q_len,), cls_id, device=cls_feat.device)
        # print('labels', labels, cls_id)
        cls_labels.append(labels)

    # print('cls_feats inside method:......')
    # print(len(cls_feats))
    # if len(cls_feats) > 0:
    #     print(cls_feats[0].shape, bool(cls_feats))
    if len(cls_feats) > 0:
        cls_feats = torch.cat(cls_feats, dim=0)
        cls_labels = torch.cat(cls_labels, dim=0)
        return cls_feats, cls_labels
    else:
        return None, None








class ConTrasLoss(nn.Module):
    def __init__(self, mem_length, num_classes):
        super().__init__()
        self.mem_length = mem_length
        self.q_len = 200
        self.num_classes = num_classes
        # self.fc = nn.Linear()
        self.nce_loss = InfoNCELoss()
        print(self.nce_loss)
        self._init_queues()

    def _init_queues(self):
        self.queues = [None for _ in range(self.num_classes)]
        self.bg_queue = None

    def clear(self):
        self._init_queues()

    def cls_feats_enqueue(self, feats, labels):
        # print(feats.shape)
        # print(labels)
        for cls_id in range(self.num_classes):

            # feats = cls_feats[cls_labels == cls_id]
            for feat in feats[cls_id ==  labels]:
                # self.add_data(feat, cls_id)
                # print(feat.shape)
                # print(feat.shape)
                self.add_data_fifo(feat, cls_id)


    def add_data_fifo(self, feat, cls_id):
        cls_queue = self.queues[cls_id]
        feat = feat.unsqueeze(dim=0)

        if cls_queue is None:
            # print('mem updated before', mems)
            cls_queue = feat.detach().clone()
        else:
            # mems = torch.cat([mems, h], dim=1)
            cls_queue = torch.cat([cls_queue, feat], dim=0)
            # mems = mems[:, -self.mem_length:, :].detach().clone()
            cls_queue = cls_queue[-self.q_len:, :].detach().clone()
            # print('111', cls_queue.shape, feat.shape)

        self.queues[cls_id] = cls_queue
        # print('after', self.queues[cls_id][:10])
        # print('after', self.queues[cls_id].shape)

    def forward(self, feats, labels, current_len):

        # print(current_len)
        if current_len < self.mem_length:
            # print('current')
            pass
        else:
            cls_feats, cls_labels = get_queue_feats_and_labels(self.queues)
            if cls_feats is not None:
                all_feats = torch.cat([cls_feats, feats], dim=0)
                all_labels = torch.cat([cls_labels, labels], dim=0)

                all_feats = all_feats.unsqueeze(dim=1)
                loss = self.nce_loss(all_feats, all_labels)
                # loss = 0
            else:
                loss = 0

            # print(current_len, cls_feats)
            # print('........')
            # for queue in self.queues:
            #     if queue is not None:
            #         # print(queue.shape)
            #         print(queue[:10])
            #     else:
            #         print(queue)

            # print('.................')

            # if self.queues[0] is not None:
                # print(len(self.queues[0]))

            # print(current_len, len(self.queues))
            self.cls_feats_enqueue(feats, labels)

            return loss
            # return loss


if __name__ == '__main__':

    # pass
    mem_len = 12
    loss_fn = ConTrasLoss(mem_length=mem_len, num_classes=2)

    bs = 8
    # epoch
    for i in range(3):
        # 10 wsis
        for j in range(10):
            print('new wsis')
            # wsis = torch.rand(8, mem_len * 2 * 1024, 3)
            labels = torch.randint(0, 2, (bs,))
            # print(labels, j)
            for k in range(0, mem_len * 2):
            #     chunk = wsis[k: k+1024]
            #     loss()
                feats = torch.rand(bs, 4)
                # print(k, loss.queuesp)
                for queue in loss_fn.queues:
                    if queue is not None:
                        print(k, queue.shape)
                    else:
                        print(k, queue)

                loss =loss_fn(feats, labels, current_len=k)
                print(loss)

                # loss.queues[0]

        loss_fn.clear()