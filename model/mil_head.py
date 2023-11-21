import torch
import torch.nn as nn
import torch.nn.functional as F
# from Model.network import Classifier_1fc



class Attention_Gated(nn.Module):
    # def __init__(self, L=512, D=128, K=1):
    def __init__(self, n_dim=384):
        super(Attention_Gated, self).__init__()

        # self.L = L
        # self.D = D
        # self.K = K

        self.attention_V = nn.Sequential(
            nn.Linear(n_dim, n_dim),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            # nn.Linear(self.L, self.D),
            nn.Linear(n_dim, n_dim),
            nn.Sigmoid()
        )

        #w ∈ R
        #L×1
        #and V ∈ R
        #L×M a
        # self.attention_weights = nn.Linear(self.D, self.K)
        self.attention_weights = nn.Linear(n_dim, 1)

        self.softmax = nn.Softmax(dim=1)

    # def forward(self, x, isNorm=True):
    def forward(self, x, return_score=False):
        ## x: N x L
        attn_v = self.attention_V(x)  # NxD
        attn_u = self.attention_U(x)  # NxD
        # A = self.attention_weights(A_V * A_U) # NxK
        attn_score = self.attention_weights(attn_v * attn_u) # NxK
        attn_score = self.softmax(attn_score)
        z = torch.sum(attn_score * x, dim=1)

        if return_score:
            return z, attn_score
        else:
            return z
        # A = torch.transpose(A, 1, 0)  # KxN
        # print(A.shape, 'ccc')

        # if isNorm:
        # A2 = F.softmax(A, dim=1)  # softmax over N
        # A1 = self.softmax(A)
        # print((A2 -A1).sum())
        # print(A.sum(dim=1))

        # return A  ### K x N

class AttentionHead(nn.Module):
    def __init__(self, n_dim, dis_mem_len, interval) -> None:
        super().__init__()

        self.interval = interval
        self.dis_mem_len = dis_mem_len
        # self.can_mem_len = can_mem_len

        # self.attention_gate = Attention_Gated(n_dim)
        self.dis_mem = []
        self.cand_mem = []

        self.dis_mem_counter = []
        self.cand_mem_counter = []


        self.attention_V = nn.Sequential(
            nn.Linear(n_dim, n_dim),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            # nn.Linear(self.L, self.D),
            nn.Linear(n_dim, n_dim),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(n_dim, 1)

        self.is_switch = False

    # def update_dis_mem(self, index):
    #     # self.dis_mem is empty
    #     # if not self.dis_mem or len(self.dis_mem[0]) < self.dis_mem_len:
    #     if len(self.dis_mem) < 0:
    #         self.dis_mem.append()

    #     for sample_id, idx in enumerate(zip(self.dis_mem, index)):
    #         self.dis_mem[sample_id][idx] += 1


    def in_queue(self, h):
        self.assert_mem_and_counter()
        """h.shape: [b, dim]"""
        # if len(self.dis_mem) == 0:
            # if self.dis_mem is empty
            # for sample_idx in range(h.shape[0]):

        # encode queue

        bs = h.shape[0]

        # dis_mem not full
        if len(self.dis_mem[0])  < self.dis_mem_len:
            for i in range(bs):
                # print('ccccc????')
                # print(len(self.dis_mem[i]), 'before')
                self.dis_mem[i].append(h[i].detach())
                # print(len(self.dis_mem[i]), 'cccc')
                self.dis_mem_counter[i].append(
                    {'total':0, 'min':0}
                )

        # if dis_mem is full but cat_mem is not full
        # elif len(self.dis_mem[0]) == self.dis_mem_len and len(self.dis_mem[0]) < self.can_mem_len:
        #     for i in range(bs):
        #         self.can_mem[i].append(h[i])

        # if both mem is full
        elif len(self.dis_mem[0]) == self.dis_mem_len:
        # and len(self.dis_mem[0]) == self.can_mem_len:
            # if self.is_switch:

            for i in range(bs):
                counters = self.cand_mem_counter[i]
                min_index = self.min_index(counters)
                # if min_index == -1:
                    # continue

                # rethe min mem
                if min_index != -1:
                    # self.cand_mem[i].remove(self.cand_mem[i][min_index])
                    # tmp_mem = self.cand_mem[i][min_index]
                    # print(min_index, i, tmp_mem.dtype)
                    # if min_index == 1:
                    #     print('cc', i, min_index)
                    #     import pickle
                    #     pickle.dump(self.cand_mem, 'test.pkl')
                    # self.cand_mem[i].remove(self.cand_mem[i][min_index])
                    # try:
                        # self.cand_mem[i].remove(tmp_mem)
                    self.cand_mem[i].pop(min_index)
                        # self.cand_mem_counter[i].remove(counters[min_index])
                    self.cand_mem_counter[i].pop(min_index)
                    #except Exception as e:
                    #    print('cc', i, min_index)
                    #    import pickle
                    #    with open('test.pkl', 'wb') as f:
                    #        pickle.dump(self.cand_mem, f)

                        # raise ValueError('...{} '.format(e))


                # insert h
                self.cand_mem[i].append(h[i].detach())
                self.cand_mem_counter[i].append(
                    # {'total': 1, 'min': 0}
                    {'total': 0, 'min': 0}
                )
                # print('????????', len(self.cand_mem[0]), min_index)



    def init_counter(self, counter, bs):
        for _ in range(bs):
            counter.append([])

    def init_queue(self, queue, bs):
        # if len(self.dis_mem) == 0:
        for _ in range(bs):
            queue.append([])


    def attention_score(self, x, dim=None):

        attn_v = self.attention_V(x)
        attn_u = self.attention_U(x)  # NxD
        attn = self.attention_weights(attn_v * attn_u) # NxK
        # print(attn.shape, attn_v.shape, attn_u.shape)
        # print(attn.shape)
        # print(attn.shape, attn_v.shape, attn_u.shape) torch.Size([3, 256, 1]) torch.Size([3, 256, 384]) torch.Size([3, 256, 384])
        attn_score = F.softmax(attn, dim=dim)  # softmax over N

        return attn_score

    # def update_dis_counters(self, x):
    #     with torch.no_grad():



    #     bs = index.shape[0]
    #     for sample_id in range(bs):
    #         mem_id = index[sample_id]
    #         self.dis_mem_counter[sample_id][mem_id]

    def min_index(self, counters):
        """return indexes with highest min prob"""
        assert isinstance(counters, list)
        # for

        # min_prob = 1 # highest prob
        min_prob = 0
        min_index = -1
        for index, cnt in enumerate(counters):
            assert isinstance(cnt, dict)
            # print('cnt', cnt)
            total = cnt['total']
            if self.interval > total:
                continue

            min = cnt['min']
            prob = min / total
            # if prob < min_prob:
            if prob > min_prob:
                min_prob = prob
                min_index = index

        return min_index # index with most times of min index

    def max_index(self, counters):
        """return indexes with lowest min prob"""
        # max_prob = 0
        max_prob = 1
        max_index = -1

        for index, cnt in enumerate(counters):
            total = cnt['total']
            if total < self.interval:
                continue
            min = cnt['min']
            prob = min / total

            # if prob > max_prob:
            if prob < max_prob:
                max_prob = prob
                max_index = index

        return max_index



    def cal_prob(self, counters, index):
        # print('index', index)
        counter = counters[index]
        # print(counter)
        return counter['min'] / counter['total']


    def assert_mem_and_counter(self):
        assert len(self.dis_mem) == len(self.dis_mem_counter)
        for mem, cnt in zip(self.dis_mem, self.dis_mem_counter):
            assert len(mem) == len(cnt)

        assert len(self.cand_mem) == len(self.cand_mem_counter)
        for mem, cnt in zip(self.cand_mem, self.cand_mem_counter):
            assert len(mem) == len(cnt)

    def switch(self):
        self.assert_mem_and_counter()
        if len(self.dis_mem_counter[0]) == self.dis_mem_len:
            bs = len(self.dis_mem_counter)

            # for each sample in batch
            for sample_idx in range(bs):
                counters = self.dis_mem_counter[sample_idx]
                dis_min_index = self.min_index(counters)

                # no suitable mem
                if dis_min_index == -1:
                    continue

                counters = self.cand_mem_counter[sample_idx]
                cand_max_index = self.max_index(counters)

                # no suitable mem
                if cand_max_index == -1:
                    continue

                # print(dis_min_index, cand_max_index)
                dis_prob = self.cal_prob(self.dis_mem_counter[sample_idx], dis_min_index)
                cand_prob = self.cal_prob(self.cand_mem_counter[sample_idx], cand_max_index)
                # print(
                #     'all the dis prob is: {}'.format([cnt['min'] / cnt['total'] for cnt in self.dis_mem_counter[sample_idx]]),
                #     'the min prob: {}  index: {} '.format(dis_prob, dis_min_index),
                #     'all the cand prob is: {}'.format([cnt['min'] / cnt['total'] for cnt in self.cand_mem_counter[sample_idx]]),
                #     'the highest dis prob: {}  index: {}'.format(cand_prob, cand_max_index),
                # )

                # if dis_prob < 1 - cand_prob:
                # if dis_prob < cand_prob:
                if dis_prob > cand_prob:
                    # switch queue
                    # print(len(self.dis_mem), dis_min_index)
                    tmp_mem = self.dis_mem[sample_idx][dis_min_index]
                    self.dis_mem[sample_idx][dis_min_index] = self.cand_mem[sample_idx][cand_max_index]
                    self.cand_mem[sample_idx][cand_max_index] = tmp_mem

                    # switch counters
                    tmp_counter = self.dis_mem_counter[sample_idx][dis_min_index]
                    self.dis_mem_counter[sample_idx][dis_min_index] = self.cand_mem_counter[sample_idx][cand_max_index]
                    self.cand_mem_counter[sample_idx][cand_max_index] = tmp_counter
                    # print('switching......................')
                    # print('switched')
                    # self.is_switch = True
                    # print(
                    #     'switched??'
                    # )

                # print('after.................')
                # print(
                #     '----'
                #     'all the dis prob is: {}'.format([cnt['min'] / cnt['total'] for cnt in self.dis_mem_counter[sample_idx]]),
                #     'the lowest dis prob: {}'.format(dis_prob),
                #     'all the cand prob is: {}'.format([cnt['min'] / cnt['total'] for cnt in self.cand_mem_counter[sample_idx]]),
                #     'the highest dis prob: {}'.format(cand_prob),
                # )







                # switch mem queue
                # tmp = self.dis_mem_counter[dis_min_index]

                # switch counters

    # def

    def assert_cand_mem(self, mem):
        bs = len(self.cand_mem)

        for sample_id in range(bs):
            # print(sampled_id)
            # for sample_id
            cand_mem = self.cand_mem[sample_id]
            for idx, m in enumerate(cand_mem):
                assert torch.equal(mem[sample_id, idx, -1], m)


    def update_all_counters(self):
        # dis_mem = torch.stack(
            # [torch.stack(dis_mem)]
        # )
        self.assert_mem_and_counter()
        bs = len(self.dis_mem)
        dis_mem = torch.stack([
                    torch.stack(mem, dim=0) for mem in self.dis_mem
        ], dim=0)

        # if dis_mem is full
        if len(self.cand_mem[0]):
            with torch.no_grad():
                    cand_mem = torch.stack([
                        torch.stack(mem, dim=0) for mem in self.cand_mem
                    ])
                    cand_len = cand_mem.shape[1]
                    dis_mem = dis_mem.unsqueeze(1)
                    dis_mem = dis_mem.expand(dis_mem.shape[0], cand_len, dis_mem.shape[2], dis_mem.shape[3])
                    cand_mem = cand_mem.unsqueeze(2)
                    mem = torch.cat([dis_mem, cand_mem], dim=2)

                    # self.assert_cand_mem(mem)
                    # compute the attention score for the whole mem
                    # e.g. self.dis_mem = [ 0 , 1, 2] self.cand_mem = [20, 30, 40, 50]
                    # then mem =  [[0, 1, 2, 20]
                    #             [0, 1, 2, 30]
                    #             [0, 1, 2, 40]
                    #             [0, 1, 2, 50]
                    #             ]
                    # print(mem.shape)
                    attn_score = self.attention_score(mem, dim=2)
                    _, min_index = torch.sort(attn_score, dim=2)
                    # print('min_index', min_index.shape)

                    for sample_id in range(bs):

                        # if len(self.cand_mem_counter[sample_id]) > 1:
                            # print(mem[sample_id, 1, -1], self.cand_mem[sample_id][1])

                        # for


                        index = min_index[sample_id]
                        assert index.shape[0] == cand_len
                        # print(index.shape)
                        # for cnt in self.dis_mem_counter[sample_id]:

                        # update min index
                        for idx, m_idx in enumerate(index.tolist()):
                        # for idx, m_idx in enumerate(index):
                            # print(m_idx[0][0])
                            # m_idx.shape   [5, 1] [dis_mem_len + 1, 1]
                            #print('m_idx', m_idx)
                            m_idx = m_idx[0][0] # pick  the  index with lowest prob

                            # if m_idx[0][0] <= self.dis_mem_len - 1:
                            if m_idx <= self.dis_mem_len - 1:
                            #     # print(m_idx[0][0])
                            #     print(self.dis_mem_counter[sample_id])
                                # print(len(self.dis_mem[sample_id]))
                                # print(len(self.dis_mem_counter[sample_id]), m_idx[0][0])
                                # print(m_idx[0])
                                # print(len(self.dis_mem_counter), sample_id, m_idx[0][0])
                                # self.dis_mem_counter[sample_id]
                                # self.dis_mem_counter[sample_id][m_idx[0][0]]['min'] += 1
                                self.dis_mem_counter[sample_id][m_idx]['min'] += 1
                                # assert self.dis_mem_counter[sample_id][m_idx[0][0]]['min'] < \
                                # self.dis_mem_counter[sample_id][m_idx[0][0]]['total']

                            # if m_idx[0][0] > self.dis_mem_len - 1:
                            if m_idx > self.dis_mem_len - 1:
                                self.cand_mem_counter[sample_id][idx]['min'] += 1
                                # assert m_idx == len(self.cand_mem_counter[sample_id])

                            # print(self.cand_mem_counter)


                        # update max index
                        for cnt in self.dis_mem_counter[sample_id]:
                            cnt['total'] += cand_len
                            # print(cnt)
                            assert cnt['total'] >= cnt['min']

                        for cnt in self.cand_mem_counter[sample_id]:
                            cnt['total'] += cand_len
                            # print(cnt)
                            assert cnt['total'] >= cnt['min']
                    # index = min_index[sample_id]
                    # print(len(index))
                #    print(sample_id)



            # print(attn_score[0, 0, :, 0].sum())
            # cand_mem = cand_mem.expand(cand_mem.shape[0])
            # print(dis_mem[0, 1, 0, 1])
            # print(dis_mem[0, 1, 1, 1])
            # print(dis_mem.shape)
            # print(dis_mem.shape)
            # torch.cat([dis_mem, cand_mem], dim=1)


        else:
            # self.dis_mem is not full
            with torch.no_grad():
                attn_score = self.attention_score(dis_mem, dim=1)
                _, index = torch.sort(attn_score, dim=1)
                # print('indx', index.shape)
                # print(index[:, 0].shape)
                # min_index = index[:, 0].to('cpu').tolist()
                min_index = index[:, 0]
                # print(min_index)
                for sample_idx in range(bs):
                    # counters = self.dis_mem_counter
                    for cnt in self.dis_mem_counter[sample_idx]:
                        cnt['total'] += 1
                        if min_index[sample_idx][0] == cnt['min']:
                            cnt['min'] += 1

                        # min_idx = min_index[sample_idx]

        # print(dis_mem.shape, self.cand_mem)

        # cand_len = cand_mem.shape[1]
        # dis_mem = dis_mem.expand(cand_len, dis_mem.shape[2])
        # torch.cat(dis_mem, cand_mem.T, dim=1)


    def reset(self):
        # for mem in self.dis_mem:

        self.dis_mem = []
        self.cand_mem = []

        self.dis_mem_counter = []
        self.cand_mem_counter = []



    def print_counters(self, counters):
        self.assert_mem_and_counter()
        for b_idx, b_counters in enumerate(counters):
            for cnt in b_counters:
                # print('total', cnt['total'], 'min', cnt['min'], end=' ')
                print('batch {}'.format(b_idx), cnt, end='\t')

            print()



    def forward(self, x):

        """input a sequence x with shape [B, len, dim],
            and return a hidden dimension with [B, dim]
        """

        bs = x.shape[0]
        if len(self.dis_mem) == 0:
            self.init_queue(self.dis_mem, bs)
            self.init_counter(self.dis_mem_counter, bs)
        if len(self.cand_mem) == 0:
            self.init_queue(self.cand_mem, bs)
            self.init_counter(self.cand_mem_counter, bs)

        # the first token is cls_token
        cls_token = x[:, 0, :]
        # print(cls_token)
        if len(self.dis_mem[0]):
            # print(torch.cat(torself.dis_mem, dim=))
            # print(torch.tensor(self.dis_mem))
            cls_token = cls_token.unsqueeze(dim=1)
            # print(cls_token.shape, torch.tensor(self.dis_mem).shape)
            # dis_mem = torch.tensor(dis_mem)
            # print(len(self.dis_mem), 'cc', len(self.dis_mem[0]))
            if len(self.dis_mem[0]):
                dis_mem = torch.stack([
                    torch.stack(dis_mem, dim=0) for dis_mem in self.dis_mem
                ], dim=0)
                # print('dis_mem', dis_mem.shape, dis_mem[2, -1])
                bag_level_seq = torch.cat([cls_token, dis_mem], dim=1)
            else:
                bag_level_seq = cls_token
            # print(bag_level_seq.shape)
            # print(bag_level_seq.shape)
        else:
            bag_level_seq = cls_token.unsqueeze(dim=1)



        # attn_v = self.attention_V(bag_level_seq)
        # attn_u = self.attention_U(bag_level_seq)  # NxD
        # attn = self.attention_weights(attn_v * attn_u) # NxK
        # # print(attn.shape, attn_v.shape, attn_u.shape)
        # # print(attn.shape)
        # # print(attn.shape, attn_v.shape, attn_u.shape) torch.Size([3, 256, 1]) torch.Size([3, 256, 384]) torch.Size([3, 256, 384])
        # attn_score = F.softmax(attn, dim=1)  # softmax over N

        # compute attention"
        attn_score = self.attention_score(bag_level_seq, dim=1)
        # compute bag_level representation
        z = torch.sum(attn_score * bag_level_seq, dim=1)
        # z = bag_level_seq[:, 0]

        # print(attn.shape, attn_score.shape, x.shape)

        # print(z.shape)

        # asending
        # sort the last one
        # _, index = torch.sort(attn_score.squeeze(dim=2), dim=1)
        # print(index.shape)
        #print(attn_score[index[0, 0]], attn_score[index[0, 1]])
        # print(attn_score[0, index[0, 0]], attn_score[0, index[0, -1]])
        # print(index.shape)

        # torch.sort(attn_score, dim=1)
        # print(index)
        # the last one index
        # min_index = index[:, 0].to('cpu').tolist()

        # self.update_dis_counters(min_index)
        # switch the nodes between
        # print('before')
        # self.print_counters(self.dis_mem_counter)
        # print('before')
        # self.print_counters(self.dis_mem_counter)
        # print('cand')
        # self.print_counters(self.cand_mem_counter)
        self.switch()
        # self.print_counters(self.cand_mem_counter)
        # print('after')
        # self.print_counters(self.dis_mem_counter)
        # print('cand')
        # self.print_counters(self.cand_mem_counter)
        # self.print_counters(self.dis_mem_counter)


        # self.print_counters(self.cand_mem_counter)
        self.in_queue(z)
        # print('after')
        # self.print_counters(self.dis_mem_counter)
        # self.print_counters(self.cand_mem_counter)
        self.update_all_counters()

        # return z
        # self.update_dis_counters(z)
        # self.update_cat_counters()

        # in_queue

        # print(min_index)
        # print(index)
        # = attn_score[:, min_index, :]
        # for i in min_index:
            # print(i)


        return z





# class Attention_with_Classifier(nn.Module):
#     # def __init__(self, L=512, D=384, K=1, num_cls=2):
#     def __init__(self, n_dim=384, n_cls=2):
#         super(Attention_with_Classifier, self).__init__()
#         # self.attention = Attention_Gated(L, D, K)
#         self.attention = Attention_Gated(n_dim)
#         # self.classifier = Classifier_1fc(L, num_cls, droprate)
#         self.fc = nn.Linear(n_dim, n_cls)

#     def forward(self, x, return_score=False): ## x: N x L
#         attn_score = self.attention(x)  ## K x N
#         print(attn_score.shape, x.shape)
#         z = torch.sum(attn_score * x, dim=1)

#         if return_score:
#             return z, attn_score
#         else:
#             return z

        # print(attn_score.shape, attn_score.shape)
        # afeat = torch.mm(AA, x) ## K x L
        # pred = self.classifier(afeat) ## K x num_cls
        # return pred
        # return attn_score


# net = Attention_with_Classifier()
# input = torch.Tensor(4, 256, 384)

# # z, attn_score = net(input)
# # print(out.shape)
# z = net(input)

# net = AttentionHead(n_dim=384, dis_mem_len=64, interval=100).cuda()
# n_dim=1
# net = AttentionHead(n_dim=n_dim, dis_mem_len=4, interval=8).cuda()

# torch.save(net.state_dict(), 'tensor.pt')
# # net.state_dict() = torch.load(net.state_dict())
# net.load_state_dict(torch.load('tensor.pt'))

# with torch.no_grad():
#  for i in range(50):
#     # img = torch.randn((3, 3, 256, 256))
#     # img = torch.randn(3, 256, 384).cuda()
#     img = torch.arange(3 * 256 * n_dim).float().view(3, 256, n_dim).cuda()
#     # img * 0 + i
#     img += i
#     # print(net.cls_token)
#     out = net(img)

    # out.mean.backward()

    # print(out.shape)
