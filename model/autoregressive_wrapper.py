import math
from functools import partial
from collections import namedtuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

# structs

Return = namedtuple('Return', ['loss', 'aux_loss', 'is_last_batch'])

# helper functions

def top_p(logits, thres = 0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cum_probs > (1 - thres)
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = 0

    sorted_logits[sorted_indices_to_remove] = float('-inf')
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)

def top_k(logits, thres = 0.9):
    k = int((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

# main class

class AutoregressiveWrapper(nn.Module):
    def __init__(self, net, ignore_index = -100, pad_value = 0):
        super().__init__()
        self.pad_value = pad_value
        self.ignore_index = ignore_index

        self.net = net
        self.seq_len = net.seq_len

    @torch.no_grad()
    def generate(self, start_tokens, seq_len, eos_token = None, temperature = 1., filter_logits_fn = top_k, filter_thres = 0.9, **kwargs):
        was_training = self.net.training
        num_dims = len(start_tokens.shape)

        if num_dims == 1:
            start_tokens = start_tokens[None, :]

        b, t = start_tokens.shape

        self.net.eval()

        out = start_tokens

        # take care of default masking

        full_mask_like = lambda x: torch.full_like(x, True, dtype=torch.bool, device=x.device)

        mask = kwargs.pop('mask', None)
        if mask is None:
            mask = full_mask_like(out)

        # take care of a primed sequence of any length

        mem = None
        *primes, out = out.split(self.seq_len, dim=1)
        *prime_masks, mask = mask.split(self.seq_len, dim=1)

        for prime, prime_mask in zip(primes, prime_masks):
            _, mem, _ = self.net(prime, memories = mem, mask = prime_mask, **kwargs)

        # generate until hit sequence length

        input_len = out.shape[1]

        for _ in range(seq_len):
            logits, mem, aux_loss = self.net(out[:, -input_len:], memories = mem, mask = mask[:, -input_len:], **kwargs)
            logits = logits[:, -1, :]
            filtered_logits = filter_logits_fn(logits, thres = filter_thres)
            probs = F.softmax(filtered_logits / temperature, dim=-1)
            sample = torch.multinomial(probs, 1)

            # unlike most models, inputs start from sequence length of 1 once full sequence length is filled

            out = torch.cat((out, sample), dim=-1)
            mask = F.pad(mask, (0, 1), value=True)

            # append sample to accumulated output            

            input_len = input_len % self.seq_len
            input_len += 1

            if eos_token is not None and (sample == eos_token).all():
                break

        out = out[:, t:]

        if num_dims == 1:
            out = out.squeeze(0)

        self.net.train(was_training)
        return out

    def forward(self, x, max_batch_size = None, return_loss = False, **kwargs):
        pad = partial(pad_sequence, batch_first = True, padding_value = self.pad_value)

        if not return_loss:
            if not isinstance(x, torch.Tensor):
                x = pad(x)
            return self.net(x, **kwargs)

        if isinstance(x, torch.Tensor):
            xi = x[:, :-1]
            xo = x[:, 1:]
        else:
            xi = pad(list(map(lambda t: t[:-1], x)))
            xo = pad(list(map(lambda t: t[1:], x)))

        # help auto-solve an area of confusion around input masks in auto-regressive
        # if user supplies a mask that is only off by one from the source sequence, resolve it for them
        mask = kwargs.pop('mask', None)
        if mask is not None and mask.shape[1] == x.shape[1]:
            mask = mask[:, :-1]

        segment_fn = lambda x: x.split(self.seq_len, dim=1)
        (xi, xo) = map(segment_fn, (xi, xo))

        num_segments = len(xi)
        mask = segment_fn(mask) if mask is not None else ((None,) * num_segments)

        max_batch_size = x.shape[0] if max_batch_size is None else max_batch_size
        split_batch_fn = lambda x: x.split(max_batch_size, dim=0)

        grad_accumulate_every = math.ceil(x.shape[0] / max_batch_size)
        mems = [None] * grad_accumulate_every

        for xi_seg, xo_seg, mask_seg in zip(xi, xo, mask):
            xi_seg, xo_seg = map(split_batch_fn, (xi_seg, xo_seg))
            mask_seg = split_batch_fn(mask_seg) if mask_seg is not None else ((None,) * grad_accumulate_every)

            new_mems = []
            for ind, (xi_seg_b, xo_seg_b, mask_seg_b, mem) in enumerate(zip(xi_seg, xo_seg, mask_seg, mems)):
                is_last = ind == (grad_accumulate_every - 1)

                logits, new_mem, aux_loss = self.net(xi_seg_b, mask = mask_seg_b, memories = mem, **kwargs)
                new_mems.append(new_mem)

                loss = F.cross_entropy(logits.transpose(1, 2), xo_seg_b, ignore_index = self.ignore_index)
                yield Return(loss, aux_loss, is_last)

            mems = new_mems
