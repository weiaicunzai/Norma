
import os
import torch.distributed as dist
import numpy as np
import torch




# def cycle(iterable):
#     while True:
#         for data in iterable:
#             yield data


def init_process():
    """ Initialize the distributed environment. """
    rank = int(os.environ['LOCAL_RANK'])
    size = int(os.environ['LOCAL_WORLD_SIZE'])
    # os.environ['MASTER_PORT'] = '29500'
    # dist.init_process_group(backend, rank=rank, world_size=size)
    # print(size, '............................', os.environ["CUDA_VISIBLE_DEVICES"])
    # dist.init_process_group('nccl', rank=rank, world_size=size)
    dist.init_process_group('gloo', rank=rank, world_size=size)
    # fn(rank, size)



def compute_mean_and_std(dataset):
    """Compute dataset mean and std, and normalize it
    Args:
        dataset: instance of torch.nn.Dataset
    Returns:
        return: mean and std of this dataset
    """

    mean = 0
    std = 0

    count = 0
    print(len(dataset), 11)
    # for img, _ in dataset:
    for img in dataset:
        img = img['img']
        mean += np.mean(img, axis=(0, 1))
        count += 1

    mean /= len(dataset)

    diff = 0
    # for img, _ in dataset:
    for img in dataset:
        img = img['img']

        diff += np.sum(np.power(img - mean, 2), axis=(0, 1))

    # N = len(dataset) * np.prod(img.shape[:2])
    N = len(dataset) * np.prod(img.size)
    std = np.sqrt(diff / N)

    mean = mean / 255
    std = std / 255

    return mean, std





@torch.no_grad()
def visualize_network(writer, net, tensor):
    tensor = tensor.to(next(net.parameters()).device)
    writer.add_graph(net, tensor)

def _get_lastlayer_params(net):
    """get last trainable layer of a net
    Args:
        network architectur

    Returns:
        last layer weights and last layer bias
    """
    last_layer_weights = None
    last_layer_bias = None
    for name, para in net.named_parameters():
        if 'weight' in name:
            if para.grad is not None:
                last_layer_weights = para
        if 'bias' in name:
            if para.grad is not None:
                last_layer_bias = para

    return last_layer_weights, last_layer_bias

# def visualize_metric(writer, metrics, values, n_iter):
#     for m, v in zip(metrics, values):
#         writer.add_scalar('{}'.format(m), v, n_iter)

def visualize_metric(writer, name, val, n_iter):
    # for m, v in zip(metrics, values):
    writer.add_scalar('{}'.format(name), val, n_iter)

def visualize_lastlayer(writer, net, n_iter):
    weights, bias = _get_lastlayer_params(net)
    writer.add_scalar('LastLayerGradients/grad_norm2_weights', weights.grad.norm(), n_iter)
    writer.add_scalar('LastLayerGradients/grad_norm2_bias', bias.grad.norm(), n_iter)

def visualize_scalar(writer, name, scalar, n_iter):
    """visualize scalar"""
    writer.add_scalar(name, scalar, n_iter)

def visualize_param_hist(writer, net, n_iter):
    """visualize histogram of params"""
    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        #writer.add_histogram("{}/{}".format(layer, attr), param.detach().cpu().numpy(), n_iter)
        writer.add_histogram("{}/{}".format(layer, attr), param, n_iter)



def draw_attn_score(attn_score, p_label, min_index, cell_size):

    # from darker to lighter
    greens = [
        [59, 95, 33],
        [88, 142, 49],
        [172, 215,142],
        [200, 229, 179],
        [227, 242, 217],
    ][::-1]

    blues =  [[30, 56, 107],
            [46, 84, 161],
            [145, 172, 224],
            [182, 199, 234],
            [218, 227, 245]
        ][::-1]
    # blues = [
    #     [30, 56, 107],
    #     [46, 84, 161],
    #     [145, 127, 224],
    #     [182, 199, 234],
    #     [218, 227, 245]
    # ][::-1]
    with torch.no_grad():
        attn_map = torch.repeat_interleave(attn_score, repeats=cell_size, dim=1)
        attn_map = torch.repeat_interleave(attn_map, repeats=cell_size, dim=0)

        attn_map = attn_map * 100
        attn_map = attn_map.long()
        # print(attn_map.shape)
        # print(attn_map)
        # print(attn_sc)
        p_label = torch.repeat_interleave(p_label, repeats=cell_size, dim=1)
        p_label = torch.repeat_interleave(p_label, repeats=cell_size, dim=0)

        # attn_map = attn_map.numpy()
        # p_label = p_label.numpy()


        h, w = attn_map.shape[:2]
        # img = np.zeros((h, w, 3), dtype=np.uint8)
        img = torch.zeros((h, w, 3), dtype=torch.uint8)
        # img = img + [0, 0, 255]
        # img = img.astype(np.uint8)
        # print(img.shape)
        # print(p_label)


        for i in range(5):

            # mask = np.logical_and(np.logical_and(p_label == 0, i * 20 < attn_map), attn_map <= (i + 1) * 20)
            mask = torch.logical_and(torch.logical_and(p_label == 0, i * 20 < attn_map), attn_map <= (i + 1) * 20)
            if i == 0:
                # tmp_mask = np.logical_and(p_label == 0,  attn_map == 0)
                tmp_mask = torch.logical_and(p_label == 0,  attn_map == 0)
                # mask = np.logical_or(tmp_mask, mask)
                mask = torch.logical_or(tmp_mask, mask)
            # print(i, i * 20, (i + 1) * 20, attn_map[mask])

            img[mask] = torch.tensor(blues[i], dtype=img.dtype)


        for i in range(5):

            # mask = np.logical_and(np.logical_and(p_label == 1, i * 20 < attn_map), attn_map <= (i + 1) * 20)
            mask = torch.logical_and(torch.logical_and(p_label == 1, i * 20 < attn_map), attn_map <= (i + 1) * 20)
            if i == 0:
                # tmp_mask = np.logical_and(p_label == 1,  attn_map == 0)
                tmp_mask = torch.logical_and(p_label == 1,  attn_map == 0)
                # mask = np.logical_or(tmp_mask, mask)
                mask = torch.logical_or(tmp_mask, mask)
            # print(i, i * 20, (i + 1) * 20, attn_map[mask])

            img[mask] = torch.tensor(greens[i], dtype=img.dtype)


        # for i in range(4):
            # mask = (p_label)
        # for i in range(5):
        #     # mask = (p_label == 1) & (i * 20 < attn_map < (i + 1) * 20)
        #     mask = np.logical_and(np.logical_and(p_label == 1, i * 20 < attn_map), attn_map < (i + 1) * 20)
        #     img[mask] = greens[i]



        # seq_len = attn_map.shape[1] / cell_size


        # print(attn_map)
        if min_index is not None:
            for idx, m_idx in enumerate(min_index):
            #    img[idx * cell_size:(idx + 1) * cell_size, m_idx * cell_size:(m_idx + 1)* cell_size ] = (0, 0, 255)
               # draw top line
            #    print(idx * cell_size, m_idx * cell_size, 1111)
            #    print((idx + 1) * cell_size, m_idx * cell_size, (m_idx + 1)* cell_size)
            #    print(idx, m_idx)
               # top line
               red = torch.tensor((255, 0, 0), dtype=img.dtype)
               img[idx * cell_size, m_idx * cell_size:(m_idx + 1)* cell_size] = red
            #    print((idx + 1) * cell_size, m_idx * cell_size, (m_idx + 1)* cell_size,  idx, cell_size)
            #    print(idx, cell_size, img.shape)
            #    img[(idx + 1) * cell_size - 1, m_idx * cell_size:(m_idx + 1)* cell_size] = (255, 0, 0)
               img[(idx + 1) * cell_size - 1, m_idx * cell_size:(m_idx + 1)* cell_size] = red

               img[idx * cell_size:(idx + 1) * cell_size, m_idx * cell_size] = red
               img[idx * cell_size:(idx + 1) * cell_size, (m_idx + 1)* cell_size - 1] = red

        #cv2.cvtColor()
        # print(
            # '117,189,66'
        # )
        # print(attn_map.shape)





        return img










# import cv2

# attn_score = torch.rand(16, 10)
# attn_score = torch.load('attn_score.pt')
# print(attn_score)
# min_idx = attn_score.min(dim=1)[1]
# p_label = torch.randint(low=0, high=2, size=attn_score.shape)
# print(p_label)

# img = draw_attn_score(attn_score=attn_score, p_label=p_label, min_index=min_idx, cell_size=5)

# cv2.imwrite('aa.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
