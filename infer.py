import os
from argparse import Namespace
from tqdm import tqdm
import time
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
import sys

sys.path.append(".")
sys.path.append("..")

from configs import data_configs
from datasets.inference_dataset import InferenceDataset
from options.test_options import TestOptions
from models.psp import pSp
from models.e4e import e4e
from utils.model_utils import ENCODER_TYPES
from utils.common import tensor2im
from utils.inference_utils import run_on_batch, get_average_image


def run():
    test_opts = TestOptions().parse()
    test_opts.exp_dir = './exp_dir/'
    test_opts.data_path = './data_path/'
    out_path_coupled = os.path.join(test_opts.exp_dir, '')
    #out_path_coupled = os.path.join(exp_dir, '')
    os.makedirs(out_path_coupled, exist_ok=True)

    # update test options with options used during training
    test_opts.checkpoint_path = './restyle_e4e_ffhq_encode.pt'
    ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
    #ckpt = torch.load(checkpoint_path, map_location='cpu')
    opts = ckpt['opts']
    opts.update(vars(test_opts))
    opts = Namespace(**opts)
    print(opts)
    if opts.encoder_type in ENCODER_TYPES['pSp']:
        net = pSp(opts)
    else:
        net = e4e(opts)

    net.eval()
    net.cuda()

    print('Loading dataset for {}'.format(opts.dataset_type))
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    dataset = InferenceDataset(root=opts.data_path,
                               transform=transforms_dict['transform_inference'],
                               opts=opts)
    dataloader = DataLoader(dataset,
                            batch_size=opts.test_batch_size,
                            shuffle=False,
                            num_workers=int(opts.test_workers),
                            drop_last=False)

    if opts.n_images is None:
        opts.n_images = len(dataset)

    # get the image corresponding to the latent average
    avg_image = get_average_image(net, opts)

    resize_amount = (256, 256) if opts.resize_outputs else (opts.output_size, opts.output_size)

    global_i = 0
    global_time = []
    for input_batch in tqdm(dataloader):
        if global_i >= opts.n_images:
            break

        with torch.no_grad():
            input_cuda = input_batch.cuda().float()
            tic = time.time()
            result_batch, result_latents = run_on_batch(input_cuda, net, opts, avg_image)
            toc = time.time()
            global_time.append(toc - tic)

        
        for i in range(input_batch.shape[0]):
            results = [tensor2im(result_batch[i][iter_idx]) for iter_idx in range(opts.n_iters_per_batch)]
            im_path = dataset.paths[global_i]
            torch.save(torch.FloatTensor([result_latents[0][-1]]), os.path.join(out_path_coupled, os.path.basename(im_path + '.pt')))
            # save step-by-step results side-by-side
            input_im = tensor2im(input_batch[i])
            res = np.array(results[0].resize(resize_amount))
#            for idx, result in enumerate(results[1:]):
            for idx, result in enumerate(results[-1:]):
                res = np.concatenate([res, np.array(result.resize(resize_amount))], axis=1)
            res = np.concatenate([res, input_im.resize(resize_amount)], axis=1)

            Image.fromarray(res).save(os.path.join(out_path_coupled, os.path.basename(im_path)))

            global_i += 1



    stats_path = os.path.join(opts.exp_dir, 'stats.txt')
    result_str = 'Runtime {:.4f}+-{:.4f}'.format(np.mean(global_time), np.std(global_time))
    print(result_str)

    with open(stats_path, 'w') as f:
        f.write(result_str)


if __name__ == '__main__':
    run()
