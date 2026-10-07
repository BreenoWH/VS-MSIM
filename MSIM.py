"""Implementation of sample attack."""
import os
import torch
from torch.autograd import Variable as V
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision.transforms import ToTensor, ToPILImage, transforms
from tqdm import tqdm
import numpy as np
from PIL import Image
from Normalize import Normalize
from loader import ImageNet
from torch.utils.data import DataLoader
import argparse
import pretrainedmodels
from attack_methods import DI, gkern, Admix
parser = argparse.ArgumentParser()
parser.add_argument('--input_csv', type=str, default='./data/labels.csv', help='Input directory with images.')
parser.add_argument('--input_dir', type=str, default='./data/images', help='Input directory with images.')
parser.add_argument('--output_dir', type=str, default='./outputs/', help='Output directory with adversarial images.')
parser.add_argument('--mean', type=float, default=np.array([0.5, 0.5, 0.5]), help='mean.')
parser.add_argument('--std', type=float, default=np.array([0.5, 0.5, 0.5]), help='std.')
parser.add_argument("--max_epsilon", type=float, default=16.0, help="Maximum size of adversarial perturbation.")
parser.add_argument("--num_iter_set", type=int, default=10, help="Number of iterations.")
parser.add_argument("--image_width", type=int, default=299, help="Width of each input images.")
parser.add_argument("--image_height", type=int, default=299, help="Height of each input images.")
parser.add_argument("--batch_size", type=int, default=20, help="How many images process at one time.")
parser.add_argument("--momentum", type=float, default=1.0, help="Momentum")
parser.add_argument("--N", type=int, default=20, help="The number of sampled examples")
parser.add_argument("--delta", type=float, default=0.5, help="The balanced coefficient")
parser.add_argument("--zeta", type=float, default=3.0, help="The upper bound of neighborhood")
parser.add_argument("--zeta1", type=float, default=1.0, help="sim")
opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

transforms = T.Compose(
    [T.Resize(299), T.ToTensor()]
)


def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result

def MSIM(x):


    return torch.cat([x / ((2 ** i) * torch.rand_like(x).uniform_(0.8, 1.0)) for i in range(5)])

    # return torch.cat([(x + torch.rand_like(x).uniform_(-eps * zeta1, eps * zeta1))
    #                  / ((2 ** i) * torch.rand_like(x).uniform_(1.0, 2.0)) for i in range(5)])




def save_image(images, names, output_dir):
    """save the adversarial images"""
    if os.path.exists(output_dir) == False:
        os.makedirs(output_dir)

    for i, name in enumerate(names):
        img = Image.fromarray(images[i].astype('uint8'))
        img.save(output_dir + name)

# T_kernel = gkern(7, 3)

def MSIM(images, gt, model, min, max):
    """
    The attack algorithm of our proposed CMI-FGSM
    :param images: the input images
    :param gt: ground-truth
    :param model: substitute model
    :param mix: the mix the clip operation
    :param max: the max the clip operation
    :return: the adversarial images
    """
    eps = opt.max_epsilon / 255.0
    num_iter = opt.num_iter_set
    alpha = eps / num_iter
    momentum = opt.momentum
    x = images.clone().detach().cuda()
    zeta = opt.zeta
    delta = opt.delta
    N = opt.N
    grad = torch.zeros_like(x).detach().cuda()
    g_t = torch.cat([torch.cat([gt for _ in range(5)])])
    for i in range(num_iter):
        x = V(x, requires_grad=True)
        noise = torch.zeros_like(x).detach().cuda()
        # for i in torch.arange(5):
        #     nes_x = x / torch.pow(2, i)
        #     output_v3 = model(nes_x)
        #     loss = F.cross_entropy(output_v3, gt)
        #     noise += torch.autograd.grad(loss, x,
        #                                  retain_graph=False, create_graph=False)[0]
        x_sim = MSIM(x)
        output_v3 = model(x_sim)
        loss = F.cross_entropy(output_v3, g_t)
        noise = torch.autograd.grad(loss, x_sim,
                                          retain_graph=False, create_graph=False)[0]
        # noise = noise / 5
        noise1, noise2, noise3, noise4, noise5 = torch.split(noise, x.shape[0], dim=0)
        noise = 1 / 5 * (noise1 + noise2 / 2 + noise3 / 4 + noise4 / 8 + noise5 / 16)


        # noise = F.conv2d(noise, T_kernel, bias=None, stride=1, padding=(3, 3), groups=3)


        noise = noise / torch.abs(noise).mean([1, 2, 3], keepdim=True)
        noise = momentum * grad + noise
        grad = noise

        x = x + alpha * torch.sign(noise)
        x = clip_by_tensor(x, min, max)
    return x.detach()


def main():
    model = torch.nn.Sequential(Normalize(opt.mean, opt.std),
                                pretrainedmodels.inceptionv3(num_classes=1000, pretrained='imagenet').eval().cuda())
    X = ImageNet(opt.input_dir, opt.input_csv, transforms)
    data_loader = DataLoader(X, batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=8)
    for images, images_ID, gt_cpu in tqdm(data_loader):
        gt = gt_cpu.cuda()
        images = images.cuda()
        images_min = clip_by_tensor(images - opt.max_epsilon / 255.0, 0.0, 1.0)
        images_max = clip_by_tensor(images + opt.max_epsilon / 255.0, 0.0, 1.0)

        adv_img = MSIM(images, gt, model, images_min, images_max)
        adv_img_np = adv_img.cpu().numpy()
        adv_img_np = np.transpose(adv_img_np, (0, 2, 3, 1)) * 255
        save_image(adv_img_np, images_ID, './MSIM/')


if __name__ == '__main__':
    main()
