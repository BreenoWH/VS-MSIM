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
parser.add_argument("--zeta", type=float, default=3.5, help="The upper bound of neighborhood")
parser.add_argument("--sub_delta", type=float, default=0.5, help="The coefficient")
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


def get_cosine_similarity(cur_grad, sam_grad):
        """
        Calculate cosine similarity to find the score
        """

        cur_grad = cur_grad.view(cur_grad.size(0), -1)
        sam_grad = sam_grad.view(sam_grad.size(0), -1)

        cos_sim = torch.sum(cur_grad * sam_grad, dim=1) / (
                    torch.sqrt(torch.sum(cur_grad ** 2, dim=1)) * torch.sqrt(torch.sum(sam_grad ** 2, dim=1)))
        cos_sim = cos_sim.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        return cos_sim


def save_image(images, names, output_dir):
    """save the adversarial images"""
    if os.path.exists(output_dir) == False:
        os.makedirs(output_dir)

    for i, name in enumerate(names):
        img = Image.fromarray(images[i].astype('uint8'))
        img.save(output_dir + name)

# T_kernel = gkern(7, 3)

def VS(images, gt, model, min, max):
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
    sub_delta = opt.sub_delta
    N = opt.N
    grad = torch.zeros_like(x).detach().cuda()
    for i in range(num_iter):
        avg_grad = torch.zeros_like(x).detach().cuda()
        for _ in range(N):
            x_near = x + torch.rand_like(x).uniform_(-eps * zeta, eps * zeta)
            x_near = V(x_near, requires_grad=True)
            output_v3 = model(x_near)
            loss = F.cross_entropy(output_v3, gt)
            g1 = torch.autograd.grad(loss, x_near,
                                     retain_graph=False, create_graph=False)[0]
            x_star = x_near.detach() + alpha * (-g1) / torch.abs(g1).mean([1, 2, 3], keepdim=True)
            nes_x = x_star.detach()
            x_star_1 = x_near.detach() + alpha * g1 / torch.abs(g1).mean([1, 2, 3], keepdim=True)
            nes_x_1 = x_star_1.detach()
            nes_x = V(nes_x, requires_grad=True)
            output_v3 = model(nes_x)
            loss = F.cross_entropy(output_v3, gt)
            g2 = torch.autograd.grad(loss, nes_x,
                                     retain_graph=False, create_graph=False)[0]
            nes_x_1 = V(nes_x_1, requires_grad=True)
            output_v3 = model(nes_x_1)
            loss = F.cross_entropy(output_v3, gt)
            g3 = torch.autograd.grad(loss, nes_x_1,
                                     retain_graph=False, create_graph=False)[0]

            g_adjust = (1 - sub_delta) * g2 + sub_delta * g3
            avg_grad += (1 - delta) * g1 + delta * g_adjust
        # avg_grad = F.conv2d(avg_grad, T_kernel, bias=None, stride=1, padding=(3, 3), groups=3)
        noise = (avg_grad) / torch.abs(avg_grad).mean([1, 2, 3], keepdim=True)
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

        adv_img = VS(images, gt, model, images_min, images_max)
        adv_img_np = adv_img.cpu().numpy()
        adv_img_np = np.transpose(adv_img_np, (0, 2, 3, 1)) * 255
        save_image(adv_img_np, images_ID, './VS, n=20/')


if __name__ == '__main__':
    main()
