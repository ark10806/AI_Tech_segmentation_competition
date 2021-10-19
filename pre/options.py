import argparse
import os
import torch

class Options():
    """Options class
    Returns:
        [argparse]: argparse containing train and test options
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        # self.parser.add_argument('--dataset', default='casting', help='folder | cifar10 | mnist ')
        self.parser.add_argument('--dataroot', default='/opt/ml/segmentation/input/data', help='path to dataset')
        self.parser.add_argument('--batchsize', type=int, default=2, help='input batch size')
        self.parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
        self.parser.add_argument('--isize', type=int, default=512, help='input image size.')
        self.parser.add_argument('--expr_name', type=str, default='experiment_name', help='name of the experiment')
        self.parser.add_argument('--manualseed', default=-1, type=int, help='manual seed')

        # Train
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--save_image_freq', type=int, default=30, help='frequency of saving real and fake images')
        self.parser.add_argument('--save_weight_freq', type=int, default=50, help='frequency of saving weight file (.pth)')
        self.parser.add_argument('--save_test_images', action='store_true', help='Save test images for demo.')
        self.parser.add_argument('--load_weights', action='store_true', help='Load the pretrained weights')
        self.parser.add_argument('--resume', default='', help="path to checkpoints (to continue training)")
        self.parser.add_argument('--mode', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--niter', type=int, default=40, help='number of epochs to train for')
        self.parser.add_argument('--lr', type=float, default=0.00006, help='initial learning rate for adam')
        self.opt = None

    def parse(self):
        self.opt = self.parser.parse_args()

        if torch.cuda.is_available():
            print(torch.cuda.get_device_name(0))

        args = vars(self.opt)

        file_name = os.path.join('./', self.expr_name, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt