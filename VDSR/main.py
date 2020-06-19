import argparse, os
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import time, math
import matplotlib.pyplot as plt

from model import VDSR
from util import *

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--batchSize", type=int, default=128, help="Training batch size")
    parser.add_argument("--Epochs", type=int, default=50, help="Number of epochs to train for")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning Rate. Default=0.1")
    parser.add_argument("--step", type=int, default=10,
                        help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
    parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
    parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
    parser.add_argument("--cuda", action="store_true", help="Use cuda?")
    parser.add_argument("--clip", type=float, default=0.4, help="Clipping Gradients. Default=0.4")
    parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1")
    parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
    parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="Weight decay, Default: 1e-4")
    parser.add_argument('--pretrained', default='', type=str, help='path to pretrained model (default: none)')

    args = parser.parse_args()

    cuda = args.cuda
    if cuda:
        print("=> use gpu id: '{}'".format(0))
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    args.seed = random.randint(1, 10000)
    print("Random Seed: ", args.seed)
    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed(args.seed)

    cudnn.benchmark = True

    print("===> Loading datasets")
    train_set = prepareDataset("data/train.h5")
    train_data = DataLoader(dataset=train_set, num_workers=args.threads, batch_size=args.batchSize, shuffle=True)

    print("===> Building model")
    model = VDSR()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = nn.MSELoss(size_average=False)
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading model '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained)
            args.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint['model'].state_dict())
        else:
            print("=> no model found at '{}'".format(opt.pretrained))

    train(train_data, optimizer, model, criterion, args.Epochs, args)
    eval(model)


def train(dataloader, optimizer, model, criterion, Epoch, args):
    for epoch in range(1, Epoch+1):

        for i, batch in enumerate(dataloader, 1):
            input, target = Variable(batch[0]), Variable(batch[1], requires_grad=False)

            if args.cuda:
                input = input.cuda()
                target = target.cuda()

            loss = criterion(model(input), target)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            if i % 100 == 0:
                print("===> Epoch[{}]({}/{}): Loss: {:.10f}".format(epoch, i, len(dataloader), loss.data[0]))

        if epoch % 2 == 0:
            save_checkpoint(model, epoch)

def eval( model, ):
    im_gt = Image.open("Set5/butterfly_GT.bmp").convert("RGB")
    im_b = Image.open("Set5/butterfly_GT_scale_4.bmp").convert("RGB")
    # Convert the images into YCbCr mode and extraction the Y channel (for PSNR calculation)
    im_gt_ycbcr = np.array(im_gt.convert("YCbCr"))
    im_b_ycbcr = np.array(im_b.convert("YCbCr"))
    im_gt_y = im_gt_ycbcr[:, :, 0].astype(float)
    im_b_y = im_b_ycbcr[:, :, 0].astype(float)

    im_b = Variable(torch.from_numpy(im_b_y/255.).float())
    im_b = im_b.view(1, -1, im_b.shape[0], im_b.shape[1]).cuda()

    start_time = time.time()
    out = model(im_b)
    print("Time taken: ", time.time()-start_time)

    out = out.cpu()
    out_y = out.data[0].numpy().astype(np.float32)
    out_y = out_y * 255.
    out_y[out_y > 255.] = 255.
    out_y[out_y < 0] = 0

    psnr_score = computePSNR(im_gt_y, out_y)
    print(" PSNR score for our predicted image is ", psnr_score)

    out_img = colorize(out_y, im_b_ycbcr)
    im_gt = Image.fromarray(im_gt_ycbcr, "YCbCr").convert("RGB")
    im_b = Image.fromarray(im_b_ycbcr, "YCbCr").convert("RGB")

    ax = plt.subplot("132")
    ax.imshow(im_b)
    ax.set_title("Input(bicubic)")

    ax = plt.subplot("133")
    ax.imshow(out_img)
    ax.set_title("Output(vdsr)")
    plt.show()



if __name__ == '__main__':
    main()