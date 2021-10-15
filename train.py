import argparse
import time
import json
import os
from tqdm import tqdm
from models import *
from efficientnet_pytorch import EfficientNet
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import *
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from tools import warmup_lr


# 初始化参数
def get_args():
    parser = argparse.ArgumentParser(description='基于Pytorch实现的分类任务')
    '''在下面初始化你的参数'''
    # exp
    parser.add_argument('--time_exp_start', type=str,
                        default=time.strftime('%m-%d-%H-%M', time.localtime(time.time())))
    parser.add_argument('--train_dir', type=str,
                        default='E:/experiments/DEEPLEARNING/classification/data/CUB_200_2011/images/')
    parser.add_argument('--val_dir', type=str,
                        default='E:/experiments/DEEPLEARNING/classification/data/CUB_200_2011/val/')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--save_station', type=int, default=80)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_classes', type=int, default=200)
    # dataset
    parser.add_argument('--data_mean', type=tuple, default=(0.48631132, 0.50019769, 0.4325376))
    parser.add_argument('--data_std', type=tuple, default=(0.18217433, 0.18107501, 0.19290891))
    # model
    parser.add_argument('--model', type=str, default='pre_efficientNetLib',
                        choices=['SENet', 'DenseNet', 'DLA', 'VGG', 'efficientNet', 'efficientNetLib',
                                 'pre_efficientNetLib'])
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--test_batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-1)
    # scheduler
    parser.add_argument('--milestones', type=tuple, default=[25, 50, 75, 100, 150, 200])
    parser.add_argument('--step', type=int, default=3)
    parser.add_argument('--gamma', type=float, default=0.963)
    parser.add_argument('--warmup_epoch', type=int, default=5)
    # optim
    parser.add_argument('--momentum', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    '''初始化参数结尾'''
    args = parser.parse_args()
    args.directory = 'dictionary/%s/Hi%s/' % (args.model, args.time_exp_start)
    if not os.path.exists(args.directory):
        os.makedirs(args.directory)
    log_file = os.path.join(args.directory, 'log.json')
    with open(log_file, 'w') as log:
        json.dump(vars(args), log,
                  indent=4)
    return args


class Worker:
    def __init__(self, opt):
        self.opt = opt
        torch.cuda.manual_seed(0)
        '''判定设备'''
        is_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if is_cuda else 'cpu')
        kwargs = {
            'num_workers': opt.num_workers,
            'pin_memory': True,
        } if is_cuda else {}
        '''载入数据'''
        train_dataset = datasets.ImageFolder(opt.train_dir, transform=transforms.Compose([
            # transforms.RandomRotation(10),
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.CenterCrop((256, 256)),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(opt.data_mean, opt.data_std)]))
        val_dataset = datasets.ImageFolder(opt.val_dir, transform=transforms.Compose([
            transforms.CenterCrop((256, 256)),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(opt.data_mean, opt.data_std)]))
        self.train_loader = DataLoaderX(
            dataset=train_dataset,
            batch_size=opt.batch_size, shuffle=True, **kwargs
        )
        self.val_loader = DataLoaderX(
            dataset=val_dataset,
            batch_size=opt.test_batch_size, shuffle=False, **kwargs
        )
        '''挑选神经网络、参数初始化'''
        if opt.model == 'SENet':
            Net = nn.DataParallel(SENet18(num_cls=opt.num_classes))
        if opt.model == 'VGG':
            Net = nn.DataParallel(VGG('VGG19', num_cls=opt.num_classes))
            # Net = ResNet18()
            # Net = PreActResNet18()
            # Net = GoogLeNet()
        if opt.model == 'DenseNet':
            Net = nn.DataParallel(DenseNet121(num_cls=opt.num_classes))
            # Net = ResNeXt29_2x64d()
            # Net = MobileNet()
            # Net = MobileNetV2()
            # Net = DPN92()
            # Net = ShuffleNetG2()
            # Net = ShuffleNetV2(1)
        if opt.model == 'efficientNet':
            Net = nn.DataParallel(EfficientNetB0(num_cls=opt.num_classes))
            # Net = RegNetX_200MF()
        if opt.model == 'efficientNetLib':
            Net = nn.DataParallel(EfficientNet.from_name('efficientnet-b7'))
        if opt.model == 'pre_efficientNetLib':
            Net = nn.DataParallel(EfficientNet.from_pretrained('efficientnet-b7'))
        if opt.model == 'DLA':
            Net = nn.DataParallel(DLA(num_cls=opt.num_classes))
        self.model = Net.to(self.device)
        self.optimizer = optim.SGD(params=self.model.parameters(), lr=opt.lr, momentum=opt.momentum,
                                   weight_decay=opt.weight_decay)
        self.loss_function = nn.CrossEntropyLoss()
        '''warm up setting'''
        self.per_epoch_size = len(train_dataset) // opt.batch_size
        self.warmup_step = opt.warmup_epoch * self.per_epoch_size
        self.max_iter = opt.epochs * self.per_epoch_size
        self.global_step = 0

    def train(self, epoch):
        self.model.train()
        bar = tqdm(enumerate(self.train_loader))
        for batch_idx, (data, target) in bar:
            self.global_step += 1
            self.optimizer.zero_grad()
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            loss = self.loss_function(output, target)
            loss.backward()
            lr = warmup_lr.adjust_learning_rate_cosine(
                self.optimizer, global_step=self.global_step,
                learning_rate_base=self.opt.lr,
                total_steps=self.max_iter,
                warmup_steps=self.warmup_step)
            self.optimizer.step()
            bar.set_description(
                'train epoch {} >> [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tlr: {:.6f} >> '.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                           100. * batch_idx / len(self.train_loader), loss.item(), lr
                )
            )
        bar.close()

    def val(self):
        self.model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.loss_function(output, target).item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        val_loss /= len(self.val_loader)

        print('val set >> Average loss: {:.4f}, Accuracy: {}/{} ({:.03f}%)\n'.format(
            val_loss, correct, len(self.val_loader.dataset),
            100. * correct / len(self.val_loader.dataset)))
        return 100. * correct / len(self.val_loader.dataset), val_loss


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


if __name__ == '__main__':
    # 初始化
    torch.backends.cudnn.benchmark = True
    args = get_args()
    worker = Worker(opt=args)
    # 训练与验证
    for epoch in range(1, args.epochs + 1):
        learning_rate = 0.0
        worker.scheduler.step()
        worker.train(epoch, learning_rate)
        val_acc, val_loss = worker.val()
        if epoch > args.save_station:
            save_dir = args.directory + '%s-epochs-%d-model-val-acc-%.3f-loss-%.6f.pt' \
                       % (args.model, epoch, val_acc, val_loss)
            torch.save(worker.model, save_dir)
