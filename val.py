from models import *
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from tqdm import tqdm
from matplotlib import pyplot as plt

model_dict = 'create/dla-epochs-200-model-val-acc-91.pt'
batch_size = 1
num_workers = 4
data_mean = (.1307,)
data_std = (.3081,)
viewing = True  # batch_size 1
auto_viewing = False

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(data_mean, data_std)
])


def snap_forward(model, dataloader, device):
    model.eval()
    correct = 0
    bar = tqdm(enumerate(dataloader))
    for index, (data, target) in bar:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        '''预测展示'''
        if viewing:
            unloader = transforms.ToPILImage()
            image = data.cpu().clone()
            image = image.squeeze(0)
            image = unloader(image)
            label = classes[int(target.view_as(pred))]
            pred_class = classes[pred]
            plt.imshow(image)
            plt.title('pred >> ' + pred_class + ' || ' + 'label >> ' + label)
            if auto_viewing:
                plt.pause(0.001)
            else:
                plt.show()
    bar.close()
    return '[%d/%d] correct rate >> %.06f' % (correct, len(dataloader.dataset), correct / len(dataloader.dataset))


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DLA(num_cls=10)
    model.load_state_dict(torch.load(model_dict))
    model = model.to(device)
    model.eval()  # 把模型转为test模式
    kwargs = {
        'num_workers': num_workers,
        'pin_memory': True
    } if torch.cuda.is_available() else {}
    dataloader = DataLoader(
        dataset=CIFAR10('data', train=False,
                        transform=trans
                        ),
        batch_size=batch_size, shuffle=False, **kwargs
    )
    report = snap_forward(model, dataloader, device)
    print(report)
