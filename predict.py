from models import *
from torchvision import transforms
from torch.autograd import Variable
from torch import nn
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

model_dict = 'create/DLA-epochs-306-model-val-acc-95-loss-0.pt'
pic_dir = '3.jpg'
num_cls = 3
size = (256, 256)
wondering_idx = -1  # -1 False
viewing = True
MEAN = (0.48631132, 0.50019769, 0.4325376)
STD = (0.18217433, 0.18107501, 0.19290891)

# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
classes = ('bird', 'cat', 'dog')


def if_view(image, label):
    if viewing:
        unloader = transforms.ToPILImage()
        image = image.cpu().clone()
        image = image.squeeze(0)
        image = unloader(image)
        b, g, r = image.split()
        image = Image.merge('RGB', (r, g, b))
        plt.imshow(image)
        plt.title(label)
        plt.show()


def pre_cpu(image):
    output = model(image)
    prob = F.softmax(output, dim=1)  # prob是10个分类的概率
    value, predicted = torch.max(output.data, 1)
    print(prob)
    print(predicted.item())
    print(value)
    pred_class = classes[predicted.item()]
    print(pred_class)
    if_view(image, pred_class)


def pre_gpu(image):
    output = model(image)
    prob = F.softmax(output, dim=1)
    prob = Variable(prob)
    prob = prob.cpu().numpy()
    pred = np.argmax(prob)
    print('%.06f' % prob[0][pred])
    if wondering_idx != -1:
        print('%.06f' % prob[0][wondering_idx])
    print(pred.item())
    pred_class = classes[pred]
    print(pred_class)
    if_view(image, pred_class)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = nn.DataParallel(DLA(num_cls=num_cls))
    model = model.to(device)
    model_dict = torch.load(model_dict).module.state_dict()
    model.module.load_state_dict(model_dict)
    model.eval()
    img = cv2.imread(pic_dir)
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    img = trans(img)
    img = img.to(device)
    img = img.unsqueeze(0)
    if torch.cuda.is_available():
        pre_gpu(img)
    else:
        pre_cpu(img)
