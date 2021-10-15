import os
from models import *
from torchvision import transforms
from torch.autograd import Variable
import cv2
import numpy as np
from tqdm import tqdm

model_dict = 'create/DLA-epochs-306-model-val-acc-95-loss-0.pt'
folder_dir = 'C:/Users/admin/Desktop/DogsVsCats_dogs-vs-cats-redux-kernels-edition/test/'
num_cls = 3
size = (256, 256)
MEAN = (0.4880121, 0.45470474, 0.41672917)
STD = (0.22934533, 0.22475937, 0.2248002)

# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
classes = ('bird', 'cat', 'dog')


def pre_cpu(image):
    output = model(image)
    prob = F.softmax(output, dim=1)  # prob是10个分类的概率
    value, predicted = torch.max(output.data, 1)
    pred_class = classes[predicted.item()]
    return pred_class, prob[0][predicted.item()]


def pre_gpu(image):
    output = model(image)
    prob = F.softmax(output, dim=1)
    prob = Variable(prob)
    prob = prob.cpu().numpy()
    pred = np.argmax(prob)
    pred_class = classes[pred]
    return pred_class, prob[0][pred]


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = nn.DataParallel(DLA(num_cls=num_cls))
    model = model.to(device)
    model_dict = torch.load(model_dict).module.state_dict()
    model.module.load_state_dict(model_dict)
    model.eval()
    folder = os.listdir(folder_dir)
    bar = tqdm(folder)
    for img_file in bar:
        img = cv2.imread(folder_dir + img_file)
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
            pred, acc = pre_gpu(img)
        else:
            pred, acc = pre_cpu(img)
        save_name = img_file + '-' + pred + '-' + '%.03f' % acc
        os.rename(os.path.join(folder_dir, img_file), os.path.join(folder_dir, save_name))
    bar.close()
