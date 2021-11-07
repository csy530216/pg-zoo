from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

class Model:
    def __init__(self, device):
        self.device = device
        self.model = Net().to(device)
        self.model.load_state_dict(torch.load('mnist_cnn.pt', map_location=device))
        self.model.eval()

    def get_loss_and_grad(self, image, label, targeted=True, get_grad=True):
        image = image.clone()
        if get_grad:
            image.requires_grad_()
        input_image = (image - 0.1307) / 0.3081
        output = self.model(input_image)
        label_term = output[:, label]
        other = output + 0.0
        other[:, label] = -1e8
        other_term = torch.max(other, dim=1).values
        if targeted:
            loss = label_term - other_term
        else:
            loss = other_term - label_term
        loss = torch.squeeze(loss)
        if get_grad:
            loss.backward()
            grad = image.grad
            return loss.detach().cpu().numpy(), grad.detach()
        else:
            return loss.detach().cpu().numpy()

    def get_pred(self, image):
        input_image = (image - 0.1307) / 0.3081
        output = self.model(input_image).detach().cpu().numpy().squeeze()
        return np.argmax(output)


# Training settings
parser = argparse.ArgumentParser(description='RGF and History-PRGF attack on MNIST')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--prior', action='store_true', default=False,
                    help='use History-PRGF instead of RGF')
parser.add_argument('--lr', type=float, default=0.2)
args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device("cuda" if use_cuda else "cpu")

test_kwargs = {'batch_size': 1}
if use_cuda:
    test_kwargs.update({'num_workers': 1,
                        'pin_memory': True})

dataset = datasets.MNIST('data', download=True, train=False, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(dataset, shuffle=False, **test_kwargs)

model = Model(device)

d = 28 * 28
method = 'prgf' if args.prior else 'rgf'
lr = args.lr
epsilon = np.sqrt(d)*32/255
q = 20
sigma = 1e-4
queries_list = []

for i, (data, target) in enumerate(test_loader):
    if i >= 500:
        break
    print("Image", i)
    ite = 0
    true_label = target.numpy().squeeze()
    data = data.to(device)
    pred_label = model.get_pred(data)
    if true_label != pred_label:
        print("original prediction is wrong")
        continue
    else:
        original_image = data
        current_image = data.clone()
        loss = -10000
        v = torch.randn_like(current_image)
        queries = 0
        while loss <= 0 and ite < 20:
            angle = 0
            actual_ite = 0
            for _ in range(d // q):
                target_label = (true_label + 1) % 10
                loss, grad = model.get_loss_and_grad(current_image, target_label, targeted=True, get_grad=True)
                if loss > 0:
                    break

                us = []
                for _ in range(q):
                    us.append(torch.randn_like(current_image))
                if method == 'prgf':
                    us.append(v)

                # Gram-Schmidt. You can also call torch.linalg.qr to perform orthogonalization
                orthos = []
                for u in us:
                    for ou in orthos:
                        u = u - torch.sum(u * ou) * ou
                    u = u / torch.sqrt(torch.sum(u * u))
                    orthos.append(u)

                images = []
                for u in orthos:
                    images.append(current_image + sigma * u)
                images = torch.cat(images)
                losses = model.get_loss_and_grad(images, target_label, targeted=True, get_grad=False)

                g = 0
                for u, l in zip(orthos, losses):
                    g = g + u * (l - loss) / sigma
                for _ in range(len(orthos) + 1):
                    queries += 1
                    if queries % d == 0:
                        print(loss, norm_delta.cpu().numpy() / epsilon)
                old_image = current_image
                new_image = current_image + lr * g
                angle += torch.sum(g / torch.sqrt(torch.sum(g*g)) * grad / torch.sqrt(torch.sum(grad*grad))).item() ** 2
                actual_ite += 1
                delta = new_image - original_image
                norm_delta = torch.sqrt(torch.sum(delta * delta))
                if norm_delta >= epsilon:
                    delta = delta / norm_delta * epsilon
                    current_image = original_image + delta
                else:
                    current_image = new_image
                current_image = torch.clamp(current_image, 0, 1)
                v = current_image - old_image
            ite += 1
            if actual_ite > 0:
                print('angle =', np.sqrt(angle / actual_ite))
        print("===End of attack, queries: {}===".format(queries))
        queries_list.append(queries)

print('Median queries', np.median(queries_list))
print(method, lr)
