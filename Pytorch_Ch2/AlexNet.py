from torchvision import models

# Untrained AlexNet
alexnet = models.AlexNet()

# Trained resnet
resnet = models.resnet101(pretrained=True)

# Take a peak
resnet

# Preprocessing functions from torchvision
from torchvision import transforms
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
    # 1) scales image to 256 x 256
    # 2) Crops image to 224 x 224 around center
    # 3) Transform to a 3D-tensor
    # 4) Normalize its TGB components 

# Load image of Kimmie and I
from PIL import Image
img = Image.open("C:/Users/isaac/Documents/ML_pytorch/Data/Pytorch_Ch2/train_KandI.jpg")

# Look at cute pic 
img.show()

# Preprocess image
img_t = preprocess(img)

# A little bit more reshaping
import torch
batch_t = torch.unsqueeze(img_t, 0)

### RUN!

# Put network in eval mode
resnet.eval()

# Inference
out = resnet(batch_t)
out

# Load file with 1,000 labels for ImageNet dataset classes
with open("C:/Users/isaac/Documents/ML_pytorch/Repos/dlwpt-code/data/p1ch2/imagenet_classes.txt") as f:
    labels = [line.strip() for line in f.readlines()]

# Which index got the max score
_, index = torch.max(out, 1)
    # 1 element, 1-D tensor

# Normalize output score to between 0 and 1
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
labels[index[0]], percentage[index[0]].item()
    # Thinks it is an oxygen mask, though not very confident
    # Would love to eventually set out testing score calibration
    # Like if it says its 44% confidence, does it actually get it right 44% of the time?
    # And is that even how I should be interpreting these normalized scores?

# Can find 2nd, 3rd, 4th best etc.
_, indices = torch.sort(out, descending=True)
[(labels[idx], percentage[idx].item()) for idx in indices[0][:5]]


