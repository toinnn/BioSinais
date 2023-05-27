import torchvision.models as models
import torch.nn as nn

vgg = models.vgg16(models.VGG16_Weights.IMAGENET1K_FEATURES)
vgg.classifier = nn.Sequential(vgg.classifier[0] ,  nn.ELU())


print(vgg.parameters)
# for i in vgg.classifier :
#     print(i)
print("Transfer Learning rodado")