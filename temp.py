import torchvision.models as models

resnet50 = models.resnet50()
print(*list(resnet50.children())[:4])