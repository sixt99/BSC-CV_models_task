import torch

model = torch.load('../models/efficientnet_b0')
model.eval()

from PIL import Image
from torchvision.transforms import ToTensor

input_image = Image.open('../pictures/zidane.jpg')
input_image = [ToTensor()(input_image).unsqueeze(0)]
batch = torch.cat(input_image)

with torch.no_grad():
    output = torch.nn.functional.softmax(efficientnet(batch), dim=1)
    
results = utils.pick_n_best(predictions=output, n=5)

print(results)
