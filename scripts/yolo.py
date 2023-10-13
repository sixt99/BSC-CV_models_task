import torch

model = torch.load('../models/yolov5x6')
imgs = ['../pictures/zidane.jpg']  # batch of images
results = model(imgs)

# Results
# results.print()
results.save()  # or .show()

results.xyxy[0]  # img1 predictions (tensor)
results.pandas().xyxy[0]

