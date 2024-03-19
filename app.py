import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Images
imgs = ['https://nft-snap-dev.s3.ap-northeast-1.amazonaws.com/text-to-img/f3e1d25a-31f1-490d-82d5-7fb2d11c5a79-2024-03-19_01-55-17_image.jpg']  # batch of images

# Inference
results = model(imgs)

#  show results
results.show()