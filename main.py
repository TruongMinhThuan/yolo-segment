from ultralytics import YOLO
import torch
import cv2

model = YOLO('yolov8n-seg.pt')  # load a pretrained YOLOv8n segmentation model

predicted = model.predict("https://nft-snap-dev.s3.ap-northeast-1.amazonaws.com/text-to-img/f3e1d25a-31f1-490d-82d5-7fb2d11c5a79-2024-03-19_01-55-17_image.jpg",
                          classes=[2]
                        )

print("Predicted: ",predicted[0].masks)
if predicted[0].masks is not None:
  
  mask = predicted[0].masks[0].data
  boxes = predicted[0].boxes[0].data

  # clss = boxes[:, 5]
  # people_indices = torch.where(clss == 0)
  # people_masks = 0

  people_mask = torch.any(mask, dim=0).int() * 255
  # save to file
  cv2.imwrite(str('person3_segs.jpg'), people_mask.cpu().numpy())


# for result in predicted:
#     # get array results
#     masks = result.masks.data
#     boxes = result.boxes.data
#     # extract classes
#     clss = boxes[:, 5]
#     # get indices of results where class is 0 (people in COCO)
#     people_indices = torch.where(clss == 0)
#     # use these indices to extract the relevant masks
#     people_masks = masks[people_indices]
#     # scale for visualizing results
#     people_mask = torch.any(people_masks, dim=0).int() * 255
#     # save to file
#     cv2.imwrite(str(model.predictor.save_dir / 'merged_segs.jpg'), people_mask.cpu().numpy())
