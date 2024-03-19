from ultralytics import YOLO
import torch
import cv2

init_model = "yolov8n-seg.pt"

model = YOLO(init_model)  # load a pretrained YOLOv8n segmentation model


image_path = "https://nft-snap-dev.s3.ap-northeast-1.amazonaws.com/original_ai_image/zQ05enzaZQntf2BcpOIq1SHv4ANZ59aUQ4TrWR0J.jpg"

predicted = model.predict(
    source=image_path,
    classes=[0],
    conf=0.7,
    retina_masks=True,
    device='cpu',
    iou=0.9
)

print("Predicted: ", predicted[0].masks)
if predicted[0].masks is not None:

    mask = predicted[0].masks[0].data
    boxes = predicted[0].boxes[0].data

    # clss = boxes[:, 5]
    # people_indices = torch.where(clss == 0)
    # people_masks = 0

    people_mask = torch.any(mask, dim=0).int() * 255
    # save to file
    cv2.imwrite(str('person5_segs.jpg'), people_mask.cpu().numpy())
else:
    print("No mask found")

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
