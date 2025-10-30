import torch
import numpy as np
from groundingdino.util.inference import load_image, predict
from torchvision.ops import box_convert
import matplotlib.pyplot as plt
import matplotlib.patches as patches



def GroundingSAM(g_dino, sam, image_path, text_prompt, box_threshold = 0.3, text_threshold = 0.2, show_boxes = False, device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")):
    ''' Function concatenating GroundingDINO and SAM models to create binary masks of text prompted obgects
    
    Arguments:
    ------------
    g_dino : groundingdino.models.GroundingDINO, GroundingDINO model
    sam :  segment_anything.predictor.SamPredictor, Segment anything model
    image_path : str
    text_prompt : str
    box_threshold : float, threshold value for box detection for GroundingDINO, default is 0.3
    text_threshold : float, threshold value for text correspondence for GroundingDINO default is 0.2
    show_boxes : bool, if True shows GroundingDINO boxes, default is False
    '''
    
    image_source, image_for_dino = load_image(image_path)

    # GroundingDINO predictions
    boxes, scores, labels = predict(
        model = g_dino,
        image = image_for_dino,
        caption = text_prompt,
        box_threshold = box_threshold,
        text_threshold = text_threshold,
        device = device )
    
    # Converting boxes output values from [0,1] to [H,W]
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    boxes = box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")

    image_np = np.array(image_source)

    fig, ax = plt.subplots(1, figsize = (15, 15))
    ax.imshow(image_np)
    
    
    sam.set_image(image_source)

    sam_masks = np.zeros((1, h, w), dtype = bool)

    # Get segmenting masks for the boxes
    for box in boxes:
        mask, _, _ = sam.predict(point_coords = None, 
                                        point_labels = None,
                                        box = box.numpy(),
                                        #mask_input=input_labels,
                                        multimask_output = False)
        
        sam_masks = np.logical_or(sam_masks, mask)

    
    if show_boxes:
        for box, score, label in zip(boxes, scores, labels):
           
            x_min, y_min, x_max, y_max = box

            rect = patches.Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                linewidth=1,
                edgecolor="red",
                facecolor="none", )

            # Add the rectangle to the plot
            ax.add_patch(rect)

            # Label with score
            label_text = f"{label}: {score:.2f}"
            ax.text(
                x_min,
                y_min - 5,  # Slightly above the box
                label_text,
                color = "white",
                fontsize = 7,
                bbox=dict(facecolor="red", alpha=0.5) )
        
    #plt.imshow(sam_masks[0], alpha=0.5)  # Overlay the mask on the image

    # Overlay the sam_masks to the original image
    color = np.array([255, 0, 0], dtype=np.uint8) 

    overlay = np.zeros((sam_masks.shape[1], sam_masks.shape[2], 4), dtype=np.uint8)
    overlay[..., :3] = color
    overlay[..., 3] = sam_masks.astype(np.uint8) * 200

    plt.imshow(overlay)
    plt.axis('off')
    plt.show()

    return sam_masks, boxes