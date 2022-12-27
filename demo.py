import cv2
import numpy as np
import torch
import flow_vis

from deeplsd.models.deeplsd import DeepLSD

def get_flow_vis(df, ang, line_neighborhood=5):
    norm = line_neighborhood + 1 - np.clip(df, 0, line_neighborhood)
    flow_uv = np.stack([norm * np.cos(ang), norm * np.sin(ang)], axis=-1)
    flow_img = flow_vis.flow_to_color(flow_uv, convert_to_bgr=False)
    return flow_img

# Load an image
img = cv2.imread('assets/images/example.jpg')[:, :, ::-1]
gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# Model config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
conf = {
    'sharpen': True,  # Use the DF normalization (should be True)
    'detect_lines': False,  # Whether to detect lines or only DF/AF
    'multiscale': False,
    'line_detection_params': {
        'merge': False,  # Whether to merge close-by lines
        'optimize': False,  # Whether to refine the lines after detecting them
        'use_vps': True,  # Whether to use vanishing points (VPs) in the refinement
        'optimize_vps': True,  # Whether to also refine the VPs in the refinement
        'filtering': True,  # Whether to filter out lines based on the DF/AF. Use 'strict' to get an even stricter filtering
        'grad_thresh': 3,
        'grad_nfa': True,  # If True, use the image gradient and the NFA score of LSD to further threshold lines. We recommand using it for easy images, but to turn it off for challenging images (e.g. night, foggy, blurry images)
    }
}

# Load the model
ckpt = 'deeplsd_md.tar'
ckpt = torch.load(str(ckpt), map_location='cpu')
net = DeepLSD(conf)
net.load_state_dict(ckpt['model'])
net = net.to(device).eval()

# Detect (and optionally refine) the lines
inputs = torch.tensor(gray_img, dtype=torch.float, device=device)[None, None] / 255.
with torch.no_grad():
    df, line_level = net(inputs)
    
    df = np.squeeze(df.cpu().numpy())
    line_level = np.squeeze(line_level.cpu().numpy())

    flow_img = get_flow_vis(df, line_level, line_neighborhood=5)
    cv2.imshow('flow_img', flow_img)
    cv2.waitKey(0)
    cv2.imwrite('flow_img.png', flow_img)

    # import matplotlib.pyplot as plt
    # plt.imshow(np.squeeze(df.numpy()))
    # plt.show()