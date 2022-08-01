import numpy as np
import cv2
import PIL.Image
from PIL import ImageFilter 
import dnnlib
import torch
import torch.nn.functional as F
import math

def resize_image(image):
    if image.shape[2] > 256:
        return F.interpolate(image, size=(256, 256), mode='area')
    return image

def blur_score(pil_image, device, vgg16):    
    # Features for target image.
    #pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius = 1))
    w, h = pil_image.size
    #target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    #target_pil = target_pil.resize((G.img_resolution, G.img_resolution), PIL.Image.LANCZOS)
    target_uint8 = np.array(pil_image, dtype=np.uint8)
    target=torch.tensor(target_uint8.transpose([2, 0, 1]), device=device)
    target_images = target.unsqueeze(0).to(device).to(torch.float32)
    coords = []
    for x in range(16):
        for y in range(16):
            canvas = [y*256, (y+1)*256, x*256, (x+1)*256] #y_start,y_end,x_start,x_end
            coords.append(canvas)
    targets = []
    for c in coords:
        target = resize_image(target_images[0:1, 0:3, c[0]:c[1], c[2]:c[3]])
        targets.append(vgg16(target, resize_images=False, return_lpips=True))

    for i in range(100):
        new_w = math.floor((w * (i + 1)) / 100)
        new_h = math.floor((h * (i + 1)) / 100)
        blurred_img = pil_image.resize((new_w, new_h), PIL.Image.LANCZOS)
        blurred_img = blurred_img.resize((w, h), PIL.Image.LANCZOS)
        blurred_img8 = np.array(blurred_img, dtype=np.uint8)
        blurred_target=torch.tensor(blurred_img8.transpose([2, 0, 1]), device=device)
        blurred_img2 = blurred_target.unsqueeze(0).to(device).to(torch.float32)
        synth_image2 = blurred_img2.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
        synth_image2 = PIL.Image.fromarray(synth_image2, 'RGB')
        synth_image2.save(f'proj{i}.png')
        aa = np.asarray(synth_image2)
        open_cv_image = cv2.cvtColor(aa, cv2.COLOR_BGR2GRAY)
        blur_map = cv2.Laplacian(open_cv_image, cv2.CV_64F)
        dist = 0
        for (c, target) in zip(coords, targets):
            synth_image_clone = blurred_img2.clone()
            synth = resize_image(synth_image_clone[0:1, 0:3, c[0]:c[1], c[2]:c[3]])
            synth_features = vgg16(synth, resize_images=False, return_lpips=True)
            dist += (target - synth_features).square().sum()
        
        open_cv_image = cv2.cvtColor(aa, cv2.COLOR_BGR2GRAY)
        blur_map = cv2.Laplacian(open_cv_image, cv2.CV_64F)
        print((dist.cpu().numpy(), np.var(blur_map), new_w, new_h))
    
    #open_cv_image = np.array(pil_image)    
    #open_cv_image = open_cv_image[:, :, ::-1].copy() # Convert RGB to BGR 
    #if open_cv_image.ndim == 3:
    #    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    #blur_map = cv2.Laplacian(open_cv_image, cv2.CV_64F)
    #print(np.var(blur_map))
    # Convert RGB to BGR 
    #open_cv_image2 = open_cv_image2[:, :, ::-1].copy() 
    #if open_cv_image2.ndim == 3:
    #    open_cv_image2 = cv2.cvtColor(open_cv_image2, cv2.COLOR_BGR2GRAY)
    #blur_map2 = cv2.Laplacian(open_cv_image2, cv2.CV_64F)
    #print(np.var(blur_map2))
    #return np.var(blur_map)
    
    return dist
