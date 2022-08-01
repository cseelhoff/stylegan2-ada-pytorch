"""Project given image to the latent space of pretrained network pickle."""
import copy
import os
from time import perf_counter
import imageio
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from blur2 import blur_score
import dnnlib
import legacy

from argparse import ArgumentParser
import math
import multiprocessing as mp
import time
#from retinaface import RetinaFace
import cv2
from retinaface.pre_trained_models import get_model
import scipy
from numpy import asarray
import PIL.ImageFilter


def resize_image(image):
    if image.shape[2] > 256:
        return F.interpolate(image, size=(256, 256), mode='area')
    return image

def project2(
    G, q0, q1, q2, q3, vgg16, starting_wplus_space,
    target: torch.Tensor, # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
    *,
    num_steps                  = 1000,
    w_avg_samples              = 10000,
    initial_learning_rate      = 0.1,
    initial_noise_factor       = 0.05,
    lr_rampdown_length         = 0.25,
    lr_rampup_length           = 0.05,
    noise_ramp_length          = 0.75,
    regularize_noise_weight    = 1e5,
    verbose                    = False,
    device: torch.device,
    target_short_name: str
):
    assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)

    def logprint(*args):
        if verbose:
            print(*args)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device) # type: ignore

    # Compute w stats.
    logprint(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)       # [N, 1, C]
    w_avg = np.mean(w_samples, axis=0, keepdims=True)      # [1, 1, C]
    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

    # Setup noise inputs.
    noise_bufs = { name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name }

    # Load VGG16 feature detector.
    #url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    #with dnnlib.util.open_url(url) as f:
    #    vgg16 = torch.jit.load(f).eval().to(device)

    # Features for target image.
    target_images = target.unsqueeze(0).to(device).to(torch.float32)
    canvas = [0, 1024, 0, 1024] #y_start,y_end,x_start,x_end
    mouth = [680, 810, 512-132, 512+132]
    eyes = [410, 535, 512-242, 512+242]
    coords = [canvas, mouth, eyes]
    targets = []
    for c in coords:
        target = resize_image(target_images[0:1, 0:3, c[0]:c[1], c[2]:c[3]])
        targets.append(vgg16(target, resize_images=False, return_lpips=True))

    #starting_wplus_space = torch.load(f'../restyle/output/inference_coupled/{target_short_name}.pt')
    w_opt = starting_wplus_space.detach().clone().cuda()
    w_opt.requires_grad = True
    w_out = torch.zeros([num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)

    # Init noise.
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    for step in range(num_steps):
        # Learning rate schedule.
        t = step / num_steps
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synth images from opt_w.
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = w_opt + w_noise
        synth_images = G.synthesis(ws, noise_mode='const')

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images = (synth_images + 1) * (255/2)
        dist = 0
        for (c, target) in zip(coords, targets):
            synth_image_clone = synth_images.clone()
            synth = resize_image(synth_image_clone[0:1, 0:3, c[0]:c[1], c[2]:c[3]])
            synth_features = vgg16(synth, resize_images=False, return_lpips=True)
            dist += (target - synth_features).square().sum()

        # Noise regularization.
        reg_loss = 0.0
        for v in noise_bufs.values():
            noise = v[None,None,:,:] # must be [1,1,H,W] for F.avg_pool2d()
            while True:
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=3)).mean()**2
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=2)).mean()**2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)
        loss = dist + reg_loss * regularize_noise_weight

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        logprint(f'step {step+1:>4d}/{num_steps}: dist {dist:<4.2f} loss {float(loss):<5.2f}')

        # Save projected W for each optimization step.
        #w_out[step] = w_opt.detach()[0]
        w_out[step] = ws.detach()[0]

        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

    #return w_out.repeat([1, G.mapping.num_ws, 1])
    return w_out

#----------------------------------------------------------------------------

def run_projection3(target_pil, q0, q1, q2, q3, device, G, vgg16, starting_wplus_space, target_short_name):
    outdir = "./outdir"
    seed = 303
    np.random.seed(seed)
    torch.manual_seed(seed)
    target_uint8 = np.array(target_pil, dtype=np.uint8)
    # Optimize projection.
    start_time = perf_counter()
    projected_w_steps = project2(
        G, q0, q1, q2, q3, vgg16, starting_wplus_space,
        target=torch.tensor(target_uint8.transpose([2, 0, 1]), device=device), # pylint: disable=not-callable
        num_steps=5000,
        device=device,
        verbose=True,
        target_short_name=target_short_name
    )
    print (f'Elapsed: {(perf_counter()-start_time):.1f} s')

    # Save final projected frame and W vector.
    #target_pil.save(f'{outdir}/{target_short_name}target.png')
    projected_w = projected_w_steps[-1]
    synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode='const')
    synth_image = (synth_image + 1) * (255/2)
    synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    PIL.Image.fromarray(synth_image, 'RGB').save(f'{outdir}/{target_short_name}proj.png')
    np.savez(f'{outdir}/{target_short_name}.npz', w=projected_w.unsqueeze(0).cpu().numpy())

#----------------------------------------------------------------------------
def parse_args():
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--num_threads", type=int, default=1)
    parser.add_argument("--raw_dir", type=str, default="")
    parser.add_argument("--saved_dir", type=str, default="")
    args = parser.parse_args()
    return args

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i: i + n]

def get_landmark2(filepath, predictor):
    image = PIL.Image.open(filepath)
    data = asarray(image)
    faces = predictor.predict_jsons(data)
    #faces = RetinaFace.detect_faces(filepath, threshold=0.5, model=predictor)
    largest_face_size = 0
    largest_face = None
    for face in faces:
        print(face['score'])        
        print(face)
        face_size = (face['bbox'][2] - face['bbox'][0]) * (
                    face['bbox'][3] - face['bbox'][1])
        if face_size > largest_face_size:
            largest_face_size = face_size * face['score'] * face['score'] 
            largest_face = face
    lm = [[0, 0]] * 68
    for i in range(36, 42):
        lm[i] = largest_face['landmarks'][0]
    for i in range(42, 48):
        lm[i] = largest_face['landmarks'][1]
    lm[48] = largest_face['landmarks'][3]
    lm[54] = largest_face['landmarks'][4]

    return np.array(lm)

def quad_to_rect(Qx0, Qy0, Qx1, Qy1, Qx2, Qy2, Qx3, Qy3, x, y):
    ax = (x - Qx0) + (Qx1 - Qx0) * (y - Qy0) / (Qy0 - Qy1)
    a3x = (Qx3 - Qx0) + (Qx1 - Qx0) * (Qy3 - Qy0) / (Qy0 - Qy1)
    a2x = (Qx2 - Qx0) + (Qx1 - Qx0) * (Qy2 - Qy0) / (Qy0 - Qy1)
    ay = (y - Qy0) + (Qy3 - Qy0) * (x - Qx0) / (Qx0 - Qx3)
    a1y = (Qy1 - Qy0) + (Qy3 - Qy0) * (Qx1 - Qx0) / (Qx0 - Qx3)
    a2y = (Qy2 - Qy0) + (Qy3 - Qy0) * (Qx2 - Qx0) / (Qx0 - Qx3)
    bx = x * y - Qx0 * Qy0 + (Qx1 * Qy1 - Qx0 * Qy0) * (y - Qy0) / (Qy0 - Qy1)
    b3x = Qx3 * Qy3 - Qx0 * Qy0 + (Qx1 * Qy1 - Qx0 * Qy0) * (Qy3 - Qy0) / (Qy0 - Qy1)
    b2x = Qx2 * Qy2 - Qx0 * Qy0 + (Qx1 * Qy1 - Qx0 * Qy0) * (Qy2 - Qy0) / (Qy0 - Qy1)
    by = x * y - Qx0 * Qy0 + (Qx3 * Qy3 - Qx0 * Qy0) * (x - Qx0) / (Qx0 - Qx3)
    b1y = Qx1 * Qy1 - Qx0 * Qy0 + (Qx3 * Qy3 - Qx0 * Qy0) * (Qx1 - Qx0) / (Qx0 - Qx3)
    b2y = Qx2 * Qy2 - Qx0 * Qy0 + (Qx3 * Qy3 - Qx0 * Qy0) * (Qx2 - Qx0) / (Qx0 - Qx3)

    l = (ax / a3x) + (1 - a2x / a3x) * (bx - b3x * ax / a3x) / (b2x - b3x * a2x / a3x)
    m = (ay / a1y) + (1 - a2y / a1y) * (by - b1y * ay / a1y) / (b2y - b1y * a2y / a1y)
    return l, m

def align_face(filepath, predictor, device, vgg16):
    """
    :param filepath: str
    :return: PIL Image
    """

    # lm = get_landmark(filepath, predictor)
    # print(lm)
    lm = get_landmark2(filepath, predictor)
    # print(lm)

    lm_eye_left = lm[36:42]  # left-clockwise
    lm_eye_right = lm[42:48]  # left-clockwise
    lm_mouth_outer = lm[48:60]  # left-clockwise
    # Calculate auxiliary vectors.
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg
    
    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    # read image
    img = PIL.Image.open(filepath)

    output_size = 1024
    transform_size = 4096
    enable_padding = True

    # Shrink.
    img_before_resize = img
    quad_before_resize = quad
    qsize_before_resize = qsize
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (
        int(np.floor(min(quad[:, 0]))),
        int(np.floor(min(quad[:, 1]))),
        int(np.ceil(max(quad[:, 0]))),
        int(np.ceil(max(quad[:, 1]))),
    )
    crop = (
        max(crop[0] - border, 0),
        max(crop[1] - border, 0),
        min(crop[2] + border, img.size[0]),
        min(crop[3] + border, img.size[1]),
    )
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    backup_quad = (
        max(quad[0][0], 0),
        max(quad[0][1], 0),
        max(quad[1][0], 0),
        min(quad[1][1], img.size[1]),
        min(quad[2][0], img.size[0]),
        min(quad[2][1], img.size[1]),
        min(quad[3][0], img.size[0]),
        max(quad[3][1], 0))
    if enable_padding:
        # Pad.
        pad = (
            int(np.floor(min(quad[:, 0]))),
            int(np.floor(min(quad[:, 1]))),
            int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))),
        )
        pad = (
            max(-pad[0] + border, 0),
            max(-pad[1] + border, 0),
            max(pad[2] - img.size[0] + border, 0),
            max(pad[3] - img.size[1] + border, 0),
        )
        if max(pad) > border - 4:
            pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
            img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), "reflect")
            h, w, _ = img.shape
            y, x, _ = np.ogrid[:h, :w, :1]
            mask = np.maximum(
                1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]),
            )
            blur = qsize * 0.02
            img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
            img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
            img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), "RGB")
            quad += pad[:2]

    # Transform.
    flat_quad = (quad + 0.5).flatten()
    img = img.transform((transform_size, transform_size), PIL.Image.Transform.QUAD, flat_quad, PIL.Image.Resampling.BILINEAR)
    l, m = quad_to_rect(flat_quad[0], flat_quad[1], flat_quad[2], flat_quad[3], flat_quad[4], flat_quad[5], flat_quad[6], flat_quad[7], backup_quad[0], backup_quad[1])
    l = max(0, math.floor(l * output_size))
    m = max(0, math.floor(m * output_size))
    q0 = (l, m)
    l, m = quad_to_rect(flat_quad[0], flat_quad[1], flat_quad[2], flat_quad[3], flat_quad[4], flat_quad[5], flat_quad[6], flat_quad[7], backup_quad[2], backup_quad[3])
    l = max(0, math.floor(l * output_size))
    m = min(output_size - 1, math.floor(m * output_size))
    q1 = (l, m)
    l, m = quad_to_rect(flat_quad[0], flat_quad[1], flat_quad[2], flat_quad[3], flat_quad[4], flat_quad[5], flat_quad[6], flat_quad[7], backup_quad[4], backup_quad[5])
    l = min(output_size - 1, math.floor(l * output_size))
    m = min(output_size - 1, math.floor(m * output_size))
    q2 = (l, m)
    l, m = quad_to_rect(flat_quad[0], flat_quad[1], flat_quad[2], flat_quad[3], flat_quad[4], flat_quad[5], flat_quad[6], flat_quad[7], backup_quad[6], backup_quad[7])
    l = min(output_size - 1, math.floor(l * output_size))
    m = max(0, math.floor(m * output_size))
    q3 = (l, m)
    print(q0, q1, q2, q3)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.Resampling.LANCZOS)

    # Save aligned image.
    return img, q0, q1, q2, q3

def extract_on_paths(file_paths, device, vgg16, G):
    # predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    #predictor = RetinaFace.build_model()
    predictor = get_model("resnet50_2020-07-20", max_size=2048)
    predictor.eval()
    pid = mp.current_process().name
    print("\t{} is starting to extract on #{} images".format(pid, len(file_paths)))
    tot_count = len(file_paths)
    count = 0
    for file_path, res_path in file_paths:
        count += 1
        if count % 100 == 0:
            print("{} done with {}/{}".format(pid, count, tot_count))
        #try:
        res, q0, q1, q2, q3 = align_face(file_path, predictor, device, vgg16)
        res = res.convert("RGB")
        os.makedirs(os.path.dirname(res_path), exist_ok=True)
        res.save(res_path)
        #use restyle to get starting wplus_space
        starting_wplus_space = 0
        #find w+ space, by comparing q0-q3
        run_projection3(res, q0, q1, q2, q3, device, G, vgg16, starting_wplus_space, target_short_name='target')
        #except Exception:
        #    continue
    print("\tDone!")

def getVgg16Device():
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    device = torch.device('cuda')
    network_pkl = "./ffhq.pkl"
    with dnnlib.util.open_url(network_pkl) as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device) # type: ignore
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)
    return device, vgg16, G


def run(args):
    root_path = args.raw_dir
    root_path = './raw'
    out_crops_path = args.saved_dir
    out_crops_path = './saved'
    if not os.path.exists(out_crops_path):
        os.makedirs(out_crops_path, exist_ok=True)

    file_paths = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            file_path = os.path.join(root, file)
            fname = os.path.join(out_crops_path, os.path.relpath(file_path, root_path))
            res_path = "{}.jpg".format(os.path.splitext(fname)[0])
            if os.path.splitext(file_path)[1] == ".txt" or os.path.exists(res_path):
                continue
            file_paths.append((file_path, res_path))

    file_chunks = list(chunks(file_paths, int(math.ceil(len(file_paths) / args.num_threads))))
    pool = mp.Pool(args.num_threads)
    print("Running on {} paths\nHere we goooo".format(len(file_paths)))
    tic = time.time()

    print('Loading networks...')
    device, vgg16, G = getVgg16Device()

    #pool.map(extract_on_paths, file_chunks)
    extract_on_paths(file_paths, device, vgg16, G)
    toc = time.time()
    print("Mischief managed in {}s".format(toc - tic))

if __name__ == "__main__":
    args = parse_args()
    run(args)
    #ldir = '../restyle/data'
    #target_short_names = os.listdir(ldir)
    #run_projection2(target_short_names) # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
