import os
import torch
import torchvision
from models.stylegan2.model import Generator
import numpy as np
from tkinter import *
from PIL import ImageTk, Image
import torch
import torchvision
import torchvision.transforms as T

models = {}
frames = {}
positive = None


def load_npz_func():
    npz_dir = npz_dir_entry.get()
    a = 0
    for file_name in os.listdir(npz_dir):
        if file_name.endswith(".npz"):
            file_path = os.path.join(npz_dir, file_name)
            if file_path not in models.keys():
                a += 1
                if a > 799:
                    break
                print(file_path)
                weights = torch.from_numpy(np.load(file_path)['w']).cuda()
                models[file_path] = torch.squeeze(weights)
                with torch.no_grad():
                    img_orig2, _ = g_ema([weights], input_is_latent=True, randomize_noise=False)
                img_orig2 = torchvision.utils.make_grid(img_orig2, normalize=True, scale_each=True, range=(-1, 1))
                img_orig2 = torch.squeeze(img_orig2)
                image2 = transform(img_orig2)
                image2 = image2.resize((256, 256), Image.ANTIALIAS)
                photo_image2 = ImageTk.PhotoImage(image2)
                frame = Frame(window, background='black', padx=5, pady=5)
                label = Label(frame, image=photo_image2, background="black", padx=1, pady=1)
                label.image = photo_image2
                label.bind('<Button-1>', lambda event, f=frame: mouse_click(f))
                label.bind('<Button-2>', lambda event, f=frame: middle_click(f))
                label.bind('<Button-3>', lambda event, f=frame: right_click(f))
                label.pack(padx=1, pady=1)
                text.configure(state="normal")
                text.window_create("insert", window=frame, padx=1, pady=1)
                text.configure(state="disabled")
                frames[frame] = file_path


def mouse_click(f):
    if f['background'] != '#000fff000':
        f['background'] = '#000fff000'
    else:
        f['background'] = 'black'
    get_positive()


def middle_click(f):
    global orig_latent
    orig_latent = models[frames[f]]
    get_positive()


def right_click(f):
    if f['background'] != 'red':
        f['background'] = 'red'
    else:
        f['background'] = 'black'
    get_positive()


def slider_move(e):
    get_positive()


def get_positive():
    latent_code = orig_latent.clone()
    positive_weights = []
    for frame, file_path in frames.items():
        if frame['background'] == '#000fff000':
            positive_weights.append(models[file_path])
    if (len(positive_weights)) == 0:
        positive_weights.append(latent_code)
    tensor_pos_weights = torch.Tensor(len(positive_weights), 18, 512).cuda()
    for i in range(len(positive_weights)):
        tensor_pos_weights[i] = positive_weights[i]
    #torch.cat(positive_weights, out=tensor_pos_weights)

    pos_std, pos_mean = torch.std_mean(tensor_pos_weights, unbiased=False, axis=0)
    pos_std *= pos_slider.get()
    pos_min = pos_mean - pos_std
    pos_max = pos_mean + pos_std

    neg_mean = pos_mean
    negative_weights = []
    for frame, file_path in frames.items():
        if frame['background'] == 'red':
            negative_weights.append(models[file_path])
    if (len(negative_weights)) > 0:
        tensor_neg_weights = torch.Tensor(len(negative_weights), 18, 512).cuda()
        for i in range(len(negative_weights)):
            tensor_neg_weights[i] = negative_weights[i]
        #torch.cat(negative_weights, out=tensor_neg_weights)
        neg_std, neg_mean = torch.std_mean(tensor_neg_weights, unbiased=False, axis=0)
        neg_std *= neg_slider.get()
        #neg_min = neg_mean - neg_std
        #neg_max = neg_mean + neg_std

        #pos_min_gt_neg_min = torch.where(pos_min > neg_min, True, False)
        #pos_max_lt_neg_max = torch.where(pos_max < neg_max, True, False)
        #cond1 = torch.logical_and(pos_min_gt_neg_min, pos_max_lt_neg_max)
        #pos_min_lt_neg_min = torch.where(pos_min < neg_min, True, False)
        #pos_max_gt_neg_max = torch.where(pos_max > neg_max, True, False)
        #latent_code_lt_pos_mean = torch.where(latent_code < pos_mean, True, False)
        #latent_code_gt_pos_mean = torch.where(latent_code >= pos_mean, True, False)
        #cond2 = torch.logical_and(pos_min_lt_neg_min, pos_max_gt_neg_max)
        #logical_or = torch.logical_or(cond1, cond2)
        #cond3 = torch.logical_and(cond2, latent_code_lt_pos_mean)
        #cond4 = torch.logical_and(cond2, latent_code_gt_pos_mean)

        #latent_code = torch.where(cond1, pos_mean, latent_code)
        #latent_code = torch.where(cond3, neg_min, latent_code)
        #latent_code = torch.where(cond4, neg_max, latent_code)
        #max_stddev = torch.maximum(pos_std, neg_std)
        
        diff = pos_mean - neg_mean
        #std_devs = diff / max_stddev
        latent_code = latent_code + (diff * std_slider.get())

        #latent_code = torch.where(logical_or, latent_code,
        #                          torch.minimum(torch.maximum(latent_code, pos_min), pos_max))
    else:
        latent_code = torch.minimum(torch.maximum(latent_code, pos_min), pos_max)
    latent_code = ((latent_code - orig_latent) * mul_slider.get()) + orig_latent

    set_image(latent_code, preview_label)
    set_image(pos_mean, preview_label2)
    set_image(neg_mean, preview_label3)


def set_image(latent_code, label):
    latent_code = torch.unsqueeze(latent_code, 0)
    with torch.no_grad():
        img_orig2, _ = g_ema([latent_code], input_is_latent=True, randomize_noise=False)
    img_orig2 = torchvision.utils.make_grid(img_orig2, normalize=True, scale_each=True, range=(-1, 1))
    img_orig2 = torch.squeeze(img_orig2)
    image2 = transform(img_orig2)
    image2 = image2.resize((256, 256), Image.ANTIALIAS)
    photo_image2 = ImageTk.PhotoImage(image2)
    label.configure(image=photo_image2)
    label.image = photo_image2


window = Tk()
toolbar = Frame(window)
text = Text(window, wrap="word", background="black", yscrollcommand=lambda *args: vsb.set(*args))
vsb = Scrollbar(window, command=text.yview)
toolbar.pack(side="top", fill="x")
vsb.pack(side="right", fill="y")
text.pack(side="left", fill="both", expand=True)

left_toolbar = Frame(toolbar)
left_toolbar.pack(side=LEFT)
npz_dir_entry = Entry(left_toolbar, width=50)
npz_dir_entry.insert(END, './outdir')
npz_dir_entry.pack(side=TOP)  # , fill=X, expand=1)

load_npz_button = Button(left_toolbar, text="Load .npz", command=load_npz_func)
load_npz_button.pack(side=TOP)

get_positive_button = Button(left_toolbar, text="Preview", command=get_positive)
get_positive_button.pack(side=TOP)

pos_slider = Scale(left_toolbar, from_=0, to=5, resolution=0.01, orient=HORIZONTAL, length=300)
pos_slider.bind('<B1-Motion>', slider_move)
pos_slider.bind('<ButtonRelease-1>', slider_move)
pos_slider.set(1)
pos_slider.pack(side=TOP)
neg_slider = Scale(left_toolbar, from_=0, to=5, resolution=0.01, orient=HORIZONTAL, length=300)
neg_slider.bind('<B1-Motion>', slider_move)
neg_slider.bind('<ButtonRelease-1>', slider_move)
neg_slider.set(1)
neg_slider.pack(side=TOP)
mul_slider = Scale(left_toolbar, from_=-2, to=2, resolution=0.01, orient=HORIZONTAL, length=300)
mul_slider.bind('<B1-Motion>', slider_move)
mul_slider.bind('<ButtonRelease-1>', slider_move)
mul_slider.set(1)
mul_slider.pack(side=TOP)
std_slider = Scale(left_toolbar, from_=-2, to=2, resolution=0.01, orient=HORIZONTAL, length=300)
std_slider.bind('<B1-Motion>', slider_move)
std_slider.bind('<ButtonRelease-1>', slider_move)
std_slider.set(1)
std_slider.pack(side=TOP)

g_ema = Generator(1024, 512, 8)
g_ema.load_state_dict(torch.load("./stylegan2-ffhq-config-f.pt")["g_ema"], strict=False)
g_ema.eval()
g_ema = g_ema.cuda()
transform = T.ToPILImage()

orig_latent = g_ema.mean_latent(4096).repeat(1, 18, 1)
with torch.no_grad():
    img_orig, _ = g_ema([orig_latent], input_is_latent=True, randomize_noise=False)
orig_latent = torch.squeeze(orig_latent)
img_orig = torchvision.utils.make_grid(img_orig, normalize=True, scale_each=True, range=(-1, 1))
img_orig = torch.squeeze(img_orig)
image = transform(img_orig)
image = image.resize((256, 256), Image.ANTIALIAS)
photo_image = ImageTk.PhotoImage(image)
preview_label = Label(toolbar, image=photo_image, width=256, height=256, background='black')
preview_label.image = photo_image
preview_label.pack(side=LEFT)
preview_label2 = Label(toolbar, image=photo_image, width=256, height=256, background='black')
preview_label2.image = photo_image
preview_label2.pack(side=LEFT)
preview_label3 = Label(toolbar, image=photo_image, width=256, height=256, background='black')
preview_label3.image = photo_image
preview_label3.pack(side=LEFT)

window.mainloop()
