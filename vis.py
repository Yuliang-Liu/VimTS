import os, sys
import torch
import numpy as np

from models.ests import build_ests
from util.slconfig import SLConfig
from util.visualizer import COCOVisualizer
from util import box_ops
from PIL import Image
import datasets.transforms as T

CTLABELS = [' ','!','"','#','$','%','&','\'','(',')','*','+',',','-','.','/','0','1','2','3','4','5','6','7','8','9',':',';','<','=','>','?','@','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','[','\\',']','^','_','`','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','{','|','}','~']
def _decode_recognition(rec):
    s = ''
    rec = rec.tolist()
    for c in rec:
        if c>94:
            continue
        s += CTLABELS[c]
    return s

def build_model_main(args):
    # we use register to maintain models from catdet6 on.
    from models.registry import MODULE_BUILD_FUNCS
    assert args.modelname in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    args.device = 'cuda'
    model, criterion, postprocessors = build_func(args)
    return model, criterion, postprocessors

model_config_path = "config/ESTS/ESTS_5scale_tt_finetune.py" # change the path of the model config file
model_checkpoint_path = "logs_cross/croos_domain_prompt_attn_using_ctw_model_TT_CTW/checkpoint0079.pth" # change the path of the model checkpoint
# See our Model Zoo section in README.md for more details about our pretrained models.

args = SLConfig.fromfile(model_config_path) 
model, criterion, postprocessors = build_model_main(args)
checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['model'])
model.eval()
model.cuda()
transform = T.Compose([
    T.RandomResize([800],max_size=1333),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]
)
image_dir = '/data/hmx/video_data/ICDAR2015/images/test/Video_15_4_1/'
image_dir1 = os.listdir(image_dir)
for idx, i in enumerate(image_dir1):
    image = Image.open(image_dir + i).convert('RGB')
    image, _ = transform(image,None)
    output = model(image[None].cuda())
    output = postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]))[0]
    rec = [_decode_recognition(k) for k in output['rec']]
    thershold = 0.4 # set a thershold
    scores = output['scores']
    labels = output['labels']
    boxes = box_ops.box_xyxy_to_cxcywh(output['boxes'])
    select_mask = scores > thershold
    recs = []
    for j,r in zip(select_mask,rec):
        if j:
            recs.append(r)
    vslzr = COCOVisualizer()
    # box_label = ['text' for item in rec[select_mask]]
    pred_dict = {
        'boxes': boxes[select_mask],
        'size': torch.tensor([image.shape[1],image.shape[2]]),
        'box_label': recs,
        'image_id': 1,
        'beziers': output['beziers'][select_mask],
        'image_name': i
    }
    vslzr.visualize(image, pred_dict, savedir='vis_video')