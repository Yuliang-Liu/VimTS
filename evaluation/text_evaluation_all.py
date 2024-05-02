import contextlib
import copy
import io
import itertools
import json
import logging
import numpy as np
import os
import re
import torch
from collections import OrderedDict
from fvcore.common.file_io import PathManager
from pycocotools.coco import COCO

from detectron2.utils import comm
from detectron2.data import MetadataCatalog
from detectron2.evaluation.evaluator import DatasetEvaluator

import glob
import shutil
from shapely.geometry import Polygon, LinearRing
from . import text_eval_script
from . import text_eval_script_ic15
import zipfile
import pickle
import editdistance
import cv2
from .lexicon_procesor import LexiconMatcher

NULL_CHAR = u'口'


class TextEvaluator():
    """
    Evaluate text proposals and recognition.
    """

    def __init__(self, task, lexicon_type, use_lexicon, use_customer_dictionary, dataset_name, distributed, output_dir='eavaluate_result'):
        self._tasks = ("polygon", "recognition")
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        self.task = self._tasks[task]
        self.use_customer_dictionary = use_customer_dictionary
        self.use_polygon = True
        if not self.use_customer_dictionary:
            self.CTLABELS = [' ','!','"','#','$','%','&','\'','(',')','*','+',',','-','.','/','0','1','2','3','4','5','6','7','8','9',':',';','<','=','>','?','@','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','[','\\',']','^','_','`','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','{','|','}','~']
            self.voc_size = 96
        else:
            with open(self.use_customer_dictionary, 'rb') as fp:
                self.CTLABELS = pickle.load(fp)
            self.voc_size = len(self.CTLABELS) + 1
        assert(int(self.voc_size - 1) == len(self.CTLABELS)), "voc_size is not matched dictionary size, got {} and {}.".format(int(self.voc_size - 1), len(self.CTLABELS))

        self._predictions = []
        # use dataset_name to decide eval_gt_path
        self.dataset_name = dataset_name
        self.submit = False
        if "totaltext" in dataset_name:
            self._text_eval_gt_path = "evaluation/gt_totaltext_line_level.zip"
            # self._text_eval_gt_path = "evaluation/gt_totaltext.zip"
            self._word_spotting = False
            self.rec_confidence = 0.0 #0.977
            self._text_eval_confidence = 0.33 #0.38
            weighted_ed = False
        elif "ctw1500" in dataset_name:
            self._text_eval_gt_path = "evaluation/gt_ctw1500_word.zip"
            # self._text_eval_gt_path = "evaluation/gt_ctw1500.zip"
            self._word_spotting = False
            self.rec_confidence = 0.0
            self._text_eval_confidence = 0.418
            weighted_ed = False
        elif "icdar2015" in dataset_name:
            self._text_eval_gt_path = "evaluation/gt_icdar2015.zip"
            self._word_spotting = False
            self.rec_confidence = 0.0
            self._text_eval_confidence = 0.355
            weighted_ed = True
        elif "mlt2019_test" in dataset_name:
            self._text_eval_gt_path = ""
            self._word_spotting = False
            self.rec_confidence = 0
            self._text_eval_confidence = 0.415
            weighted_ed = False
            self.submit = True
        elif "vintext" in dataset_name:
            self.CTLABELS = [' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~', 'ˋ', 'ˊ', '﹒', 'ˀ', '˜', 'ˇ', 'ˆ', '˒', '‑']
            self.voc_size = 107
            self._text_eval_gt_path = "evaluation/gt_vintext.zip"
            self._word_spotting = True
            self.rec_confidence = 0.975
            self._text_eval_confidence = 0.415
            weighted_ed = False
            use_lexicon = False
        elif "ic13_video" in dataset_name:
            self._text_eval_gt_path = "/data/gt_ic13_video.zip"
            self._word_spotting = False
            self.rec_confidence = 0.97
            self._text_eval_confidence = 0.415
            self.dataset_name = "ic13_video"
            self.lexicon_type = None
            weighted_ed = False
            use_lexicon = False
        else:
            self._text_eval_gt_path = ""
            self.rec_confidence = 0
            self._text_eval_confidence = 0.415
        self._lexicon_matcher = LexiconMatcher(dataset_name, lexicon_type, use_lexicon, 
                                               self.CTLABELS + [NULL_CHAR],
                                               weighted_ed=weighted_ed)

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}
            instances = output
            prediction["instances"] = self.instances_to_coco_json(instances, input)
            self._predictions.append(prediction)

    def to_eval_format(self, file_path, temp_dir="temp_det_results", cf_th=0.5):
        def fis_ascii(s):
            a = (ord(c) < 128 for c in s)
            return all(a)

        def de_ascii(s):
            if "vintext" in self.dataset_name:
                a = [c for c in s]
            else:
                a = [c for c in s if ord(c) < 128]
            outa = ''
            for i in a:
                outa +=i
            return outa

        with open(file_path, 'r') as f:
            data = json.load(f)
            with open('temp_all_det_cors.txt', 'w') as f2:
                for ix in range(len(data)):
                    if data[ix]['score'] > 0.1:
                        outstr = '{}: '.format(data[ix]['image_id'])
                        xmin = 1000000
                        ymin = 1000000
                        xmax = 0 
                        ymax = 0
                        for i in range(len(data[ix]['polys'])):
                            outstr = outstr + str(int(data[ix]['polys'][i][0])) +','+str(int(data[ix]['polys'][i][1])) +','
                        ass = de_ascii(data[ix]['rec'])
                        if len(ass)>=0: # 
                            outstr = outstr + str(round(data[ix]['score'], 3)) +',####'+ass+'\n'	
                            f2.writelines(outstr)
                f2.close()
        dirn = temp_dir
        lsc = [cf_th] 
        fres = open('temp_all_det_cors.txt', 'r').readlines()
        for isc in lsc:	
            if not os.path.isdir(dirn):
                os.mkdir(dirn)

            for line in fres:
                line = line.strip()
                s = line.split(': ')
                filename = '{:07d}.txt'.format(int(s[0]))
                outName = os.path.join(dirn, filename)
                with open(outName, 'a') as fout:
                    ptr = s[1].strip().split(',####')
                    score = ptr[0].split(',')[-1]
                    if float(score) < isc:
                        continue
                    cors = ','.join(e for e in ptr[0].split(',')[:-1])
                    fout.writelines(cors+',####'+ptr[1]+'\n')
        os.remove("temp_all_det_cors.txt")

    def sort_detection(self, temp_dir):
        origin_file = temp_dir
        output_file = "final_"+temp_dir

        if not os.path.isdir(output_file):
            os.mkdir(output_file)

        files = glob.glob(origin_file+'*.txt')
        files.sort()

        for i in files:
            out = i.replace(origin_file, output_file)
            fin = open(i, 'r').readlines()
            fout = open(out, 'w')
            for iline, line in enumerate(fin):
                ptr = line.strip().split(',####')
                rec  = ptr[1]
                cors = ptr[0].split(',')
                assert(len(cors) %2 == 0), 'cors invalid.'
                pts = [(int(cors[j]), int(cors[j+1])) for j in range(0,len(cors),2)]
                try:
                    pgt = Polygon(pts)
                except Exception as e:
                    print(e)
                    print('An invalid detection in {} line {} is removed ... '.format(i, iline))
                    continue
                
                if not pgt.is_valid:
                    print('An invalid detection in {} line {} is removed ... '.format(i, iline))
                    continue
                    
                pRing = LinearRing(pts)
                if pRing.is_ccw:
                    pts.reverse()
                outstr = ''
                for ipt in pts[:-1]:
                    outstr += (str(int(ipt[0]))+','+ str(int(ipt[1]))+',')
                outstr += (str(int(pts[-1][0]))+','+ str(int(pts[-1][1])))
                outstr = outstr+',####' + rec
                fout.writelines(outstr+'\n')
            fout.close()
        os.chdir(output_file)

        def zipdir(path, ziph):
            # ziph is zipfile handle
            for root, dirs, files in os.walk(path):
                for file in files:
                    ziph.write(os.path.join(root, file))

        zipf = zipfile.ZipFile('../det.zip', 'w', zipfile.ZIP_DEFLATED)
        zipdir('./', zipf)
        zipf.close()
        os.chdir("../")
        # clean temp files
        shutil.rmtree(origin_file)
        shutil.rmtree(output_file)
        return "det.zip"
    
    def evaluate_with_official_code(self, result_path, gt_path):
        return text_eval_script.text_eval_main(det_file=result_path, gt_file=gt_path, is_word_spotting=self._word_spotting)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
            return {}

        PathManager.mkdirs(self._output_dir)
        if self.submit:
            if 'rects'in self.dataset_name:
                file_path = os.path.join(self._output_dir, self.dataset_name+"_submit.txt")
                self._logger.info("Saving results to {}".format(file_path))
                with PathManager.open(file_path, "w") as f:
                    for prediction in predictions:
                        write_id = "{:06d}".format(prediction["image_id"]+1)
                        write_img_name = "test_"+write_id+'.jpg\n'
                        f.write(write_img_name)
                        if len(prediction["instances"]) > 0:
                            for inst in prediction["instances"]:
                                write_poly, write_text, write_score = inst["polys"], inst["rec"], inst["score"]
                                if write_score < 0.35:
                                    continue
                                if write_text == '':
                                    continue
                                if not LinearRing(write_poly).is_ccw:
                                    write_poly.reverse()
                                write_poly = np.array(write_poly).reshape(-1).tolist()
                                write_poly = ','.join(list(map(str,write_poly)))
                                f.write(write_poly+','+write_text+'\n')
                    f.flush()
                self._logger.info("Ready to submit results from {}".format(file_path))
            elif 'mlt'in self.dataset_name:
                file_path = os.path.join(self._output_dir, self.dataset_name+"_test")
                self._logger.info("Saving results to {}".format(file_path))
                PathManager.mkdirs(file_path)
                for prediction in predictions:
                    write_id = "{:05d}".format(prediction["image_id"].item())
                    write_img_name = os.path.join(file_path, "res_img_"+write_id+'.txt')
                    f = open(write_img_name, 'w')
                    if len(prediction["instances"]) > 0:
                        for inst in prediction["instances"]:
                            write_poly, write_text, write_score = inst["polys"], inst["rec"], inst["score"]
                            if write_score < 0.305:
                                continue
                            if not LinearRing(write_poly).is_ccw:
                                write_poly.reverse()
                            write_poly = np.array(write_poly).reshape(-1).tolist()
                            write_poly = ','.join(list(map(str,write_poly)))
                            f.write(write_poly + ',' + str(write_score) +'\n')
                    f.flush()
        else:
            coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))
            file_path = os.path.join(self._output_dir, "text_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(coco_results))
                f.flush()

        self._results = OrderedDict()
        
        if not self._text_eval_gt_path:
            return copy.deepcopy(self._results)
        # eval text
        temp_dir = "temp_det_results/"
        self.to_eval_format(file_path, temp_dir, self._text_eval_confidence)
        result_path = self.sort_detection(temp_dir)
        text_result = self.evaluate_with_official_code(result_path, self._text_eval_gt_path)
        os.remove(result_path)

        # parse
        template = "(\S+): (\S+): (\S+), (\S+): (\S+), (\S+): (\S+)"
        if self.task == "recognition":
            task = "e2e_method"
        else:
            task = "det_only_method"
        result = text_result[task]
        groups = re.match(template, result).groups()
        self._results[groups[0]] = {groups[i*2+1]: float(groups[(i+1)*2]) for i in range(3)}
        return copy.deepcopy(self._results)


    def instances_to_coco_json(self, instances, inputs):
        img_id = inputs["image_id"]
        width = inputs['orig_size'][0].item()
        height = inputs['orig_size'][1].item()
        num_instances = len(instances)
        if num_instances == 0:
            return []
        scores = instances['scores']
        beziers = instances['beziers']
        recs = instances['rec']
        rec_scores = instances['rec_score']
        rec_probs = instances['rec_prob']
        results = []
        for pnt, rec, score, rec_score, rec_prob in zip(beziers, recs, scores, rec_scores, rec_probs):
            if self.task == "recognition":
                if rec_score.mean() < self.rec_confidence:
                    continue
            # convert beziers to polygons
            poly = self.pnt_to_polygon(pnt)
            if 'rects' in self.dataset_name or 'mlt' in self.dataset_name:
                poly = polygon2rbox(poly, height, width)
            s = self.decode(rec)

            s = self._lexicon_matcher.find_match_word(s, img_id=str(img_id), scores=rec_prob.cpu().numpy())
            if s is None:
                continue
            result = {
                "image_id": img_id,
                "category_id": 1,
                "polys": poly,
                "rec": s,
                "score": score.item(),
            }
            results.append(result)
        return results

    def pnt_to_polygon(self, ctrl_pnt):
        if self.use_polygon:
            return ctrl_pnt.reshape(-1, 2).tolist()
        else:
            u = np.linspace(0, 1, 20)
            ctrl_pnt = ctrl_pnt.reshape(2, 4, 2).transpose(0, 2, 1).reshape(4, 4)
            points = np.outer((1 - u) ** 3, ctrl_pnt[:, 0]) \
                + np.outer(3 * u * ((1 - u) ** 2), ctrl_pnt[:, 1]) \
                + np.outer(3 * (u ** 2) * (1 - u), ctrl_pnt[:, 2]) \
                + np.outer(u ** 3, ctrl_pnt[:, 3])
            
            # convert points to polygon
            points = np.concatenate((points[:, :2], points[:, 2:]), axis=0)
            return points.tolist()

    def ctc_decode(self, rec):
        # ctc decoding
        last_char = False
        s = ''
        for c in rec:
            c = int(c)
            if c < self.voc_size - 1:
                if last_char != c:
                    if self.voc_size == 96:
                        s += self.CTLABELS[c]
                        last_char = c
                    else:
                        s += str(chr(self.CTLABELS[c]))
                        last_char = c
            elif c == self.voc_size -1:
                s += u'口'
            else:
                last_char = False
        return s
    
    
    def decode(self, rec):
        s = ''
        for c in rec:
            c = int(c)
            if c < self.voc_size - 1:
                if self.voc_size < 108:
                    if c > len(self.CTLABELS):
                        continue
                    s += self.CTLABELS[c]
                else:
                    s += str(chr(self.CTLABELS[c]))
            elif c == self.voc_size -1:
                s += NULL_CHAR
        if "vintext" in self.dataset_name:
            s = vintext_decoder(s)
        return s

def polygon2rbox(polygon, image_height, image_width):
    poly = np.array(polygon).reshape((-1, 2)).astype(np.float32)
    rect = cv2.minAreaRect(poly)
    corners = cv2.boxPoints(rect)
    corners = np.array(corners, dtype="int")
    pts = get_tight_rect(corners, 0, 0, image_height, image_width, 1)
    pts = np.array(pts).reshape(-1,2)
    pts = pts.tolist()
    return pts

def get_tight_rect(points, start_x, start_y, image_height, image_width, scale):
    points = list(points)
    ps = sorted(points, key=lambda x: x[0])

    if ps[1][1] > ps[0][1]:
        px1 = ps[0][0] * scale + start_x
        py1 = ps[0][1] * scale + start_y
        px4 = ps[1][0] * scale + start_x
        py4 = ps[1][1] * scale + start_y
    else:
        px1 = ps[1][0] * scale + start_x
        py1 = ps[1][1] * scale + start_y
        px4 = ps[0][0] * scale + start_x
        py4 = ps[0][1] * scale + start_y
    if ps[3][1] > ps[2][1]:
        px2 = ps[2][0] * scale + start_x
        py2 = ps[2][1] * scale + start_y
        px3 = ps[3][0] * scale + start_x
        py3 = ps[3][1] * scale + start_y
    else:
        px2 = ps[3][0] * scale + start_x
        py2 = ps[3][1] * scale + start_y
        px3 = ps[2][0] * scale + start_x
        py3 = ps[2][1] * scale + start_y

    px1 = min(max(px1, 1), image_width - 1)
    px2 = min(max(px2, 1), image_width - 1)
    px3 = min(max(px3, 1), image_width - 1)
    px4 = min(max(px4, 1), image_width - 1)
    py1 = min(max(py1, 1), image_height - 1)
    py2 = min(max(py2, 1), image_height - 1)
    py3 = min(max(py3, 1), image_height - 1)
    py4 = min(max(py4, 1), image_height - 1)
    return [px1, py1, px2, py2, px3, py3, px4, py4]

dictionary = "aàáạảãâầấậẩẫăằắặẳẵAÀÁẠẢÃĂẰẮẶẲẴÂẦẤẬẨẪeèéẹẻẽêềếệểễEÈÉẸẺẼÊỀẾỆỂỄoòóọỏõôồốộổỗơờớợởỡOÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠiìíịỉĩIÌÍỊỈĨuùúụủũưừứựửữƯỪỨỰỬỮUÙÚỤỦŨyỳýỵỷỹYỲÝỴỶỸ"


def make_groups():
    groups = []
    i = 0
    while i < len(dictionary) - 5:
        group = [c for c in dictionary[i : i + 6]]
        i += 6
        groups.append(group)
    return groups


groups = make_groups()

TONES = ["", "ˋ", "ˊ", "﹒", "ˀ", "˜"]
SOURCES = ["ă", "â", "Ă", "Â", "ê", "Ê", "ô", "ơ", "Ô", "Ơ", "ư", "Ư", "Đ", "đ"]
TARGETS = ["aˇ", "aˆ", "Aˇ", "Aˆ", "eˆ", "Eˆ", "oˆ", "o˒", "Oˆ", "O˒", "u˒", "U˒", "D-", "d‑"]


def correct_tone_position(word):
    word = word[:-1]
    if len(word) < 2:
        pass
    first_ord_char = ""
    second_order_char = ""
    for char in word:
        for group in groups:
            if char in group:
                second_order_char = first_ord_char
                first_ord_char = group[0]
    if word[-1] == first_ord_char and second_order_char != "":
        pair_chars = ["qu", "Qu", "qU", "QU", "gi", "Gi", "gI", "GI"]
        for pair in pair_chars:
            if pair in word and second_order_char in ["u", "U", "i", "I"]:
                return first_ord_char
        return second_order_char
    return first_ord_char


def vintext_decoder(recognition):
    for char in TARGETS:
        recognition = recognition.replace(char, SOURCES[TARGETS.index(char)])
    if len(recognition) < 1:
        return recognition
    if recognition[-1] in TONES:
        if len(recognition) < 2:
            return recognition
        replace_char = correct_tone_position(recognition)
        tone = recognition[-1]
        recognition = recognition[:-1]
        for group in groups:
            if replace_char in group:
                recognition = recognition.replace(replace_char, group[TONES.index(tone)])
    return recognition
