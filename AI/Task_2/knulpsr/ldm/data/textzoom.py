import io
import glob
import json
import random
import os
from collections import OrderedDict
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
import lmdb
import six
import sys
import bisect
import warnings
from PIL import Image
import numpy as np
import string
import cv2
import os
import re
import xml.etree.ElementTree as ET
import imgaug.augmenters as iaa
from tqdm import tqdm

from utils import utils_deblur
from utils import utils_sisr as sr
from einops import rearrange, repeat
from matplotlib import pyplot as plt

kernel = utils_deblur.fspecial('gaussian', 15, 1.)
_CHINESE_ALPHABET = None


def buf2PIL(txn, key, type='RGB'):
    imgbuf = txn.get(key)
    buf = six.BytesIO()
    buf.write(imgbuf)
    buf.seek(0)
    im = Image.open(buf).convert(type)
    return im


def gauss_unsharp_mask(rgb, shp_kernel, shp_sigma, shp_gain):
    LF = cv2.GaussianBlur(rgb, (shp_kernel, shp_kernel), shp_sigma)
    HF = rgb - LF
    RGB_peak = rgb + HF * shp_gain
    RGB_noise_NR_shp = np.clip(RGB_peak, 0.0, 255.0)
    return RGB_noise_NR_shp, LF


def add_shot_gauss_noise(rgb, shot_noise_mean, read_noise):
    noise_var_map = shot_noise_mean * rgb + read_noise
    noise_dev_map = np.sqrt(noise_var_map)
    noise = np.random.normal(loc=0.0, scale=noise_dev_map, size=None)
    if (rgb.mean() > 252.0):
        noise_rgb = rgb
    else:
        noise_rgb = rgb + noise
    noise_rgb = np.clip(noise_rgb, 0.0, 255.0)
    return noise_rgb


def degradation(src_img):
    # RGB Image input
    GT_RGB = np.array(src_img)
    GT_RGB = GT_RGB.astype(np.float32)

    pre_blur_kernel_set = [3, 5]
    sharp_kernel_set = [3, 5]
    blur_kernel_set = [5, 7, 9, 11]
    NR_kernel_set = [3, 5]

    # Pre Blur
    kernel = pre_blur_kernel_set[random.randint(0, (len(pre_blur_kernel_set) - 1))]
    blur_sigma = random.uniform(5., 6.)
    RGB_pre_blur = cv2.GaussianBlur(GT_RGB, (kernel, kernel), blur_sigma)

    rand_p = random.random()
    if rand_p > 0.2:
        # Noise
        shot_noise = random.uniform(0, 0.005)
        read_noise = random.uniform(0, 0.015)
        GT_RGB_noise = add_shot_gauss_noise(RGB_pre_blur, shot_noise, read_noise)
    else:
        GT_RGB_noise = RGB_pre_blur

    # Noise Reduction
    choice = random.uniform(0, 1.0)
    GT_RGB_noise = np.round(GT_RGB_noise)
    GT_RGB_noise = GT_RGB_noise.astype(np.uint8)
    # if (shot_noise < 0.06):
    if (choice < 0.7):
        NR_kernel = NR_kernel_set[random.randint(0, (len(NR_kernel_set) - 1))]  ###3,5,7,9
        NR_sigma = random.uniform(2., 3.)
        GT_RGB_noise_NR = cv2.GaussianBlur(GT_RGB_noise, (NR_kernel, NR_kernel), NR_sigma)
    else:
        value_sigma = random.uniform(70, 80)
        space_sigma = random.uniform(70, 80)
        GT_RGB_noise_NR = cv2.bilateralFilter(GT_RGB_noise, 7, value_sigma, space_sigma)

    # Sharpening
    GT_RGB_noise_NR = GT_RGB_noise_NR.astype(np.float32)
    shp_kernel = sharp_kernel_set[random.randint(0, (len(sharp_kernel_set) - 1))]  ###5,7,9
    shp_sigma = random.uniform(2., 3.)
    shp_gain = random.uniform(3., 4.)
    RGB_noise_NR_shp, LF = gauss_unsharp_mask(GT_RGB_noise_NR, shp_kernel, shp_sigma, shp_gain)

    # print("RGB_noise_NR_shp:", RGB_noise_NR_shp.shape)

    return Image.fromarray(RGB_noise_NR_shp.astype(np.uint8))


def str_filt(str_, voc_type):
    global _CHINESE_ALPHABET
    if _CHINESE_ALPHABET is None:
        alphabet_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "al_chinese.txt"))
        with open(alphabet_path, "r", encoding="utf-8") as f:
            _CHINESE_ALPHABET = f.readline().replace("\n", "")

    alpha_dict = {
        'digit': string.digits,
        'lower': string.digits + string.ascii_lowercase,
        'upper': string.digits + string.ascii_letters,
        'all': string.digits + string.ascii_letters + string.punctuation,
        'chinese': _CHINESE_ALPHABET
    }
    if voc_type == 'lower':
        str_ = str_.lower()

    if voc_type == 'chinese':  # Chinese character only
        new_str = ""
        for ch in str_:
            if '\u4e00' <= ch <= '\u9fa5' or ch in string.digits + string.ascii_letters:
                new_str += ch
        str_ = new_str
    for char in str_:
        if char not in alpha_dict[voc_type]:  # voc_type
            str_ = str_.replace(char, '')
    return str_


class lmdbDataset_real(Dataset):
    def __init__(
            self, root=None,
            voc_type='upper',
            max_len=100,
            test=False,
            cutblur=False,
            manmade_degrade=False,
            rotate=None,
            ocr_data=False
    ):
        super(lmdbDataset_real, self).__init__()
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        self.cb_flag = cutblur
        self.rotate = rotate

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get(b'num-samples'))
            self.nSamples = nSamples
            print("nSamples:", nSamples)
        self.voc_type = voc_type
        self.max_len = max_len
        self.test = test

        self.manmade_degrade = manmade_degrade
        self.ocr_data = ocr_data

    def __len__(self):
        return self.nSamples

    def rotate_img(self, image, angle):
        # convert to cv2 image

        if not angle == 0.0:
            image = np.array(image)
            (h, w) = image.shape[:2]
            scale = 1.0
            # set the rotation center
            center = (w / 2, h / 2)
            # anti-clockwise angle in the function
            M = cv2.getRotationMatrix2D(center, angle, scale)
            image = cv2.warpAffine(image, M, (w, h))
            # back to PIL image
            image = Image.fromarray(image)

        return image

    def cutblur(self, img_hr, img_lr):
        p = random.random()

        img_hr_np = np.array(img_hr)
        img_lr_np = np.array(img_lr)

        randx = int(img_hr_np.shape[1] * (0.2 + 0.8 * random.random()))

        if p > 0.7:
            left_mix = random.random()
            if left_mix <= 0.5:
                img_lr_np[:, randx:] = img_hr_np[:, randx:]
            else:
                img_lr_np[:, :randx] = img_hr_np[:, :randx]

        return Image.fromarray(img_lr_np)

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        txn = self.env.begin(write=False)
        label_key = b'label-%09d' % index
        word = "  "  # str(txn.get(label_key).decode())
        # print("in dataset....")
        img_HR_key = b'image_hr-%09d' % index  # 128*32
        img_lr_key = b'image_lr-%09d' % index  # 64*16

        if self.ocr_data:
            img_HR_key = img_lr_key = b'image-%09d' % index

        try:
            img_HR = buf2PIL(txn, img_HR_key, 'RGB')
            if self.manmade_degrade:
                img_lr = degradation(img_HR)
            else:
                img_lr = buf2PIL(txn, img_lr_key, 'RGB')
            # print("GOGOOGO..............", img_HR.size)
            if self.cb_flag and not self.test:
                img_lr = self.cutblur(img_HR, img_lr)

            if not self.rotate is None:

                if not self.test:
                    angle = random.random() * self.rotate * 2 - self.rotate
                else:
                    angle = 0  # self.rotate

                # img_HR = self.rotate_img(img_HR, angle)
                # img_lr = self.rotate_img(img_lr, angle)

            img_lr_np = np.array(img_lr).astype(np.uint8)
            img_lry = cv2.cvtColor(img_lr_np, cv2.COLOR_RGB2YUV)
            img_lry = Image.fromarray(img_lry)

            img_HR_np = np.array(img_HR).astype(np.uint8)
            img_HRy = cv2.cvtColor(img_HR_np, cv2.COLOR_RGB2YUV)
            img_HRy = Image.fromarray(img_HRy)
            word = txn.get(label_key)
            if word is None:
                print("None word:", label_key)
                word = " "
            else:
                word = str(word.decode())
            # print("img_HR:", img_HR.size, img_lr.size())

        except IOError or len(word) > self.max_len:
            return self[index + 1]
        label_str = str_filt(word, self.voc_type)

        return img_HR, img_lr, img_HRy, img_lry, label_str, index


class multi_lmdbDataset(ConcatDataset):
    def __init__(self, roots):
        datasets = []
        for path in roots:
            datasets.append(lmdbDataset_real(root=path, voc_type='all'))
        super(multi_lmdbDataset, self).__init__(datasets)


class plateFolderDataset(Dataset):
    def __init__(
            self,
            roots,
            voc_type='all',
            max_len=100,
            num_lr=5,
            num_hr=5,
            hr_target_index=1,
            random_hr=False,
            random_main_lr=False,
            main_lr_index=1,
            sync_hr_with_main_lr=False,
            hr_select_by_parseq=False,
            hr_select_parseq_checkpoint_path='parseq/weights/parseq.pt',
            hr_select_parseq_repo_path='parseq',
            hr_select_parseq_use_format_constraint=True,
            hr_select_parseq_device='cuda',
            hr_select_debug=False,
            hr_select_debug_max_samples=20,
            hr_select_debug_output=None,
            annotation_required=True,
            fallback_to_plate_name=False,
            hr_select_by_crnn=None,
            crnn_recognizer_path=None,
            crnn_imgH=32,
            crnn_imgW=100,
            crnn_use_cuda=None
    ):
        super(plateFolderDataset, self).__init__()
        if isinstance(roots, str):
            roots = [roots]
        self.roots = roots
        self.voc_type = voc_type
        self.max_len = max_len
        self.num_lr = num_lr
        self.num_hr = num_hr
        self.hr_target_index = max(1, int(hr_target_index))
        self.random_hr = random_hr
        self.random_main_lr = random_main_lr
        self.main_lr_index = max(1, int(main_lr_index))
        self.sync_hr_with_main_lr = sync_hr_with_main_lr
        if hr_select_by_crnn is not None:
            warnings.warn(
                "[plateFolderDataset] `hr_select_by_crnn` is deprecated; using PARSeq-based HR selection instead.",
                stacklevel=2
            )
            hr_select_by_parseq = bool(hr_select_by_parseq or hr_select_by_crnn)
        if hr_select_by_parseq and crnn_recognizer_path:
            warnings.warn(
                "[plateFolderDataset] `crnn_recognizer_path` is ignored; use `hr_select_parseq_checkpoint_path` instead.",
                stacklevel=2
            )
        if hr_select_by_parseq and crnn_use_cuda is not None:
            hr_select_parseq_device = "cuda" if crnn_use_cuda else "cpu"

        self.hr_select_by_parseq = bool(hr_select_by_parseq)
        self.hr_select_parseq_checkpoint_path = hr_select_parseq_checkpoint_path
        self.hr_select_parseq_repo_path = hr_select_parseq_repo_path
        self.hr_select_parseq_use_format_constraint = bool(hr_select_parseq_use_format_constraint)
        self.hr_select_parseq_device = hr_select_parseq_device
        if not os.path.isabs(self.hr_select_parseq_checkpoint_path):
            self.hr_select_parseq_checkpoint_path = os.path.abspath(
                os.path.join(os.getcwd(), self.hr_select_parseq_checkpoint_path)
            )
        if not os.path.isabs(self.hr_select_parseq_repo_path):
            self.hr_select_parseq_repo_path = os.path.abspath(
                os.path.join(os.getcwd(), self.hr_select_parseq_repo_path)
            )
        self.hr_select_debug = bool(hr_select_debug)
        self.hr_select_debug_max_samples = int(hr_select_debug_max_samples)
        self.hr_select_debug_output = hr_select_debug_output
        self.annotation_required = annotation_required
        self.fallback_to_plate_name = fallback_to_plate_name
        self._hr_select_model = None
        self._hr_select_device = torch.device("cpu")
        self._hr_select_alphabet = "-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.samples = self._build_index()

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No valid plate samples found. roots={self.roots}, "
                f"expected files like plate-*/lr-001.* and hr-001.*"
            )
        if self.hr_select_by_parseq:
            self._select_target_hr_with_parseq()
        print(f"plateFolderDataset: loaded {len(self.samples)} samples from {len(self.roots)} roots.")

    @staticmethod
    def _find_numbered_image(folder, prefix, idx):
        prefix_candidates = [prefix.lower(), prefix.upper(), prefix.capitalize()]
        seen = set()
        for cand_prefix in prefix_candidates:
            stem = f"{cand_prefix}-{idx:03d}"
            if stem in seen:
                continue
            seen.add(stem)
            candidates = sorted(glob.glob(os.path.join(folder, stem + ".*")))
            if len(candidates) > 0:
                for candidate in candidates:
                    if plateFolderDataset._is_image_file(candidate):
                        return candidate
                return candidates[0]
            direct = os.path.join(folder, stem)
            if os.path.isfile(direct):
                return direct
        return None

    @staticmethod
    def _parse_label_from_plate_dir(plate_dir):
        dirname = os.path.basename(plate_dir)
        if dirname.lower().startswith("plate-"):
            label = dirname[len("plate-"):]
        else:
            label = dirname
        if label is None or len(label.strip()) == 0:
            label = " "
        return label

    @staticmethod
    def _is_image_file(path):
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        return os.path.splitext(path)[1].lower() in image_exts

    @staticmethod
    def _extract_json_text(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(key, str) and key.lower() == "plate" and isinstance(value, str):
                    text = value.strip()
                    if len(text) > 0:
                        return text
            for value in obj.values():
                nested = plateFolderDataset._extract_json_text(value)
                if nested is not None and len(nested) > 0:
                    return nested
        elif isinstance(obj, list):
            for value in obj:
                nested = plateFolderDataset._extract_json_text(value)
                if nested is not None and len(nested) > 0:
                    return nested
        return None

    @staticmethod
    def _read_annotation_text(path):
        ext = os.path.splitext(path)[1].lower()
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read()

            raw_strip = raw.strip()
            if ext == ".json" or raw_strip.startswith("{") or raw_strip.startswith("["):
                try:
                    obj = json.loads(raw)
                    text = plateFolderDataset._extract_json_text(obj)
                    return text.strip() if isinstance(text, str) else None
                except Exception:
                    pass

            if ext == ".xml":
                root = ET.parse(path).getroot()
                for tag in ["plate", "Plate", "PLATE"]:
                    node = root.find(f".//{tag}")
                    if node is not None and node.text is not None and len(node.text.strip()) > 0:
                        return node.text.strip()
                return None

            # Plain text: only accept explicit `plate: XXXXX` forms.
            for line in raw.splitlines():
                line = line.strip()
                if len(line) == 0:
                    continue
                m = re.search(r'(?i)\bplate\b\s*[:=]\s*["\']?([0-9a-zA-Z-]+)', line)
                if m is not None:
                    return m.group(1).strip()
            return None
        except Exception:
            return None

    def _find_annotation_text_for_hr(self, hr_path):
        folder = os.path.dirname(hr_path)
        stem = os.path.splitext(os.path.basename(hr_path))[0]

        candidate_paths = []
        extensioned = sorted(glob.glob(os.path.join(folder, stem + ".*")))
        for p in extensioned:
            if p == hr_path:
                continue
            if self._is_image_file(p):
                continue
            candidate_paths.append(p)
        no_ext_path = os.path.join(folder, stem)
        if os.path.isfile(no_ext_path) and not self._is_image_file(no_ext_path):
            candidate_paths.append(no_ext_path)

        for ann_path in candidate_paths:
            text = self._read_annotation_text(ann_path)
            if text is not None and len(text) > 0:
                return text
        return None

    @staticmethod
    def _normalize_label_for_compare(text):
        if text is None:
            return ""
        text = text.upper()
        text = re.sub(r'[^0-9A-Z]+', '', text)
        return text

    def _get_sample_gt_label(self, hr_target_path, hr_paths, plate_dir):
        label_raw = self._find_annotation_text_for_hr(hr_target_path)
        if label_raw is None:
            for hp in hr_paths:
                label_raw = self._find_annotation_text_for_hr(hp)
                if label_raw is not None:
                    break
        if label_raw is None and self.fallback_to_plate_name:
            label_raw = self._parse_label_from_plate_dir(plate_dir)
        if label_raw is None:
            return None
        label = str_filt(label_raw, self.voc_type)
        label = label.upper()
        if len(label) > self.max_len:
            label = label[:self.max_len]
        return label

    def _init_parseq_selector(self):
        from ldm.modules.encoders.tp_generator import PARSeqTPG

        if not os.path.isfile(self.hr_select_parseq_checkpoint_path):
            raise RuntimeError(
                f"PARSeq checkpoint not found: {self.hr_select_parseq_checkpoint_path}"
            )
        if not os.path.isdir(self.hr_select_parseq_repo_path):
            raise RuntimeError(
                f"PARSeq repo not found: {self.hr_select_parseq_repo_path}"
            )

        model = PARSeqTPG(
            checkpoint_path=self.hr_select_parseq_checkpoint_path,
            parseq_repo_path=self.hr_select_parseq_repo_path,
            device=self.hr_select_parseq_device,
            use_format_constraint=self.hr_select_parseq_use_format_constraint,
            freeze_backbone=True,
        )
        model.eval()
        self._hr_select_model = model
        self._hr_select_device = model.device

    def _prepare_parseq_input(self, pil_image):
        return transforms.ToTensor()(pil_image).unsqueeze(0).to(self._hr_select_device)

    def _decode_parseq_with_confidence(self, output):
        if torch.is_tensor(output) and output.dim() == 3:
            probs = output
        else:
            raise RuntimeError(f"Unexpected PARSeq output format for HR selection: {type(output)}")

        if not getattr(self._hr_select_model, "output_is_prob", False):
            probs = torch.softmax(probs, dim=-1)

        token_probs, token_indices = probs.max(dim=-1)
        token_probs = token_probs[0]
        token_indices = token_indices[0]

        pred_text = ""
        kept_probs = []
        for idx, prob in zip(token_indices.tolist(), token_probs.tolist()):
            if idx <= 0:
                break
            if idx >= len(self._hr_select_alphabet):
                break
            pred_text += self._hr_select_alphabet[idx]
            kept_probs.append(float(prob))

        if len(kept_probs) > 0:
            conf = float(np.mean(kept_probs))
        else:
            conf = float(token_probs.mean().item())
        return pred_text.upper(), conf

    def _predict_parseq_text_and_conf(self, image_path):
        img = Image.open(image_path).convert("RGB")
        inp = self._prepare_parseq_input(img)
        with torch.no_grad():
            probs = self._hr_select_model(inp)
        return self._decode_parseq_with_confidence(probs)

    def _select_target_hr_with_parseq(self):
        self._init_parseq_selector()
        matched_cnt = 0
        gt_nonempty_cnt = 0
        debug_rows = []
        for sample in tqdm(self.samples, desc="Selecting target HR with PARSeq"):
            gt_norm = self._normalize_label_for_compare(sample.get("gt_label", ""))
            if len(gt_norm) > 0:
                gt_nonempty_cnt += 1
            best_path = sample["hr_target_path"]
            best_match = -1
            best_conf = -1.0
            best_pred_text = ""
            best_pred_norm = ""
            cand_rows = []

            for hr_path in sample["hr_paths"]:
                pred_text, conf = self._predict_parseq_text_and_conf(hr_path)
                pred_norm = self._normalize_label_for_compare(pred_text)
                is_match = 1 if (len(gt_norm) > 0 and pred_norm == gt_norm) else 0
                if self.hr_select_debug and len(debug_rows) < self.hr_select_debug_max_samples:
                    cand_rows.append({
                        "hr_file": os.path.basename(hr_path),
                        "pred_text": pred_text,
                        "pred_norm": pred_norm,
                        "conf": round(float(conf), 6),
                        "is_match": int(is_match)
                    })
                if (is_match > best_match) or (is_match == best_match and conf > best_conf):
                    best_match = is_match
                    best_conf = conf
                    best_path = hr_path
                    best_pred_text = pred_text
                    best_pred_norm = pred_norm

            sample["hr_target_path"] = best_path
            if best_match == 1:
                matched_cnt += 1
            if self.hr_select_debug and len(debug_rows) < self.hr_select_debug_max_samples:
                debug_rows.append({
                    "plate_dir": sample.get("plate_dir", ""),
                    "gt_label_raw": sample.get("gt_label", ""),
                    "gt_norm": gt_norm,
                    "best_hr_file": os.path.basename(best_path),
                    "best_pred_text": best_pred_text,
                    "best_pred_norm": best_pred_norm,
                    "best_match": int(best_match),
                    "best_conf": round(float(best_conf), 6),
                    "candidates": cand_rows
                })

        total = len(self.samples)
        print(
            f"[plateFolderDataset] PARSeq HR selection done: GT-matched {matched_cnt}/{total} "
            f"(GT non-empty: {gt_nonempty_cnt}/{total})"
        )
        if self.hr_select_debug:
            print(f"[plateFolderDataset][debug] showing {len(debug_rows)} sample decisions")
            for i, row in enumerate(debug_rows):
                print(
                    f"[plate-debug:{i}] gt='{row['gt_label_raw']}' norm='{row['gt_norm']}' "
                    f"best='{row['best_hr_file']}' pred='{row['best_pred_text']}' "
                    f"pred_norm='{row['best_pred_norm']}' match={row['best_match']} conf={row['best_conf']}"
                )
            if isinstance(self.hr_select_debug_output, str) and len(self.hr_select_debug_output) > 0:
                try:
                    out_dir = os.path.dirname(self.hr_select_debug_output)
                    if len(out_dir) > 0:
                        os.makedirs(out_dir, exist_ok=True)
                    with open(self.hr_select_debug_output, "w", encoding="utf-8") as f:
                        json.dump({
                            "matched_cnt": matched_cnt,
                            "total": total,
                            "gt_nonempty_cnt": gt_nonempty_cnt,
                            "rows": debug_rows
                        }, f, ensure_ascii=False, indent=2)
                    print(f"[plateFolderDataset][debug] wrote {self.hr_select_debug_output}")
                except Exception as exc:
                    print(f"[plateFolderDataset][debug] failed to write debug json: {exc}")
        self._hr_select_model = None

    def _build_index(self):
        samples = []
        for root in self.roots:
            if not os.path.isdir(root):
                warnings.warn(f"[plateFolderDataset] root not found: {root}")
                continue

            plate_dir_set = set()
            for pattern in ["plate-*", "Plate-*", "PLATE-*"]:
                for p in glob.glob(os.path.join(root, "**", pattern), recursive=True):
                    if os.path.isdir(p):
                        plate_dir_set.add(p)
            plate_dirs = sorted(list(plate_dir_set))

            for plate_dir in plate_dirs:
                lr_paths = []
                for i in range(1, self.num_lr + 1):
                    lr_path = self._find_numbered_image(plate_dir, "lr", i)
                    if lr_path is None:
                        lr_paths = []
                        break
                    lr_paths.append(lr_path)
                if len(lr_paths) != self.num_lr:
                    continue

                hr_paths = []
                for i in range(1, self.num_hr + 1):
                    hr_path = self._find_numbered_image(plate_dir, "hr", i)
                    if hr_path is not None:
                        hr_paths.append(hr_path)
                if len(hr_paths) == 0:
                    continue

                hr_target_idx = min(self.hr_target_index - 1, len(hr_paths) - 1)
                hr_target_path = hr_paths[hr_target_idx]
                gt_label = self._get_sample_gt_label(hr_target_path, hr_paths, plate_dir)
                if self.annotation_required and gt_label is None:
                    continue
                samples.append({
                    "plate_dir": plate_dir,
                    "lr_paths": lr_paths,
                    "hr_paths": hr_paths,
                    "hr_target_path": hr_target_path,
                    "gt_label": gt_label if gt_label is not None else " "
                })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        if self.random_main_lr:
            main_lr_idx = random.randint(0, len(sample["lr_paths"]) - 1)
        else:
            main_lr_idx = min(self.main_lr_index - 1, len(sample["lr_paths"]) - 1)

        if self.sync_hr_with_main_lr:
            if len(sample["hr_paths"]) > 0:
                hr_idx = min(main_lr_idx, len(sample["hr_paths"]) - 1)
                hr_path = sample["hr_paths"][hr_idx]
            else:
                hr_path = sample["hr_target_path"]
        elif self.random_hr:
            hr_path = random.choice(sample["hr_paths"])
        else:
            hr_path = sample["hr_target_path"]

        img_HR = Image.open(hr_path).convert("RGB")
        hr_images_all = [Image.open(p).convert("RGB") for p in sample["hr_paths"]]
        img_HR_np = np.array(img_HR).astype(np.uint8)
        img_HRy = cv2.cvtColor(img_HR_np, cv2.COLOR_RGB2YUV)
        img_HRy = Image.fromarray(img_HRy)

        ordered_lr_paths = [sample["lr_paths"][main_lr_idx]] + [
            p for i, p in enumerate(sample["lr_paths"]) if i != main_lr_idx
        ]

        lr_images = []
        for lr_path in ordered_lr_paths:
            lr_img = Image.open(lr_path).convert("RGB")
            lr_images.append(lr_img)

        # Keep compatibility with existing tuple format:
        # img_lr is now List[PIL] for multi-frame LR condition.
        img_lry = cv2.cvtColor(np.array(lr_images[0]).astype(np.uint8), cv2.COLOR_RGB2YUV)
        img_lry = Image.fromarray(img_lry)

        label_raw = self._find_annotation_text_for_hr(hr_path)
        if label_raw is None:
            label_raw = sample.get("gt_label", None)
        if label_raw is None and self.fallback_to_plate_name:
            label_raw = self._parse_label_from_plate_dir(sample["plate_dir"])
        if label_raw is None:
            if self.annotation_required:
                raise RuntimeError(
                    f"Annotation not found for target HR file: {hr_path}. "
                    f"Expected non-image file with same basename."
                )
            label_raw = " "

        label_str = str_filt(label_raw, self.voc_type)
        label_str = label_str.upper()
        if len(label_str) > self.max_len:
            label_str = label_str[:self.max_len]

        return img_HR, lr_images, img_HRy, img_lry, label_str, index, hr_images_all


class resizeNormalize(object):
    def __init__(self, size, mask=False, interpolation=Image.BICUBIC, aug=None, blur=False):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()
        self.mask = mask
        self.aug = aug

        self.blur = blur

    def __call__(self, img, ratio_keep=False):

        size = self.size

        if ratio_keep:
            ori_width, ori_height = img.size
            ratio = float(ori_width) / ori_height

            if ratio < 3:
                width = 100  # if self.size[0] == 32 else 50
            else:
                width = int(ratio * self.size[1])

            size = (width, self.size[1])

        # print("size:", size)
        img = img.resize(size, self.interpolation)

        if self.blur:
            # img_np = np.array(img)
            # img_np = cv2.GaussianBlur(img_np, (5, 5), 1)
            # print("in degrade:", np.unique(img_np))
            # img_np = noisy("gauss", img_np).astype(np.uint8)
            # img_np = apply_brightness_contrast(img_np, 40, 40).astype(np.uint8)
            # img_np = JPEG_compress(img_np)

            # img = Image.fromarray(img_np)
            pass

        if not self.aug is None:
            img_np = np.array(img)
            # print("imgaug_np:", imgaug_np.shape)
            imgaug_np = self.aug(images=img_np[None, ...])
            img = Image.fromarray(imgaug_np[0, ...])

        img_tensor = self.toTensor(img)
        if self.mask:
            mask = img.convert('L')
            thres = np.array(mask).mean()
            mask = mask.point(lambda x: 0 if x > thres else 255)
            mask = self.toTensor(mask)
            img_tensor = torch.cat((img_tensor, mask), 0)

        return img_tensor


class alignCollate_syn(object):
    def __init__(self, imgH=64,
                 imgW=256,
                 down_sample_scale=4,
                 keep_ratio=False,
                 min_ratio=1,
                 mask=False,
                 alphabet=53,
                 train=True,
                 y_domain=False
                 ):

        sometimes = lambda aug: iaa.Sometimes(0.2, aug)

        aug = [
            iaa.GaussianBlur(sigma=(0.0, 3.0)),
            iaa.AverageBlur(k=(1, 5)),
            iaa.MedianBlur(k=(3, 7)),
            iaa.BilateralBlur(
                d=(3, 9), sigma_color=(10, 250), sigma_space=(10, 250)),
            iaa.MotionBlur(k=3),
            iaa.MeanShiftBlur(),
            iaa.Superpixels(p_replace=(0.1, 0.5), n_segments=(1, 7))
        ]

        self.aug = iaa.Sequential([sometimes(a) for a in aug], random_order=True)

        # self.y_domain = y_domain

        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio
        self.down_sample_scale = down_sample_scale
        self.mask = mask
        # self.alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
        self.alphabet = open("al_chinese.txt", "r", encoding="utf-8").readlines()[0].replace("\n", "")
        self.d2a = "-" + self.alphabet
        self.alsize = len(self.d2a)
        self.a2d = {}
        cnt = 0
        for ch in self.d2a:
            self.a2d[ch] = cnt
            cnt += 1

        imgH = self.imgH
        imgW = self.imgW

        self.transform = resizeNormalize((imgW, imgH), self.mask)
        self.transform2 = resizeNormalize((imgW // self.down_sample_scale, imgH // self.down_sample_scale), self.mask,
                                          blur=True)
        self.transform_pseudoLR = resizeNormalize((imgW // self.down_sample_scale, imgH // self.down_sample_scale),
                                                  self.mask, aug=self.aug)

        self.train = train

    def degradation(self, img_L):
        # degradation process, blur + bicubic downsampling + Gaussian noise
        # if need_degradation:
        # img_L = util.modcrop(img_L, sf)
        img_L = np.array(img_L)
        # print("img_L_before:", img_L.shape, np.unique(img_L))
        img_L = sr.srmd_degradation(img_L, kernel)

        noise_level_img = 0.
        if not self.train:
            np.random.seed(seed=0)  # for reproducibility
        # print("unique:", np.unique(img_L))
        img_L = img_L + np.random.normal(0, noise_level_img, img_L.shape)

        # print("img_L_after:", img_L_beore.shape, img_L.shape, np.unique(img_L))

        return Image.fromarray(img_L.astype(np.uint8))

    def __call__(self, batch):
        images, images_lr, _, _, label_strs = zip(*batch)

        # [self.degradation(image) for image in images]
        # images_hr = images
        '''
        images_lr = [image.resize(
            (image.size[0] // self.down_sample_scale, image.size[1] // self.down_sample_scale),
            Image.BICUBIC) for image in images]

        if self.train:
            if random.random() > 1.5:
                images_hr = [image.resize(
                (image.size[0]//self.down_sample_scale, image.size[1]//self.down_sample_scale),
                Image.BICUBIC) for image in images]
            else:
                images_hr = images
        else:
            images_hr = images
            #[image.resize(
            #    (image.size[0] // self.down_sample_scale, image.size[1] // self.down_sample_scale),
            #    Image.BICUBIC) for image in images]
        '''
        # images_hr = [self.degradation(image) for image in images]
        images_hr = images
        # images_lr = [image.resize(
        #     (image.size[0] // 4, image.size[1] // 4),
        #     Image.BICUBIC) for image in images_lr]
        # images_lr = images

        # images_lr_new = []
        # for image in images_lr:
        #    image_np = np.array(image)
        #    image_aug = self.aug(images=image_np[None, ])[0]
        #    images_lr_new.append(Image.fromarray(image_aug))
        # images_lr = images_lr_new

        images_hr = [self.transform(image) for image in images_hr]
        images_hr = torch.cat([t.unsqueeze(0) for t in images_hr], 0)

        if self.train:
            images_lr = [image.resize(
                (image.size[0] // 2, image.size[1] // 2),  # self.down_sample_scale
                Image.BICUBIC) for image in images_lr]
        else:
            pass
        #    # for image in images_lr:
        #    #     print("images_lr:", image.size)
        #    images_lr = [image.resize(
        #         (image.size[0] // self.down_sample_scale, image.size[1] // self.down_sample_scale),  # self.down_sample_scale
        #        Image.BICUBIC) for image in images_lr]
        #    pass
        # images_lr = [self.degradation(image) for image in images]
        images_lr = [self.transform2(image) for image in images_lr]

        images_lr = torch.cat([t.unsqueeze(0) for t in images_lr], 0)

        max_len = 26

        label_batches = []
        weighted_tics = []
        weighted_masks = []

        for word in label_strs:
            word = word.lower()
            # Complement

            if len(word) > 4:
                word = [ch for ch in word]
                word[2] = "e"
                word = "".join(word)

            if len(word) <= 1:
                pass
            elif len(word) < 26 and len(word) > 1:
                # inter_com = 26 - len(word)
                # padding = int(inter_com / (len(word) - 1))
                # new_word = word[0]
                # for i in range(len(word) - 1):
                #    new_word += "-" * padding + word[i + 1]

                # word = new_word
                pass
            else:
                word = word[:26]

            label_list = [self.a2d[ch] for ch in word if ch in self.a2d]

            if len(label_list) <= 0:
                # blank label
                weighted_masks.append(0)
            else:
                weighted_masks.extend(label_list)

            labels = torch.tensor(label_list)[:, None].long()
            label_vecs = torch.zeros((labels.shape[0], self.alsize))
            # print("labels:", labels)
            # if labels.shape[0] > 0:
            #    label_batches.append(label_vecs.scatter_(-1, labels, 1))
            # else:
            #    label_batches.append(label_vecs)

            if labels.shape[0] > 0:
                label_vecs = torch.zeros((labels.shape[0], self.alsize))
                label_batches.append(label_vecs.scatter_(-1, labels, 1))
                weighted_tics.append(1)
            else:
                label_vecs = torch.zeros((1, self.alsize))
                label_vecs[0, 0] = 1.
                label_batches.append(label_vecs)
                weighted_tics.append(0)

        label_rebatches = torch.zeros((len(label_strs), max_len, self.alsize))

        for idx in range(len(label_strs)):
            label_rebatches[idx][:label_batches[idx].shape[0]] = label_batches[idx]

        label_rebatches = label_rebatches.unsqueeze(1).float().permute(0, 3, 1, 2)

        # print(images_lr.shape, images_hr.shape)

        return images_hr, images_lr, images_hr, images_lr, label_strs, label_rebatches, torch.tensor(
            weighted_masks).long(), torch.tensor(weighted_tics)


class alignCollate_realWTL(alignCollate_syn):
    def __call__(self, batch):
        images_HR, images_lr, images_HRy, images_lry, label_strs, indexes, images_HR_all = zip(*batch)
        imgH = self.imgH
        imgW = self.imgW
        # transform = resizeNormalize((imgW, imgH), self.mask)
        # transform2 = resizeNormalize((imgW // self.down_sample_scale, imgH // self.down_sample_scale), self.mask)
        images_HR = [self.transform(image) for image in images_HR]
        images_HR = torch.cat([t.unsqueeze(0) for t in images_HR], 0)

        multi_lr_mode = isinstance(images_lr[0], (list, tuple))
        if multi_lr_mode:
            images_lr = [
                torch.stack([self.transform2(frame) for frame in frame_group], dim=0)
                for frame_group in images_lr
            ]
            images_lr = torch.stack(images_lr, dim=0)  # B x T x C x H x W
        else:
            images_lr = [self.transform2(image) for image in images_lr]
            images_lr = torch.cat([t.unsqueeze(0) for t in images_lr], 0)

        images_lry = [self.transform2(image) for image in images_lry]
        images_lry = torch.cat([t.unsqueeze(0) for t in images_lry], 0)

        images_HRy = [self.transform(image) for image in images_HRy]
        images_HRy = torch.cat([t.unsqueeze(0) for t in images_HRy], 0)

        images_HR_all = [
            torch.stack([self.transform(frame) for frame in frame_group], dim=0)
            for frame_group in images_HR_all
        ]
        images_HR_all = torch.stack(images_HR_all, dim=0)  # B x T x C x H x W

        max_len = 26

        label_batches = []

        for word in label_strs:
            word = word.lower()
            # Complement

            if len(word) > 4:
                word = [ch for ch in word]
                word[2] = "e"
                word = "".join(word)

            if len(word) <= 1:
                pass
            elif len(word) < 26 and len(word) > 1:
                inter_com = 26 - len(word)
                padding = int(inter_com / (len(word) - 1))
                new_word = word[0]
                for i in range(len(word) - 1):
                    new_word += "-" * padding + word[i + 1]

                word = new_word
                pass
            else:
                word = word[:26]

            label_list = [self.a2d[ch] for ch in word if ch in self.a2d]

            labels = torch.tensor(label_list)[:, None].long()
            label_vecs = torch.zeros((labels.shape[0], self.alsize))
            # print("labels:", labels)
            if labels.shape[0] > 0:
                label_batches.append(label_vecs.scatter_(-1, labels, 1))
            else:
                label_batches.append(label_vecs)
        label_rebatches = torch.zeros((len(label_strs), max_len, self.alsize))

        for idx in range(len(label_strs)):
            label_rebatches[idx][:label_batches[idx].shape[0]] = label_batches[idx]

        label_rebatches = label_rebatches.unsqueeze(1).float().permute(0, 3, 1, 2)

        images_HR = rearrange(images_HR, 'b c h w -> b h w c')
        images_HR_all = rearrange(images_HR_all, 'b t c h w -> b t h w c')
        if multi_lr_mode:
            images_lr = rearrange(images_lr, 'b t c h w -> b t h w c')
        else:
            images_lr = rearrange(images_lr, 'b c h w -> b h w c')

        example = {
            'image': images_HR,
            'HR_all_images': images_HR_all,
            'LR_image': images_lr,
            'label': label_strs,
            'id': indexes
        }
        return example

        # return images_HR, images_lr, images_HRy, images_lry, label_strs, label_rebatches


class alignCollate_realWTL_forVQGAN(alignCollate_syn):
    def __init__(self, *args, use_all_hr_frames=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_all_hr_frames = bool(use_all_hr_frames)

    def __call__(self, batch):
        use_all_hr = self.use_all_hr_frames and len(batch[0]) >= 7

        if len(batch[0]) >= 7:
            images_HR, images_lr, images_HRy, images_lry, label_strs, indexes, images_HR_all = zip(*batch)
            if use_all_hr:
                hr_flat = []
                for frame_group in images_HR_all:
                    hr_flat.extend(frame_group)
                images_HR = hr_flat
        else:
            images_HR, images_lr, images_HRy, images_lry, label_strs = zip(*batch)

        images_HR = [self.transform(image) for image in images_HR]
        images_HR = torch.cat([t.unsqueeze(0) for t in images_HR], 0)
        images_HR = rearrange(images_HR, 'b c h w -> b h w c')
        return {'image': images_HR}


def sample(path, ocr_data, n):
    if ocr_data:
        txzm = lmdbDataset_real(root='<OCR_DATA_ROOT>/' + path, voc_type='all', ocr_data=True)
    else:
        txzm = lmdbDataset_real(root='<TEXTZOOM_ROOT>/' + path, voc_type='all')
    items = []
    idx = []
    os.makedirs(f'imgs/{path}/', exist_ok=True)
    for i in range(n):
        id = random.randint(0, txzm.__len__())
        img_HR, img_lr, img_HRy, img_lry, label_str = txzm.__getitem__(id)
        items.append(txzm.__getitem__(id))
        idx.append(id)
        print(img_HR.size, img_lr.size, label_str)
        img_HR.save(f'imgs/{path}/{id}HR.jpg')

    collate_fn = alignCollate_realWTL(imgH=64, imgW=256, down_sample_scale=4, mask=False, train=True)

    resize_res = collate_fn(items)
    # print(resize_res)
    image = resize_res['image']
    LR_image = resize_res['LR_image']
    label = resize_res['label']
    os.makedirs(f'r_imgs/{path}/', exist_ok=True)
    for i in range(n):
        img = np.array(image[i])
        img -= np.min(img)
        img /= np.max(img)
        img = img * 255
        img = Image.fromarray(img.astype('uint8'))
        img.save(f'r_imgs/{path}/{idx[i]}HR.jpg')

        img = np.array(LR_image[i])
        img -= np.min(img)
        img /= np.max(img)
        img = img * 255
        img = Image.fromarray(img.astype('uint8'))
        img.save(f'r_imgs/{path}/{idx[i]}LR.jpg')

        print(image[i].shape, LR_image[i].shape, label[i])


def count(path, ocr_data):
    os.makedirs('statistics/', exist_ok=True)
    if ocr_data:
        txzm = lmdbDataset_real(root='<OCR_DATA_ROOT>/' + path, voc_type='all', ocr_data=True)
    else:
        txzm = lmdbDataset_real(root='<TEXTZOOM_ROOT>/' + path, voc_type='all')
    H = []
    R = []
    for i in tqdm(range(txzm.__len__())):
        img_HR, img_lr, img_HRy, img_lry, label_str = txzm.__getitem__(i)
        width, height = img_HR.size
        ratio = width / height
        if height < 200:
            H.append(height)
        R.append(ratio)

    plt.hist(H, bins=50)
    plt.savefig(f'statistics/{path}_heights.png')
    plt.cla()
    plt.hist(R, bins=50)
    plt.savefig(f'statistics/{path}_ratio.png')
    plt.cla()


def count2(path, ocr_data):
    if ocr_data:
        txzm = lmdbDataset_real(root='<OCR_DATA_ROOT>/' + path, voc_type='all', ocr_data=True)
    else:
        txzm = lmdbDataset_real(root='<TEXTZOOM_ROOT>/' + path, voc_type='all')
    cnt = 0
    for i in tqdm(range(txzm.__len__())):
        img_HR, img_lr, img_HRy, img_lry, label_str = txzm.__getitem__(i)
        width, height = img_HR.size
        ratio = width / height
        if height >= 25 and height <= 200 and ratio >= 1 and ratio <= 4:
            cnt += 1

    print(path, cnt)
    return cnt


def write_cache(env, cache):
    txn = env.begin(write=True)
    for k, v in cache.items():
        txn.put(k, v)
    txn.commit()


def select(path, n=None):
    old_env = lmdb.open(
        '<OCR_DATA_ROOT>/' + path,
        max_readers=1,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False)

    if not old_env:
        print('cannot creat lmdb from %s' % (path))
        sys.exit(0)

    with old_env.begin(write=False) as txn:
        nSamples = int(txn.get(b'num-samples'))
        print(f"{path} nSamples:", nSamples)

    idx = list(range(nSamples))
    random.shuffle(idx)
    if n is None:
        n = nSamples
    txn = old_env.begin(write=False)

    sr_path = 'data/sr_data/' + path
    os.makedirs(sr_path, exist_ok=True)
    env = lmdb.open(sr_path, map_size=1099511627776)
    cache = {}
    valid_num = 0

    for i in tqdm(range(nSamples)):

        label_key = b'label-%09d' % idx[i]
        img_key = b'image-%09d' % idx[i]  # 128*32

        try:
            img = buf2PIL(txn, img_key, 'RGB')
            width, height = img.size
            ratio = width / height
            if height > 200 or height < 25 or ratio < 0.8 or ratio > 4:
                continue

            word = txn.get(label_key)
            if word is None:
                continue
            else:
                word = str(word.decode())
            word = str_filt(word, 'all')
            if len(word) > 100 or len(word) < 1:
                continue

            valid_num += 1
            buff = io.BytesIO()
            img.save(buff, format='PNG')
            image_lr_key = 'image_lr-%09d'.encode() % valid_num
            image_hr_key = 'image_hr-%09d'.encode() % valid_num
            label_key = 'label-%09d'.encode() % valid_num
            cache[image_lr_key] = buff.getvalue()
            cache[image_hr_key] = buff.getvalue()
            cache[label_key] = word.encode()

            if valid_num % 1000 == 0:
                write_cache(env, cache)
                cache = {}
                # print('Written %d / %d' % (valid_num, output_size))
            if valid_num >= n:
                break
        except Exception:
            print(f'error at {path} {idx}')
            continue

    n_samples = valid_num
    cache['num-samples'.encode()] = str(n_samples).encode()
    write_cache(env, cache)
    print('Created dataset %s with %d samples' % (path, n_samples))


def toMask(img_tensor):
    unloader = transforms.ToPILImage()
    toTensor = transforms.ToTensor()
    # mask = unloader(img_tensor).convert('L')
    mask=img_tensor.convert('L')
    thres = np.array(mask).mean()
    mask = mask.point(lambda x: 0 if x > thres else 255)
    # mask = toTensor(mask).unsqueeze(0)
    # mask = mask.repeat(1, 3, 1, 1)
    return mask

if __name__ == '__main__':

    txzm = lmdbDataset_real(root='<TEXTZOOM_TEST_ROOT>' , voc_type='all')
    img_HR, img_lr, img_HRy, img_lry, label_str = txzm.__getitem__(21)
    img_lr = img_lr.resize((128, 32))
    img_HR = img_HR.resize((128, 32))
    mask=toMask(img_lr)
    print(mask)
    img_lr.save('outputs/a.png')
    mask.save('outputs/b.png')
    img_HR.save('outputs/c.png')


    exit(0)

    for dataset in ['IIIT5K', 'COCO_Text', 'ICDAR2013', 'ICDAR2015']:
        select(dataset)
    select('synth90K_shuffle', 50000)
    for dataset in ['SynthAdd',
                    'SynthText800K_shuffle_1_40',
                    'SynthText800K_shuffle_41_80',
                    'SynthText800K_shuffle_81_160',
                    'SynthText800K_shuffle_161_200']:
        select(dataset, 30000)
    exit(0)
    count('<TEXTZOOM_TRAIN_SPLIT_A>', False)
    count('<TEXTZOOM_TRAIN_SPLIT_B>', False)
    for dataset in ['IIIT5K', 'COCO_Text', 'ICDAR2013', 'ICDAR2015',
                    'synth90K_shuffle', 'SynthAdd',
                    'SynthText800K_shuffle_1_40',
                    'SynthText800K_shuffle_41_80',
                    'SynthText800K_shuffle_81_160',
                    'SynthText800K_shuffle_161_200']:
        count(dataset, True)
