import argparse, os, sys, datetime, glob, importlib, csv, re, random
from typing import Optional

import numpy as np
import time
import torch
import torchvision
import pytorch_lightning as pl

from packaging import version
from omegaconf import OmegaConf
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from functools import partial
from PIL import Image

from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info

from ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.util import instantiate_from_config
from text_super_resolution.utils.util import str_filt
from text_super_resolution.utils.labelmaps import get_vocabulary
from text_super_resolution.model import recognizer
from text_super_resolution.utils.metrics import get_string_aster
from text_super_resolution.utils import ssim_psnr
from einops import rearrange
import matplotlib.pyplot as plt


def parse_gpu_ids(gpus):
    if gpus is None:
        return []
    if isinstance(gpus, (list, tuple)):
        return [int(g) for g in gpus]
    if isinstance(gpus, int):
        if gpus < 0:
            return []
        if gpus == 0:
            return [0] if torch.cuda.is_available() else []
        return list(range(gpus))
    if isinstance(gpus, str):
        gpu_str = gpus.strip()
        if gpu_str == "":
            return []
        if "," in gpu_str:
            return [int(x.strip()) for x in gpu_str.split(",") if x.strip() != ""]
        if gpu_str.isdigit():
            value = int(gpu_str)
            if value == 0:
                return [0] if torch.cuda.is_available() else []
            return list(range(value))
    return []


def normalize_trainer_gpus_arg(gpus):
    if isinstance(gpus, int) and gpus == 0 and torch.cuda.is_available():
        return "0,"
    return gpus


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument(
        "-p",
        "--project",
        help="name of new or path to existing project"
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging outputs",
    )
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )
    return parser


def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()

    dataset = worker_info.dataset
    worker_id = worker_info.id

    if isinstance(dataset, Txt2ImgIterableBaseDataset):
        split_size = dataset.num_records // worker_info.num_workers
        # reset num_records to the true number to retain reliable length information
        dataset.sample_ids = dataset.valid_ids[worker_id * split_size:(worker_id + 1) * split_size]
        current_id = np.random.choice(len(np.random.get_state()[1]), 1)
        return np.random.seed(np.random.get_state()[1][current_id] + worker_id)
    else:
        return np.random.seed(np.random.get_state()[1][0] + worker_id)


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None, predict=None,
                 wrap=False, num_workers=None, shuffle_test_loader=False, use_worker_init_fn=False,
                 shuffle_val_dataloader=False, train_align_collate_fn=None, val_align_collate_fn=None,
                 prepare_data_instantiate=False):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn
        self.prepare_data_instantiate = prepare_data_instantiate

        if train_align_collate_fn is not None:
            self.train_align_collate_fn = instantiate_from_config(train_align_collate_fn)
        else:
            self.train_align_collate_fn = None

        if val_align_collate_fn is not None:
            self.val_align_collate_fn = instantiate_from_config(val_align_collate_fn)
        else:
            self.val_align_collate_fn = None

        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
        self.wrap = wrap

    def prepare_data(self):
        # Avoid expensive side effects (e.g., CRNN-based HR selection) twice.
        # Dataset objects are created in `setup()`.
        if self.prepare_data_instantiate:
            for data_cfg in self.dataset_configs.values():
                instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False if is_iterable_dataset else True,
                          worker_init_fn=init_fn, collate_fn=self.train_align_collate_fn)

    def _val_dataloader(self, shuffle=False):
        if isinstance(self.datasets['validation'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          worker_init_fn=init_fn,
                          shuffle=shuffle, collate_fn=self.val_align_collate_fn)

    def _test_dataloader(self, shuffle=False):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        # do not shuffle dataloader for iterable dataset
        shuffle = shuffle and (not is_iterable_dataset)

        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn, shuffle=shuffle)

    def _predict_dataloader(self, shuffle=False):
        if isinstance(self.datasets['predict'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["predict"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn)


class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_keyboard_interrupt(self, trainer, pl_module):
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)

    def on_pretrain_routine_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            if "callbacks" in self.lightning_config:
                if 'metrics_over_trainsteps_checkpoint' in self.lightning_config['callbacks']:
                    os.makedirs(os.path.join(self.ckptdir, 'trainstep_checkpoints'), exist_ok=True)
            print("Project config")
            print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            print("Lightning config")
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))

        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass


class ImageLogger(Callback):
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.logger_log_images = {
            pl.loggers.TestTubeLogger: self._testtube,
        }
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

    @rank_zero_only
    def _testtube(self, pl_module, images, batch_idx, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w

            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(
                tag, grid,
                global_step=pl_module.global_step)

    @rank_zero_only
    def log_local(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k,
                global_step,
                current_epoch,
                batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, pl_module.global_step, split)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)) and (
                check_idx > 0 or self.log_first_step):
            try:
                self.log_steps.pop(0)
            except IndexError as e:
                print(e)
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
            self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled and pl_module.global_step > 0:
            self.log_img(pl_module, batch, batch_idx, split="val")
        if hasattr(pl_module, 'calibrate_grad_norm'):
            if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)


class AsterInfo(object):
    def __init__(self, voc_type):
        super(AsterInfo, self).__init__()
        self.voc_type = voc_type
        assert voc_type in ['digit', 'lower', 'upper', 'all', 'chinese']
        self.EOS = 'EOS'
        self.max_len = 100
        self.PADDING = 'PADDING'
        self.UNKNOWN = 'UNKNOWN'
        self.voc = get_vocabulary(voc_type, EOS=self.EOS, PADDING=self.PADDING, UNKNOWN=self.UNKNOWN)
        self.char2id = dict(zip(self.voc, range(len(self.voc))))
        self.id2char = dict(zip(range(len(self.voc)), self.voc))
        self.rec_num_classes = len(self.voc)


class RecognizeCallback(Callback):
    def __init__(self, gpus, rec_interval, parseq_checkpoint_path=None, parseq_repo_path=None,
                 parseq_use_format_constraint=True, prefer_parseq=True,
                 eval_ddim_eta=1.0, eval_seed_base=23):
        self.save_dir = os.getcwd() + '/logs/' + datetime.datetime.now().strftime('%Y-%m-%dT%H-%M_train')
        print(self.save_dir)
        os.makedirs(self.save_dir, exist_ok=True)
        self.epoch_cnt = 0
        self.rec_interval = rec_interval
        self.voc_type = 'all'
        gpus = parse_gpu_ids(gpus)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.aster_info = AsterInfo(self.voc_type)
        aster_real = self.Aster_init(gpus)
        self.aster = [{
            'model': aster_real,
            'data_in_fn': self.parse_aster_data,
            'string_process': get_string_aster
        }]
        default_parseq_repo = os.path.join(os.getcwd(), 'parseq')
        default_parseq_ckpt = os.path.join(default_parseq_repo, 'weights', 'parseq.pt')
        self.parseq_repo_path = parseq_repo_path or default_parseq_repo
        self.parseq_checkpoint_path = parseq_checkpoint_path or default_parseq_ckpt
        if not os.path.isabs(self.parseq_repo_path):
            self.parseq_repo_path = os.path.abspath(os.path.join(os.getcwd(), self.parseq_repo_path))
        if not os.path.isabs(self.parseq_checkpoint_path):
            self.parseq_checkpoint_path = os.path.abspath(os.path.join(os.getcwd(), self.parseq_checkpoint_path))
        self.prefer_parseq = prefer_parseq
        self.use_format_constraint = parseq_use_format_constraint
        self.eval_ddim_eta = float(eval_ddim_eta)
        self.eval_seed_base = int(eval_seed_base)
        self.parseq = self.PARSeq_init()
        self.cal_psnr = ssim_psnr.calculate_psnr
        self.cal_ssim = ssim_psnr.SSIM()

        self.n_correct = 0
        self.n_correct_lr = 0
        self.n_correct_hr = 0
        self.sum_images = 0
        self.sum_lr_images = 0
        self.sum_hr_images = 0
        self.metric_dict = {}
        self.image_counter = 0
        self.false_cnt = 0
        self.last_accuracy = 0.0
        self._recognition_skip_logged = False
        self.metric_init()

    def _supports_recognition(self, pl_module, batch=None):
        if not hasattr(pl_module, "recognize_sample"):
            return False
        if batch is not None:
            if not isinstance(batch, dict):
                return False
            required = ("image", "LR_image", "label")
            for key in required:
                if key not in batch:
                    return False
        return True

    def metric_init(self):
        self.n_correct = 0
        self.n_correct_lr = 0
        self.n_correct_hr = 0
        self.sum_images = 0
        self.sum_lr_images = 0
        self.sum_hr_images = 0
        self.metric_dict = {
            'psnr_lr': [],
            'ssim_lr': [],
            'cnt_psnr_lr': [],
            'cnt_ssim_lr': [],
            'psnr': [],
            'ssim': [],
            'cnt_psnr': [],
            'cnt_ssim': [],
            'accuracy': 0.0,
            'psnr_avg': 0.0,
            'ssim_avg': 0.0,
            'edis_LR': [],
            'edis_SR': [],
            'edis_HR': [],
            'LPIPS_VGG_LR': [],
            'LPIPS_VGG_SR': []
        }
        self.image_counter = 0
        self.false_cnt = 0

    def PARSeq_init(self):
        from ldm.modules.encoders.tp_generator import PARSeqTPG

        if not self.prefer_parseq:
            return None

        checkpoint_path = self.parseq_checkpoint_path
        parseq_repo_path = self.parseq_repo_path

        if not os.path.isfile(checkpoint_path):
            print(f"[RecognizeCallback] PARSeq checkpoint not found, fallback to ASTER: {checkpoint_path}")
            return None
        if not os.path.isdir(parseq_repo_path):
            print(f"[RecognizeCallback] PARSeq repo not found, fallback to ASTER: {parseq_repo_path}")
            return None

        try:
            model = PARSeqTPG(
                checkpoint_path=checkpoint_path,
                parseq_repo_path=parseq_repo_path,
                device='cuda' if self.device.type == 'cuda' else 'cpu',
                use_format_constraint=self.use_format_constraint,
            )
            model = model.to(self.device)
            model.eval()
            return model
        except Exception as exc:
            print(f"[RecognizeCallback] PARSeq init failed, fallback to ASTER: {exc}")
            return None

    def Aster_init(self, gpus):
        aster = recognizer.RecognizerBuilder(arch='ResNet_ASTER', rec_num_classes=self.aster_info.rec_num_classes,
                                             sDim=512, attDim=512, max_len_labels=self.aster_info.max_len,
                                             eos=self.aster_info.char2id[self.aster_info.EOS], STN_ON=True)
        aster_ckpt_path = 'aster.pth.tar'
        aster.load_state_dict(torch.load(aster_ckpt_path)['state_dict'])
        print('load pred_trained aster model from %s' % aster_ckpt_path)
        aster = aster.to(self.device)
        aster = torch.nn.DataParallel(aster, device_ids=gpus)
        aster.eval()
        for p in aster.parameters():
            p.requires_grad = False
        return aster

    def parse_aster_data(self, imgs_input):
        input_dict = {}
        images_input = imgs_input.to(self.device)
        input_dict['images'] = images_input * 2 - 1
        batch_size = images_input.shape[0]
        input_dict['rec_targets'] = torch.IntTensor(batch_size, self.aster_info.max_len).fill_(1)
        input_dict['rec_lengths'] = [self.aster_info.max_len] * batch_size
        return input_dict

    @staticmethod
    def _normalize_text_upper(text):
        if text is None:
            return ""
        return re.sub(r'[^0-9A-Z]+', '', str(text).upper())

    def _to_bchw_or_fused_lr(self, images):
        if images.dim() == 4:
            return rearrange(images, 'b h w c -> b c h w')
        if images.dim() == 5:
            if images.shape[-1] <= 4:
                images = rearrange(images, 'b t h w c -> b t c h w')
            elif images.shape[2] <= 4:
                pass
            else:
                raise ValueError(f"Unsupported LR image shape: {tuple(images.shape)}")
            return images.mean(dim=1)
        raise ValueError(f"Unsupported image rank: {images.dim()}")

    @staticmethod
    def _to_bchw_preserve_frames(images):
        if images.dim() == 4:
            return rearrange(images, 'b h w c -> b c h w')
        if images.dim() == 5:
            if images.shape[-1] <= 4:
                return rearrange(images, 'b t h w c -> b t c h w')
            if images.shape[2] <= 4:
                return images
            raise ValueError(f"Unsupported image shape: {tuple(images.shape)}")
        raise ValueError(f"Unsupported image rank: {images.dim()}")

    @staticmethod
    def _fuse_frame_batch(images):
        frames = RecognizeCallback._to_bchw_preserve_frames(images)
        if frames.dim() == 4:
            return frames
        return rearrange(frames, 'b t c h w -> b (t c) h w')

    @staticmethod
    def _flatten_frame_batch(images):
        frames = RecognizeCallback._to_bchw_preserve_frames(images)
        if frames.dim() == 4:
            return frames, 1
        return rearrange(frames, 'b t c h w -> (b t) c h w'), frames.shape[1]

    def _predict_aster_texts(self, images_bchw, aster, aster_info):
        aster_dict = aster[0]["data_in_fn"](images_bchw)
        aster_output = aster[0]["model"](aster_dict)
        predict_result, _ = aster[0]["string_process"](
            aster_output['output']['pred_rec'],
            aster_dict['rec_targets'],
            dataset=aster_info
        )
        return predict_result

    @staticmethod
    def _parse_parseq_data(imgs_input):
        if imgs_input.ndim == 3:
            imgs_input = imgs_input.unsqueeze(0)
        imgs = torch.nn.functional.interpolate(imgs_input, size=(32, 128), mode='bicubic', align_corners=False)
        imgs = (imgs - 0.5) / 0.5
        return imgs

    def _predict_parseq_texts(self, images_bchw):
        if self.parseq is None:
            return None
        parseq_core = getattr(self.parseq, "parseq", None)
        tokenizer = getattr(parseq_core, "tokenizer", None) if parseq_core is not None else None
        if parseq_core is None or tokenizer is None:
            return None

        with torch.no_grad():
            logits = parseq_core(self._parse_parseq_data(images_bchw))
            if isinstance(logits, dict):
                logits = logits.get("logits", logits.get("output", logits))
            elif isinstance(logits, (tuple, list)):
                logits = logits[0]
            probs = logits.softmax(dim=-1)
            preds, _ = tokenizer.decode(probs)

        return [re.sub(r'[^A-Z0-9]', '', str(p).upper()) for p in preds]

    def _decode_with_parseq_tokenizer(self, output):
        if not (torch.is_tensor(output) and output.dim() == 3):
            return None
        if self.parseq is None:
            return None

        parseq_core = getattr(self.parseq, "parseq", None)
        tokenizer = getattr(parseq_core, "tokenizer", None)
        if tokenizer is None:
            return None

        probs = output
        try:
            if probs.min().item() < 0.0 or probs.max().item() > 1.0 + 1e-5:
                probs = probs.softmax(dim=-1)
            preds, _ = tokenizer.decode(probs)
        except Exception:
            return None

        cleaned = []
        for pred in preds:
            s = str(pred).upper()
            s = re.sub(r'[^A-Z0-9]', '', s)
            cleaned.append(s)
        return cleaned

    def get_string_simple(self, output):
        parseq_preds = self._decode_with_parseq_tokenizer(output)
        if parseq_preds is not None:
            return parseq_preds

        alphabet_37 = "-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        alphabet_95 = "-0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"

        if torch.is_tensor(output) and output.dim() == 3:
            vocab_size = int(output.shape[-1])
            if vocab_size == len(alphabet_37):
                alphabet = alphabet_37
            elif vocab_size == len(alphabet_95):
                alphabet = alphabet_95
            else:
                alphabet = alphabet_95 if vocab_size > len(alphabet_37) else alphabet_37
        else:
            alphabet = alphabet_37

        if self.use_format_constraint and torch.is_tensor(output) and output.dim() == 3:
            probs = output.softmax(dim=-1)
            mask = torch.ones_like(probs)
            idx_eos = 0
            idx_digit = slice(1, 11)
            seq_len = probs.size(1)

            if seq_len > 0:
                mask[:, 0:2, idx_eos] = 0
                mask[:, 0:2, idx_digit] = 0
            if seq_len > 2:
                mask[:, 2:7, idx_eos] = 0
            if seq_len > 7:
                mask[:, 7:, 1:] = 0

            probs = probs * mask
            indices = probs.argmax(dim=-1)
        else:
            if torch.is_tensor(output) and output.dim() == 3:
                indices = output.argmax(dim=-1)
            else:
                indices = output

        res = []
        for batch_idx in indices:
            s = ""
            for i in batch_idx:
                idx = i.item()
                if idx >= len(alphabet) or alphabet[idx] == '-':
                    break
                s += alphabet[idx]
            res.append(s.upper())
        return res

    def recognize(self, pl_module, batch, batch_idx, split="train"):
        print('******************************recognize*********************************')
        is_train = pl_module.training
        if is_train:
            pl_module.eval()

        seed = self.eval_seed_base + int(batch_idx)
        py_state = random.getstate()
        np_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

        try:
            random.seed(seed)
            np.random.seed(seed % (2 ** 32 - 1))
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            with torch.no_grad():
                images = pl_module.recognize_sample(
                    batch, N=114514, split=split, inpaint=False, ddim_eta=self.eval_ddim_eta
                )
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)
            torch.random.set_rng_state(torch_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)

        images_sr = images['samples']
        # for k in images:
        #     N = min(images[k].shape[0], self.max_images)
        #     images[k] = images[k][:N]
        #     if isinstance(images[k], torch.Tensor):
        #         images[k] = images[k].detach().cpu()
        #         if self.clamp:
        #             images[k] = torch.clamp(images[k], -1., 1.)

        if is_train:
            pl_module.train()

        return images_sr

    def eval(self, batch, images_sr, aster, aster_info, save_dir):

        #############################################
        # Print the computational cost and param size
        # self.cal_all_models(model_list, aster[1])
        #############################################

        images_hr_raw = batch['image']
        images_hr_all_raw = batch.get('HR_all_images', batch['image'])
        images_lr_raw = batch['LR_image']
        label_strs = batch['label']
        indexes = batch['id']

        images_lr = self._to_bchw_or_fused_lr(images_lr_raw)
        images_hr = self._to_bchw_or_fused_lr(images_hr_raw)

        images_lr = images_lr.to(self.device)
        images_hr = images_hr.to(self.device)
        images_sr = images_sr.to(self.device)

        val_batch_size = images_lr.shape[0]
        images_lr_flat, lr_frame_count = self._flatten_frame_batch(images_lr_raw)
        images_hr_flat, hr_frame_count = self._flatten_frame_batch(images_hr_all_raw)
        images_lr_flat = images_lr_flat.to(self.device)
        images_hr_flat = images_hr_flat.to(self.device)

        if self.parseq is not None:
            predict_result_sr = self._predict_parseq_texts(images_sr[:, :3, :, :])
            predict_result_lr = self._predict_parseq_texts(images_lr_flat[:, :3, :, :])
            predict_result_hr = self._predict_parseq_texts(images_hr_flat[:, :3, :, :])
            if predict_result_sr is None or predict_result_lr is None or predict_result_hr is None:
                with torch.no_grad():
                    output_sr = self.parseq(images_sr[:, :3, :, :])
                    output_lr = self.parseq(images_lr_flat[:, :3, :, :])
                    output_hr = self.parseq(images_hr_flat[:, :3, :, :])

                predict_result_sr = self.get_string_simple(output_sr)
                predict_result_lr = self.get_string_simple(output_lr)
                predict_result_hr = self.get_string_simple(output_hr)
        else:
            predict_result_sr = self._predict_aster_texts(images_sr[:, :3, :, :], aster, aster_info)
            predict_result_lr = self._predict_aster_texts(images_lr_flat[:, :3, :, :], aster, aster_info)
            predict_result_hr = self._predict_aster_texts(images_hr_flat[:, :3, :, :], aster, aster_info)

        img_lr = torch.nn.functional.interpolate(images_lr, images_sr.shape[-2:], mode="bicubic")

        self.metric_dict['psnr'].append(self.cal_psnr(images_sr[:, :3], images_hr[:, :3]))
        self.metric_dict['ssim'].append(self.cal_ssim(images_sr[:, :3], images_hr[:, :3]))

        self.metric_dict['psnr_lr'].append(self.cal_psnr(img_lr[:, :3], images_hr[:, :3]))
        self.metric_dict['ssim_lr'].append(self.cal_ssim(img_lr[:, :3], images_hr[:, :3]))

        for batch_i in range(images_lr.shape[0]):
            label = label_strs[batch_i]
            label_norm = self._normalize_text_upper(label)
            # print(predict_result_sr[batch_i],predict_result_lr[batch_i],predict_result_hr[batch_i],label)
            self.image_counter += 1

            if self._normalize_text_upper(predict_result_sr[batch_i]) == label_norm:
                self.n_correct += 1
            else:
                self.false_cnt += 1
                # plt.figure()
                # plt.title(label)
                # plt.subplot(1, 3, 1)
                # plt.imshow(images_lr[batch_i, :3, :, :].cpu().numpy().transpose(1, 2, 0))
                # plt.title(predict_result_lr[batch_i])
                # plt.axis('off')
                #
                # plt.subplot(1, 3, 2)
                # plt.imshow(images_hr[batch_i, :3, :, :].cpu().numpy().transpose(1, 2, 0))
                # plt.title(predict_result_hr[batch_i])
                # plt.axis('off')
                #
                # plt.subplot(1, 3, 3)
                # plt.imshow(images_sr[batch_i, :3, :, :].cpu().numpy().transpose(1, 2, 0))
                # plt.title(predict_result_sr[batch_i])
                # plt.axis('off')
                # plt.savefig(os.path.join(save_dir,
                #                          f'{torch.cuda.current_device()}_{self.image_counter}_{self.false_cnt}.jpg'))

            lr_start = batch_i * lr_frame_count
            lr_end = lr_start + lr_frame_count
            for lr_pred in predict_result_lr[lr_start:lr_end]:
                if self._normalize_text_upper(lr_pred) == label_norm:
                    self.n_correct_lr += 1

            hr_start = batch_i * hr_frame_count
            hr_end = hr_start + hr_frame_count
            for hr_pred in predict_result_hr[hr_start:hr_end]:
                if self._normalize_text_upper(hr_pred) == label_norm:
                    self.n_correct_hr += 1

        self.sum_images += val_batch_size
        self.sum_lr_images += val_batch_size * lr_frame_count
        self.sum_hr_images += val_batch_size * hr_frame_count
        torch.cuda.empty_cache()
        # print(f'sr correct:{self.n_correct}/{self.sum_images}')
        # print(f'lr correct:{self.n_correct_lr}/{self.sum_images}')
        # print(f'hr correct:{self.n_correct_hr}/{self.sum_images}')

    def show_results(self, pl_module):
        if self.sum_images <= 0:
            pl_module.log('accuracy', float(self.last_accuracy),
                          prog_bar=False, logger=True, on_step=False, on_epoch=True)
            return

        psnr_avg = sum(self.metric_dict['psnr']) / (len(self.metric_dict['psnr']) + 1e-10)
        ssim_avg = sum(self.metric_dict['ssim']) / (len(self.metric_dict['ssim']) + 1e-10)

        psnr_avg_lr = sum(self.metric_dict['psnr_lr']) / (len(self.metric_dict['psnr_lr']) + 1e-10)
        ssim_avg_lr = sum(self.metric_dict['ssim_lr']) / (len(self.metric_dict['ssim_lr']) + 1e-10)

        print('[{}]\t'
              'PSNR {:.2f} | SSIM {:.4f}\t'
              .format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                      float(psnr_avg), float(ssim_avg)))

        print('[{}]\t'
              'PSNR_LR {:.2f} | SSIM_LR {:.4f}\t'
              .format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                      float(psnr_avg_lr), float(ssim_avg_lr)))

        # self.tripple_display(images_lr, images_sr, images_hr, pred_str_lr, pred_str_sr, label_strs, index)

        accuracy = round(self.n_correct / self.sum_images, 4)
        lr_den = self.sum_lr_images if self.sum_lr_images > 0 else self.sum_images
        hr_den = self.sum_hr_images if self.sum_hr_images > 0 else self.sum_images
        accuracy_lr = round(self.n_correct_lr / lr_den, 4)
        accuracy_hr = round(self.n_correct_hr / hr_den, 4)
        psnr_avg = round(psnr_avg.item(), 6)
        ssim_avg = round(ssim_avg.item(), 6)

        print('sr_accuracy: %.2f%%' % (accuracy * 100))

        # print('sr_NED: %.4f' % (edis_SR))
        print('lr_accuracy: %.2f%%' % (accuracy_lr * 100))
        # print('lr_NED: %.4f' % (edis_LR))
        print('hr_accuracy: %.2f%%' % (accuracy_hr * 100))
        # print('hr_NED: %.4f' % (edis_HR))
        log_dict = {'accuracy': accuracy, 'psnr_avg': psnr_avg, 'ssim_avg': ssim_avg}
        self.last_accuracy = accuracy

        pl_module.log_dict(log_dict, prog_bar=False, logger=True, on_step=False, on_epoch=True)

        print("sum_images:", self.sum_images)
        print("sum_lr_images:", self.sum_lr_images)
        print("sum_hr_images:", self.sum_hr_images)
        self.metric_init()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if getattr(trainer, "sanity_checking", False):
            return
        if not self._supports_recognition(pl_module, batch):
            if not self._recognition_skip_logged:
                print("[RecognizeCallback] Skip recognizer eval for this model/batch (no recognize path).")
                self._recognition_skip_logged = True
            return
        if (trainer.current_epoch + 1) % self.rec_interval == 0:
            images_sr = self.recognize(pl_module, batch, batch_idx, split="val")
            svd = os.path.join(self.save_dir, str(trainer.current_epoch))
            os.makedirs(svd, exist_ok=True)
            self.eval(batch, images_sr, self.aster, self.aster_info, svd)

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if getattr(trainer, "sanity_checking", False):
            return
        if not self._supports_recognition(pl_module):
            return
        if (trainer.current_epoch + 1) % self.rec_interval == 0:
            self.show_results(pl_module)
        else:
            # Keep `accuracy` available for checkpoint monitor every epoch.
            pl_module.log('accuracy', float(self.last_accuracy),
                          prog_bar=False, logger=True, on_step=False, on_epoch=True)

    def on_train_epoch_end(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", unused: Optional = None
    ):
        self.epoch_cnt += 1


class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(trainer.root_gpu)
        torch.cuda.synchronize(trainer.root_gpu)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        torch.cuda.synchronize(trainer.root_gpu)
        max_memory = torch.cuda.max_memory_allocated(trainer.root_gpu) / 2 ** 20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.training_type_plugin.reduce(max_memory)
            epoch_time = trainer.training_type_plugin.reduce(epoch_time)

            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            pass


if __name__ == "__main__":
    # custom parser to specify config files, train, test and debug mode,
    # postfix, resume.
    # `--key value` arguments are interpreted as arguments to the trainer.
    # `nested.key=value` arguments are interpreted as config parameters.
    # configs are merged from left-to-right followed by command line parameters.

    # model:
    #   base_learning_rate: float
    #   target: path to lightning module
    #   params:
    #       key: value
    # data:
    #   target: main.DataModuleFromConfig
    #   params:
    #      batch_size: int
    #      wrap: bool
    #      train:
    #          target: path to train dataset
    #          params:
    #              key: value
    #      validation:
    #          target: path to validation dataset
    #          params:
    #              key: value
    #      test:
    #          target: path to test dataset
    #          params:
    #              key: value
    # lightning: (optional, has sane defaults and can be specified on cmdline)
    #   trainer:
    #       additional arguments to trainer
    #   logger:
    #       logger to instantiate
    #   modelcheckpoint:
    #       modelcheckpoint to instantiate
    #   callbacks:
    #       callback1:
    #           target: importpath
    #           params:
    #               key: value

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    # (in particular `main.DataModuleFromConfig`)
    sys.path.append(os.getcwd())

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)

    opt, unknown = parser.parse_known_args()
    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            # idx = len(paths)-paths[::-1].index("logs")+1
            # logdir = "/".join(paths[:idx])
            logdir = "/".join(paths[:-2])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:
        if opt.name:
            name = "_" + opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_" + cfg_name
        else:
            name = ""
        nowname = now + name + opt.postfix
        logdir = os.path.join(opt.logdir, nowname)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed)

    try:
        # init and save configs
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)
        lightning_config = config.pop("lightning", OmegaConf.create())
        # merge trainer cli with config
        trainer_config = lightning_config.get("trainer", OmegaConf.create())
        # default to ddp on Unix-like systems; Windows runs more reliably without forcing ddp here
        if os.name != "nt":
            trainer_config["accelerator"] = "ddp"
        for k in nondefault_trainer_args(opt):
            trainer_config[k] = getattr(opt, k)
        if "gpus" in trainer_config:
            normalized_gpus = normalize_trainer_gpus_arg(trainer_config["gpus"])
            if normalized_gpus != trainer_config["gpus"]:
                print(f"Normalizing trainer gpus from {trainer_config['gpus']} to {normalized_gpus}")
                trainer_config["gpus"] = normalized_gpus
        if not "gpus" in trainer_config:
            trainer_config.pop("accelerator", None)
            cpu = True
        else:
            gpuinfo = trainer_config["gpus"]
            print(f"Running on GPUs {gpuinfo}")
            cpu = False
        trainer_opt = argparse.Namespace(**trainer_config)
        lightning_config.trainer = trainer_config

        # model
        model = instantiate_from_config(config.model)

        # trainer and callbacks
        trainer_kwargs = dict()

        # default logger configs
        default_logger_cfgs = {
            "wandb": {
                "target": "pytorch_lightning.loggers.WandbLogger",
                "params": {
                    "name": nowname,
                    "save_dir": logdir,
                    "offline": opt.debug,
                    "id": nowname,
                }
            },
            "testtube": {
                "target": "pytorch_lightning.loggers.TestTubeLogger",
                "params": {
                    "name": "testtube",
                    "save_dir": logdir,
                }
            },
        }
        default_logger_cfg = default_logger_cfgs["testtube"]
        if "logger" in lightning_config:
            logger_cfg = lightning_config.logger
        else:
            logger_cfg = OmegaConf.create()
        logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
        trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

        # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
        # specify which metric is used to determine best models
        default_modelckpt_cfg = {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch:06}-{accuracy:.2f}",
                # "filename": "{epoch:06}",
                "verbose": True,
                "save_last": True,
                "every_n_epochs": 30,
                "save_top_k": -1,
                "mode": "max"
            }
        }
        if hasattr(model, "monitor"):
            print(f"Monitoring {model.monitor} as checkpoint metric.")
            default_modelckpt_cfg["params"]["monitor"] = model.monitor

        if "modelcheckpoint" in lightning_config:
            modelckpt_cfg = lightning_config.modelcheckpoint
        else:
            modelckpt_cfg = OmegaConf.create()
        modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
        print(f"Merged modelckpt-cfg: \n{modelckpt_cfg}")
        if version.parse(pl.__version__) < version.parse('1.4.0'):
            trainer_kwargs["checkpoint_callback"] = instantiate_from_config(modelckpt_cfg)

        # add callback which sets up log directory
        parseq_settings = config.get("parseq_settings", OmegaConf.create())

        default_callbacks_cfg = {
            "setup_callback": {
                "target": "main.SetupCallback",
                "params": {
                    "resume": opt.resume,
                    "now": now,
                    "logdir": logdir,
                    "ckptdir": ckptdir,
                    "cfgdir": cfgdir,
                    "config": config,
                    "lightning_config": lightning_config,
                }
            },
            "image_logger": {
                "target": "main.ImageLogger",
                "params": {
                    "increase_log_steps": False,
                    "batch_frequency": 2000,
                    "max_images": 4,
                    "clamp": True
                }
            },
            "learning_rate_logger": {
                "target": "main.LearningRateMonitor",
                "params": {
                    "logging_interval": "step",
                    # "log_momentum": True
                }
            },
            "cuda_callback": {
                "target": "main.CUDACallback"
            },
            "recognize_callback": {
                "target": "main.RecognizeCallback",
                "params": {
                    "gpus": lightning_config.trainer.gpus,
                    "rec_interval": parseq_settings.get("rec_interval", 30),
                    "parseq_checkpoint_path": parseq_settings.get(
                        "eval_parseq_checkpoint_path",
                        config.model.params.get("parseq_checkpoint_path", os.path.join("parseq", "weights", "parseq.pt"))
                    ),
                    "parseq_repo_path": parseq_settings.get(
                        "eval_parseq_repo_path",
                        config.model.params.get("parseq_repo_path", "parseq")
                    ),
                    "parseq_use_format_constraint": parseq_settings.get(
                        "eval_parseq_use_format_constraint",
                        config.model.params.get(
                            "parseq_use_format_constraint",
                            config.model.params.get(
                                "cond_stage_config",
                                OmegaConf.create()
                            ).get("params", OmegaConf.create()).get("use_format_constraint", True)
                        )
                    ),
                    "prefer_parseq": parseq_settings.get(
                        "prefer_parseq_for_eval",
                        config.model.params.get("prefer_parseq_for_eval", True)
                    ),
                    "eval_ddim_eta": parseq_settings.get("eval_ddim_eta", 1.0),
                    "eval_seed_base": parseq_settings.get("eval_seed_base", opt.seed),
                }
            },
        }
        if version.parse(pl.__version__) >= version.parse('1.4.0'):
            default_callbacks_cfg.update({'checkpoint_callback': modelckpt_cfg})

        if "callbacks" in lightning_config:
            callbacks_cfg = lightning_config.callbacks
        else:
            callbacks_cfg = OmegaConf.create()

        if 'metrics_over_trainsteps_checkpoint' in callbacks_cfg:
            print(
                'Caution: Saving checkpoints every n train steps without deleting. This might require some free space.')
            default_metrics_over_trainsteps_ckpt_dict = {
                'metrics_over_trainsteps_checkpoint':
                    {"target": 'pytorch_lightning.callbacks.ModelCheckpoint',
                     'params': {
                         "dirpath": os.path.join(ckptdir, 'trainstep_checkpoints'),
                         "filename": "{epoch:06}-{step:09}",
                         "verbose": True,
                         'save_top_k': -1,
                         'every_n_train_steps': 10000,
                         'save_weights_only': True
                     }
                     }
            }
            default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)

        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
        if 'ignore_keys_callback' in callbacks_cfg and hasattr(trainer_opt, 'resume_from_checkpoint'):
            callbacks_cfg.ignore_keys_callback.params['ckpt_path'] = trainer_opt.resume_from_checkpoint
        elif 'ignore_keys_callback' in callbacks_cfg:
            del callbacks_cfg['ignore_keys_callback']

        trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]

        print('**************trainer***************')
        print(trainer_opt)
        print(trainer_kwargs)
        trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
        trainer.logdir = logdir  ###
        trainer.check_val_every_n_epoch = 1

        # data
        data = instantiate_from_config(config.data)
        # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
        # calling these ourselves should not be necessary but it is.
        # lightning still takes care of proper multiprocessing though
        data.prepare_data()
        data.setup()
        print("#### Data #####")
        for k in data.datasets:
            print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")

        # configure learning rate
        bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
        if not cpu:
            ngpu = max(1, len(parse_gpu_ids(lightning_config.trainer.gpus)))
        else:
            ngpu = 1
        if 'accumulate_grad_batches' in lightning_config.trainer:
            accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
        else:
            accumulate_grad_batches = 1
        print(f"accumulate_grad_batches = {accumulate_grad_batches}")
        lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
        if opt.scale_lr:
            model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
            print(
                "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                    model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))
        else:
            model.learning_rate = base_lr
            print("++++ NOT USING LR SCALING ++++")
            print(f"Setting learning rate to {model.learning_rate:.2e}")


        # allow checkpointing via USR1
        def melk(*args, **kwargs):
            # run all checkpoint hooks
            if trainer.global_rank == 0:
                print("Summoning checkpoint.")
                ckpt_path = os.path.join(ckptdir, "last.ckpt")
                trainer.save_checkpoint(ckpt_path)


        def divein(*args, **kwargs):
            if trainer.global_rank == 0:
                import pudb;
                pudb.set_trace()


        import signal

        if hasattr(signal, "SIGUSR1") and hasattr(signal, "SIGUSR2"):
            signal.signal(signal.SIGUSR1, melk)
            signal.signal(signal.SIGUSR2, divein)

        # run
        if opt.train:
            try:
                trainer.fit(model, data)
            except Exception:
                melk()
                raise
        if not opt.no_test and not trainer.interrupted:
            trainer.test(model, data)
    except Exception:
        trainer_obj = locals().get("trainer", None)
        if opt.debug and trainer_obj is not None and trainer_obj.global_rank == 0:
            try:
                import pudb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise
    finally:
        trainer_obj = locals().get("trainer", None)
        # move newly created debug project to debug_runs
        if opt.debug and not opt.resume and trainer_obj is not None and trainer_obj.global_rank == 0:
            dst, name = os.path.split(logdir)
            dst = os.path.join(dst, "debug_runs", name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            os.rename(logdir, dst)
        if trainer_obj is not None and trainer_obj.global_rank == 0:
            print(trainer_obj.profiler.summary())
