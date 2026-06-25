import datetime
import math
import os
import torch
import torch.nn.functional as F
from torch import nn
from collections import OrderedDict
import sys
from torch.nn import init
import numpy as np
from IPython import embed

from text_super_resolution.model.transformer_v2 import InfoTransformer
from text_super_resolution.model.transformer_v2 import PositionalEncoding

import ptflops
from text_super_resolution.model import crnn
from text_super_resolution.utils.labelmaps import get_vocabulary


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


class TP_generator(nn.Module):

    def __init__(
            self,
            imgH,
            recognizer_path=None,
    ):
        super(TP_generator, self).__init__()
        self.imgH = imgH
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.crnn_model, _ = self.CRNN_init(recognizer_path, imgH)
        # ':' is invalid in Windows filenames
        self.timestamp=datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')

    def CRNN_init(self, recognizer_path=None, imgH=32):
        model = crnn.CRNN(imgH, 1, 37, 256)
        model = model.to(self.device)

        macs, params = ptflops.get_model_complexity_info(model, (1, 32, 100), as_strings=True,
                                                         print_per_layer_stat=False, verbose=True)
        print("---------------- TP Module -----------------")
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
        print("--------------------------------------------")

        print("recognizer_path:", recognizer_path)

        aster_info = AsterInfo('all')
        if recognizer_path is not None:
            model_path = recognizer_path
            print('loading pretrained crnn model from %s' % model_path)
            stat_dict = torch.load(model_path)
            # print("stat_dict:", stat_dict.keys())
            # if recognizer_path is None:
            #     model.load_state_dict(stat_dict)
            # else:
                # print("stat_dict:", stat_dict)
                # print("stat_dict:", type(stat_dict) == OrderedDict)
            if type(stat_dict) == OrderedDict:
                print("The dict:")
                model.load_state_dict(stat_dict)
            else:
                print("The model:")
                model = stat_dict
        # model #.eval()
        # model.eval()
        return model, aster_info

    def save_state_dict(self, path, epoch):
        os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, self.timestamp + f'-e{epoch}.pth')
        torch.save(self.crnn_model.state_dict(), save_path)

    def parse_crnn_data(self, imgs_input_, ratio_keep=False):

        # in_width = self.config.TRAIN.width if self.config.TRAIN.width != 128 else 100
        in_width = 100

        if ratio_keep:
            real_height, real_width = imgs_input_.shape[2:]
            ratio = real_width / float(real_height)

            if ratio > 3:
                in_width = int(ratio * 32)
        imgs_input = torch.nn.functional.interpolate(imgs_input_, (self.imgH, in_width), mode='bicubic')

        # print("imgs_input:", imgs_input.shape)

        R = imgs_input[:, 0:1, :, :]
        G = imgs_input[:, 1:2, :, :]
        B = imgs_input[:, 2:3, :, :]
        tensor = 0.299 * R + 0.587 * G + 0.114 * B
        return tensor

    def forward(self, image):

        # H, W = self.output_size
        # x = tp_input #b,h,1,l
        image = self.parse_crnn_data(image)
        # print(image.shape)
        label_vecs_logits = self.crnn_model(image)
        label_vecs = torch.nn.functional.softmax(label_vecs_logits, -1)  # l,b,h

        # print(label_vecs.shape,len(label_strs),label_strs)
        # print(get_string_crnn(label_vecs,use_chinese=False))
        # print(get_string_crnn(label_vecs_hr,use_chinese=False))
        # exit(0)

        label_vecs_final = label_vecs.permute(1, 0, 2)  # b,l,h
        return label_vecs_final


class TPInterpreter(nn.Module):
    def __init__(
            self,
            t_emb,
            out_text_channels,
            output_size=(16, 64),
            feature_in=64,
            # d_model=512,
            t_encoder_num=1,
            t_decoder_num=2,
    ):
        super(TPInterpreter, self).__init__()

        d_model = out_text_channels  # * output_size[0]

        self.fc_in = nn.Linear(t_emb, d_model)
        self.fc_in2 = nn.Linear(4096, d_model)
        self.fc_feature_in = nn.Linear(feature_in, d_model)

        self.activation = nn.PReLU()

        self.transformer = InfoTransformer(d_model=d_model,
                                           dropout=0.1,
                                           nhead=4,
                                           dim_feedforward=d_model,
                                           num_encoder_layers=t_encoder_num,
                                           num_decoder_layers=t_decoder_num,
                                           normalize_before=False,
                                           return_intermediate_dec=True, feat_height=output_size[0],
                                           feat_width=output_size[1])

        self.pe = PositionalEncoding(d_model=d_model, dropout=0.1, max_len=5000)

        self.output_size = output_size
        self.seq_len = output_size[1] * output_size[0]  # output_size[1] ** 2 #
        self.init_factor = nn.Embedding(self.seq_len, d_model)

        self.masking = torch.ones(output_size)

        # self.tp_uper = InfoGen(t_emb, out_text_channels)

    def forward(self, image_feature, tp_input):
        # H, W = self.output_size
        x = tp_input  # b,h,1,l
        # print(x.shape)

        N_i, C_i, H_i, W_i = image_feature.shape
        H, W = H_i, W_i

        x_tar = image_feature

        # [1024, N, 64]
        x_im = x_tar.view(N_i, C_i, H_i * W_i).permute(2, 0, 1)

        device = x.device
        # print('x:{} s:{}'.format( x.shape,s.shape))

        x = x.permute(0, 3, 1, 2).squeeze(-1)  # b,l,h
        x = self.activation(self.fc_in(x))
        N, L, C = x.shape

        x_pos = self.pe(torch.zeros((N, L, C)).to(device)).permute(1, 0, 2)
        x_mask = torch.zeros((N, L)).to(device).bool()
        x = x.permute(1, 0, 2)  # l,b,h (26,b,1024)

        # print('sequence shape', x.shape)

        text_prior, pr_weights = self.transformer(x, x_mask, self.init_factor.weight, x_pos,
                                                  # s, s_mask, s_pos,
                                                  tgt=x_im, spatial_size=(H, W))  # self.init_factor.weight
        text_prior = text_prior.mean(0)
        # print(text_prior.shape)
        # exit(0)
        text_prior = text_prior.permute(1, 2, 0).view(N, C, H, W)

        return text_prior

# class TP_generator(nn.Module):
#     def __init__(self,
#                  scale_factor=2,
#                  width=128,
#                  height=32,
#                  STN=False,
#                  srb_nums=5,
#                  mask=True,
#                  hidden_units=32,
#                  word_vec_d=300,
#                  text_emb=37,  # 37, #26+26+1 3965
#                  out_text_channels=64,  # 32 256
#                  feature_rotate=False,
#                  rotate_train=3.):
#         super(TP_generator, self).__init__()
#         in_planes = 3
#         if mask:
#             in_planes = 4
#         assert math.log(scale_factor, 2) % 1 == 0
#         upsample_block_num = int(math.log(scale_factor, 2))
#         self.block1 = nn.Sequential(
#             nn.Conv2d(in_planes, 2 * hidden_units, kernel_size=9, padding=4),
#             nn.PReLU()
#         )
#
#         self.infoGen = TPInterpreter(text_emb, out_text_channels, output_size=(
#         height // scale_factor, width // scale_factor))  # InfoGen(text_emb, out_text_channels)
#
#         self.feature_rotate = feature_rotate
#         self.rotate_train = rotate_train
#
#         if not SHUT_BN:
#             setattr(self, 'block%d' % (srb_nums + 2),
#                     nn.Sequential(
#                         nn.Conv2d(2 * hidden_units, 2 * hidden_units, kernel_size=3, padding=1),
#                         nn.BatchNorm2d(2 * hidden_units)
#                     ))
#         else:
#             setattr(self, 'block%d' % (srb_nums + 2),
#                     nn.Sequential(
#                         nn.Conv2d(2 * hidden_units, 2 * hidden_units, kernel_size=3, padding=1),
#                         # nn.BatchNorm2d(2 * hidden_units)
#                     ))
#
#         block_ = [UpsampleBLock(2 * hidden_units, 2) for _ in range(upsample_block_num)]
#         block_.append(nn.Conv2d(2 * hidden_units, in_planes, kernel_size=9, padding=4))
#         setattr(self, 'block%d' % (srb_nums + 3), nn.Sequential(*block_))
#         self.tps_inputsize = [height // scale_factor, width // scale_factor]
#         tps_outputsize = [height // scale_factor, width // scale_factor]
#         num_control_points = 20
#         tps_margins = [0.05, 0.05]
#         self.stn = STN
#         if self.stn:
#             self.tps = TPSSpatialTransformer(
#                 output_image_size=tuple(tps_outputsize),
#                 num_control_points=num_control_points,
#                 margins=tuple(tps_margins))
#
#             self.stn_head = STNHead(
#                 in_planes=in_planes,
#                 num_ctrlpoints=num_control_points,
#                 activation='none',
#                 input_size=self.tps_inputsize)
#
#         self.block_range = [k for k in range(2, self.srb_nums + 2)]
#
#     # print("self.block_range:", self.block_range)
#
#     def forward(self, x, text_emb=None, text_emb_gt=None, feature_arcs=None, rand_offs=None, stroke_map=None):
#
#         if self.stn and self.training:
#             _, ctrl_points_x = self.stn_head(x)
#             x, _ = self.tps(x, ctrl_points_x)
#         block = {'1': self.block1(x)}
#
#         if text_emb is None:
#             text_emb = torch.zeros(1, 37, 1, 26).to(x.device)  # 37
#         if stroke_map is None:
#             stroke_map = torch.zeros(1, 16, 26, 256).to(x.device)
#         padding_feature = block['1']
#
#         tp_map_gt, pr_weights_gt = None, None
#         # print('text_emb:{} stroke_map:{}'.format( text_emb.shape,stroke_map.shape))
#         tp_map, pr_weights = self.infoGen(padding_feature, text_emb, stroke_map)
#         # N, C, H, W


class PARSeqTPG(nn.Module):
    def __init__(self, checkpoint_path, parseq_repo_path, target_dim=37, max_len=26,
                 device='cuda', use_format_constraint=False, freeze_backbone=True, **kwargs):
        super().__init__()
        self.max_len = max_len
        self.timestamp = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        self.use_format_constraint = use_format_constraint
        self.freeze_backbone = freeze_backbone
        self.output_is_prob = True
        checkpoint_path = os.path.abspath(checkpoint_path)
        parseq_repo_path = os.path.abspath(parseq_repo_path)

        if parseq_repo_path not in sys.path:
            sys.path.insert(0, parseq_repo_path)

        try:
            from strhub.models.parseq.system import PARSeq
            from strhub.models.utils import create_model
        except ImportError as exc:
            raise RuntimeError(f"PARSeq import failed: {exc}") from exc

        print(f"--- PARSeqTPG: loading weights ({checkpoint_path}) ---")

        try:
            self.parseq = PARSeq.load_from_checkpoint(checkpoint_path).to(self.device)
            print("PARSeq checkpoint loaded with load_from_checkpoint")
        except Exception as e:
            print(f"PARSeq load_from_checkpoint failed: {e}")
            print("Trying local torch.hub fallback for PARSeq")
            try:
                ckpt = torch.load(checkpoint_path, map_location=self.device)
                state_dict = ckpt.get('state_dict', ckpt)
                state_dict = self._strip_model_prefix(state_dict)
                model_kwargs = self._infer_model_kwargs_from_state_dict(state_dict)
                self.parseq = create_model('parseq', False, **model_kwargs).to(self.device)
                new_state_dict = {}
                for k, v in state_dict.items():
                    name = k if k.startswith('model.') else f'model.{k}'
                    new_state_dict[name] = v
                self.parseq.load_state_dict(new_state_dict, strict=True)
                print("PARSeq weights loaded with manual fallback")
            except Exception as e_final:
                raise RuntimeError(f"PARSeq fallback load failed: {e_final}") from e_final

        if self.freeze_backbone:
            self.parseq.eval()
            for p in self.parseq.parameters():
                p.requires_grad = False
        else:
            self.parseq.train()
            for p in self.parseq.parameters():
                p.requires_grad = True

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.parseq_output_dim = int(self.parseq.model.head.out_features)
        if self.parseq_output_dim == target_dim:
            mapping = list(range(target_dim))
        elif self.parseq_output_dim >= 63:
            mapping = [0] + list(range(1, 11)) + list(range(37, 63))
        else:
            raise RuntimeError(
                f"Unsupported PARSeq output dimension {self.parseq_output_dim} for target_dim={target_dim}"
            )
        self.register_buffer('mapping_tensor', torch.tensor(mapping))
        self.to(self.device)

    @staticmethod
    def _strip_model_prefix(state_dict):
        cleaned = OrderedDict()
        for k, v in state_dict.items():
            name = k[len('model.'):] if k.startswith('model.') else k
            cleaned[name] = v
        return cleaned

    @staticmethod
    def _infer_model_kwargs_from_state_dict(state_dict):
        charset_36 = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        charset_94_full = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"

        pos_queries = state_dict.get('pos_queries')
        head_weight = state_dict.get('head.weight')
        if pos_queries is None or head_weight is None:
            raise RuntimeError("PARSeq checkpoint missing `pos_queries` or `head.weight` for fallback init")

        max_label_length = int(pos_queries.shape[1] - 1)
        num_classes = int(head_weight.shape[0])

        if num_classes == 37:
            charset_train = charset_36
            charset_test = charset_36
        elif num_classes == 95:
            # PARSeq official pretrained uses 94_full charset (+EOS => 95 head dim).
            charset_train = charset_94_full
            charset_test = "0123456789abcdefghijklmnopqrstuvwxyz"
        else:
            raise RuntimeError(
                f"Unsupported PARSeq checkpoint head size {num_classes}. "
                "Expected one of {37, 95}."
            )

        return {
            'charset_train': charset_train,
            'charset_test': charset_test,
            'max_label_length': max_label_length,
            'batch_size': 1,
            'lr': 1e-4,
            'warmup_pct': 0.0,
            'weight_decay': 0.0,
        }

    def apply_format_constraint(self, probs):
        mask = torch.ones_like(probs)

        idx_eos = 0
        idx_digit = slice(1, 11)
        idx_alpha = slice(11, 37)
        seq_len = probs.size(1)

        if seq_len > 0:
            mask[:, 0:3, idx_eos] = 0
            mask[:, 0:3, idx_digit] = 0
        if seq_len > 3:
            mask[:, 3, idx_eos] = 0
            mask[:, 3, idx_alpha] = 0
        if seq_len > 4:
            mask[:, 4, idx_eos] = 0
        if seq_len > 5:
            mask[:, 5:7, idx_eos] = 0
            mask[:, 5:7, idx_alpha] = 0
        if seq_len > 7:
            mask[:, 7:, 1:] = 0

        return probs * mask

    def _renorm_probs(self, x, eps=1e-8):
        return x / (x.sum(dim=-1, keepdim=True) + eps)

    def forward(self, image):
        b, c_total, h, w = image.shape
        num_frames = c_total // 3

        img = image.view(b, num_frames, 3, h, w).reshape(b * num_frames, 3, h, w)

        if img.min() < 0:
            img = (img + 1.0) / 2.0

        v_min = img.view(img.size(0), -1).min(dim=1)[0].view(-1, 1, 1, 1)
        v_max = img.view(img.size(0), -1).max(dim=1)[0].view(-1, 1, 1, 1)
        valid_mask = (v_max - v_min) > 1e-5
        img = torch.where(valid_mask, (img - v_min) / (v_max - v_min + 1e-6), img)
        img = torch.clamp(img, 0, 1)

        img = F.interpolate(img, size=(32, 128), mode='bicubic', align_corners=False)
        img = (img - self.mean) / self.std

        logits = self.parseq(img)
        if isinstance(logits, dict):
            logits = logits.get("logits", logits.get("output", logits))
        elif isinstance(logits, (tuple, list)):
            logits = logits[0]

        probs = F.softmax(logits, dim=-1)
        probs = probs.view(b, num_frames, probs.shape[1], probs.shape[2]).mean(dim=1)

        if self.parseq_output_dim == self.mapping_tensor.numel():
            out = probs[:, :, self.mapping_tensor]
        else:
            probs_combined = probs.clone()
            probs_combined[:, :, 37:63] += probs[:, :, 11:37]
            out = probs_combined[:, :, self.mapping_tensor]

        if self.use_format_constraint:
            out = self.apply_format_constraint(out)

        out = self._renorm_probs(out)
        final_out = out.clone()

        if not self.training:
            pred_ids = out.argmax(dim=-1)
            for batch_idx in range(out.size(0)):
                eos_pos = (pred_ids[batch_idx] == 0).nonzero(as_tuple=True)[0]
                if len(eos_pos) > 0:
                    first_eos = eos_pos[0].item()
                    if first_eos < self.max_len - 1:
                        final_out[batch_idx, first_eos + 1:, :] = 0
                        final_out[batch_idx, first_eos + 1:, 0] = 1.0

            final_out = self._renorm_probs(final_out)

        return final_out

    def save_state_dict(self, path, epoch):
        os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, self.timestamp + f'-e{epoch}.pth')
        torch.save(self.state_dict(), save_path)
