#!/usr/bin/env python3
# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
import os
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities.model_summary import summarize

from strhub.data.module import SceneTextDataModule
from strhub.models.base import BaseSystem
from strhub.models.utils import get_pretrained_weights


# Copied from OneCycleLR
def _annealing_cos(start, end, pct):
    'Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0.'
    cos_out = math.cos(math.pi * pct) + 1
    return end + (start - end) / 2.0 * cos_out


def get_swa_lr_factor(warmup_pct, swa_epoch_start, div_factor=25, final_div_factor=1e4) -> float:
    """Get the SWA LR factor for the given `swa_epoch_start`. Assumes OneCycleLR Scheduler."""
    total_steps = 1000  # Can be anything. We use 1000 for convenience.
    start_step = int(total_steps * warmup_pct) - 1
    end_step = total_steps - 1
    step_num = int(total_steps * swa_epoch_start) - 1
    pct = (step_num - start_step) / (end_step - start_step)
    return _annealing_cos(1, 1 / (div_factor * final_div_factor), pct)


@hydra.main(config_path='configs', config_name='main', version_base='1.2')
def main(config: DictConfig):
    trainer_strategy = 'auto'
    with open_dict(config):
        # Resolve absolute path to data.root_dir
        config.data.root_dir = hydra.utils.to_absolute_path(config.data.root_dir)
        # Special handling for GPU-affected config
        gpu = config.trainer.get('accelerator') == 'gpu'
        devices = config.trainer.get('devices', 0)
        if gpu:
            # Use mixed-precision training
            config.trainer.precision = 'bf16-mixed' if torch.get_autocast_gpu_dtype() is torch.bfloat16 else '16-mixed'
        if gpu and devices > 1:
            # Use DDP with optimizations
            trainer_strategy = DDPStrategy(find_unused_parameters=False, gradient_as_bucket_view=True)
            # Scale steps-based config
            config.trainer.val_check_interval //= devices
            if config.trainer.get('max_steps', -1) > 0:
                config.trainer.max_steps //= devices
            # 그 줄 바로 밑에 이렇게 적으세요.
            config.trainer.val_check_interval = 1 
            config.trainer.limit_val_batches = 1  # <--- 이 줄을 추가!

    # Special handling for PARseq
    if config.model.get('perm_mirrored', False):
        assert config.model.perm_num % 2 == 0, 'perm_num should be even if perm_mirrored = True'

    model: BaseSystem = hydra.utils.instantiate(config.model)
# If specified, use pretrained weights to initialize the model
    if config.pretrained is not None:
        from strhub.models.utils import get_pretrained_weights
        
        m = model.model if config.model._target_.endswith('PARSeq') else model
        
        # 1. 로컬 파일 경로인지 체크
        if os.path.exists(config.pretrained):
            print(f"✅ 로컬 가중치 파일 로드 중: {config.pretrained}")
            checkpoint = torch.load(config.pretrained, map_location='cpu')
            
            # .pt 파일 구조에 따라 state_dict 추출
            state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint

            # 2. 현재 모델과 크기가 맞는 가중치만 골라내기 (95개 -> 37개 필터링)
            model_state_dict = m.state_dict()
            filtered_state_dict = {}
            
            for k, v in state_dict.items():
                if k in model_state_dict and v.shape == model_state_dict[k].shape:
                    filtered_state_dict[k] = v
                else:
                    # 크기가 다른 head나 text_embed 등은 여기서 제외됨
                    print(f"⚠️ 레이어 무시됨 (크기 불일치): {k}")

            # 3. 필터링된 가중치 로드
            m.load_state_dict(filtered_state_dict, strict=False)
            print("✨ 뼈대 가중치 로드 완료! (출력 레이어는 새로 학습됩니다.)")
            
        else:
            # 파일이 없으면 공식 이름으로 간주하고 다운로드 시도
            print(f"🚀 공식 별명으로 가중치 로드 시도: {config.pretrained}")
            m.load_state_dict(get_pretrained_weights(config.pretrained))
    print(summarize(model, max_depth=2))

    datamodule: SceneTextDataModule = hydra.utils.instantiate(config.data)

    checkpoint = ModelCheckpoint(
        monitor='val_accuracy',
        mode='max',
        save_top_k=3,
        save_last=True,
        filename='{epoch}-{step}-{val_accuracy:.4f}-{val_NED:.4f}',
    )
    swa_epoch_start = 0.75
    swa_lr = config.model.lr * get_swa_lr_factor(config.model.warmup_pct, swa_epoch_start)
    swa = StochasticWeightAveraging(swa_lr, swa_epoch_start)
    cwd = (
        HydraConfig.get().runtime.output_dir
        if config.ckpt_path is None
        else str(Path(config.ckpt_path).parents[1].absolute())
    )
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer,
        logger=TensorBoardLogger(cwd, '', '.'),
        strategy=trainer_strategy,
        enable_model_summary=False,
        callbacks=[checkpoint, swa],
    )
    trainer.fit(model, datamodule=datamodule, ckpt_path=config.ckpt_path)


if __name__ == '__main__':
    main()
