from ldm.models.diffusion.ddpm_5 import LatentDiffusion as LatentDiffusion5


class LatentDiffusion(LatentDiffusion5):
    """
    Single-LR variant:
    - keeps only the primary LR frame as concat condition
    - disables multi-LR attention token path by default
    """

    def __init__(self, *args, force_single_lr=True, **kwargs):
        self.force_single_lr = bool(force_single_lr)
        if "multi_lr_attn_enable" not in kwargs:
            kwargs["multi_lr_attn_enable"] = False
        super().__init__(*args, **kwargs)

    def _compose_concat_lr(self, image):
        if not self.force_single_lr:
            return super()._compose_concat_lr(image)
        return self._extract_primary_lr_rgb(image)

    def _collapse_multi_lr_to_rgb(self, image):
        if not self.force_single_lr:
            return super()._collapse_multi_lr_to_rgb(image)
        return self._extract_primary_lr_rgb(image)

    def _compute_multi_lr_token(self, c):
        if self.force_single_lr:
            return None
        return super()._compute_multi_lr_token(c)
