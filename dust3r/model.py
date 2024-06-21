# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# DUSt3R model class
# --------------------------------------------------------
from copy import deepcopy
import torch
import os
from packaging import version
import huggingface_hub

from .utils.misc import fill_default_args, freeze_all_params, is_symmetrized, interleave, transpose_to_landscape
from .heads import head_factory
from dust3r.patch_embed import get_patch_embed

import dust3r.utils.path_to_croco  # noqa: F401
from models.croco import CroCoNet  # noqa

inf = float('inf')

hf_version_number = huggingface_hub.__version__
assert version.parse(hf_version_number) >= version.parse("0.22.0"), "Outdated huggingface_hub version, please reinstall requirements.txt"

def load_model(model_path, device, verbose=True):
    if verbose:
        print('... loading model from', model_path)
    ckpt = torch.load(model_path, map_location='cpu')
    args = ckpt['args'].model.replace("ManyAR_PatchEmbed", "PatchEmbedDust3R")
    if 'landscape_only' not in args:
        args = args[:-1] + ', landscape_only=False)'
    else:
        args = args.replace(" ", "").replace('landscape_only=True', 'landscape_only=False')
    assert "landscape_only=False" in args
    if verbose:
        print(f"instantiating : {args}")
    net = eval(args)
    s = net.load_state_dict(ckpt['model'], strict=False)
    if verbose:
        print(s)
    return net.to(device)


class AsymmetricCroCo3DStereo (
    CroCoNet,
    huggingface_hub.PyTorchModelHubMixin,
    library_name="dust3r",
    repo_url="https://github.com/naver/dust3r",
    tags=["image-to-3d"],
):
    """ Two siamese encoders, followed by two decoders.
    The goal is to output 3d points directly, both images in view1's frame
    (hence the asymmetry).   
    """

    def __init__(self,
                 output_mode='pts3d',
                 head_type='linear',
                 depth_mode=('exp', -inf, inf),
                 conf_mode=('exp', 1, inf),
                 freeze='none',
                 landscape_only=True,
                 patch_embed_cls='PatchEmbedDust3R',  # PatchEmbedDust3R or ManyAR_PatchEmbed
                 **croco_kwargs):
        self.patch_embed_cls = patch_embed_cls
        self.croco_args = fill_default_args(croco_kwargs, super().__init__)
        super().__init__(**croco_kwargs)

        # dust3r specific initialization
        self.dec_blocks2 = deepcopy(self.dec_blocks)
        self.set_downstream_head(output_mode, head_type, landscape_only, depth_mode, conf_mode, **croco_kwargs)
        self.set_freeze(freeze)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        if os.path.isfile(pretrained_model_name_or_path):
            return load_model(pretrained_model_name_or_path, device='cpu')
        else:
            return super(AsymmetricCroCo3DStereo, cls).from_pretrained(pretrained_model_name_or_path, **kw)

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_embed = get_patch_embed(self.patch_embed_cls, img_size, patch_size, enc_embed_dim)

    def load_state_dict(self, ckpt, **kw):
        # duplicate all weights for the second decoder if not present
        new_ckpt = dict(ckpt)
        if not any(k.startswith('dec_blocks2') for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith('dec_blocks'):
                    new_ckpt[key.replace('dec_blocks', 'dec_blocks2')] = value
        return super().load_state_dict(new_ckpt, **kw)

    def set_freeze(self, freeze):  # this is for use by downstream models
        self.freeze = freeze
        to_be_frozen = {
            'none':     [],
            'mask':     [self.mask_token],
            'encoder':  [self.mask_token, self.patch_embed, self.enc_blocks],
        }
        freeze_all_params(to_be_frozen[freeze])

    def _set_prediction_head(self, *args, **kwargs):
        """ No prediction head """
        return

    def set_downstream_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode, patch_size, img_size,
                            **kw):
        assert img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0, \
            f'{img_size=} must be multiple of {patch_size=}'
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        # allocate heads
        self.downstream_head1 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        self.downstream_head2 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        # magic wrapper
        self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)
        self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)

    def _encode_image(self, image, true_shape):
        # embed the image into patches  (x has size B x Npatches x C)
        x, pos = self.patch_embed(image, true_shape=true_shape)

        # add positional embedding without cls token
        assert self.enc_pos_embed is None

        # now apply the transformer encoder and normalization
        for blk in self.enc_blocks:
            x = blk(x, pos)

        x = self.enc_norm(x)
        return x, pos, None

    def _encode_image_pairs(self, img_ref, imgs_source, true_shape_ref, true_shapes_source):
        if img_ref.shape[-2:] == imgs_source[0].shape[-2:]:
            out, pos, _ = self._encode_image(torch.cat((img_ref, *imgs_source), dim=0),
                                             torch.cat((true_shape_ref, *true_shapes_source), dim=0))
            out_ref, outs_source = out[0], out[1:]
            pos_ref, pos_source = pos[0], pos[1:]
        else:
            out_ref, pos_ref, _ = self._encode_image(img_ref, true_shape_ref)
            outs_source, pos_source = [], []
            for img, true_shape in zip(imgs_source, true_shapes_source):
                out, pos, _ = self._encode_image(img, true_shape)
                outs_source.append(out)
                pos_source.append(pos)

        return out_ref, outs_source, pos_ref, pos_source

    def _encode_symmetrized(self, view_ref: dict[str,torch.Tensor], views_source: list[dict[str, torch.Tensor]]):
        if not isinstance(views_source, (tuple, list)):
            views_source = [views_source]
        img_ref = view_ref['img']
        imgs_source = [view['img'] for view in views_source]
        B = img_ref.shape[0]
        # Recover true_shape when available, otherwise assume that the img shape is the true one
        shape_ref = view_ref.get('true_shape', torch.tensor(img_ref.shape[-2:])[None].repeat(B, 1))
        shapes_source = [views_source[i].get('true_shape', torch.tensor(imgs_source[i].shape[-2:])[None].repeat(B, 1)) for i in range(len(imgs_source))]
        # warning! maybe the images have different portrait/landscape orientations
        feat_ref, feats_source, pos_ref, pos_source = self._encode_image_pairs(img_ref, imgs_source, shape_ref, shapes_source)

        return (shape_ref, shapes_source), (feat_ref, feats_source), (pos_ref, pos_source)

    def _decoder(self, f_ref, pos_ref, fs_source, pos_source):
        final_output = [(f_ref, fs_source)]  # before projection

        # project to decoder dim
        f_ref = self.decoder_embed(f_ref)
        fs_source = [self.decoder_embed(f_source) for f_source in fs_source]

        final_output.append((f_ref, fs_source))
        for blk_ref, blk_source in zip(self.dec_blocks, self.dec_blocks2):
            # img1 side
            f_ref, _ = blk_ref(*final_output[-1][::+1], pos_ref, pos_source)
            # img2 side
            fs_source, _ = blk_source(*final_output[-1][::-1], pos_source, pos_ref)
            # store the result
            final_output.append((f_ref, fs_source))

        # normalize last output
        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = (self.dec_norm(f_ref), [self.dec_norm(f_source) for f_source in fs_source])
        dec_ref = [o[0] for o in final_output]
        decs_source = []
        for v in range(len(final_output[0][1])):
            decs_source.append([o[1][v] for o in final_output])
        return dec_ref, decs_source

    def _downstream_head(self, head_num, decout, img_shape):
        B, S, D = decout[-1].shape
        # img_shape = tuple(map(int, img_shape))
        head = getattr(self, f'head{head_num}')
        return head(decout, img_shape)

    def forward(self, view_ref, views_source: torch.Tensor | list[torch.Tensor]):
        # encode the two images --> B,S,D
        (shape_ref, shapes_source), (feat_ref, feats_source), (pos_ref, pos_source) = self._encode_symmetrized(view_ref, views_source)

        # combine all ref images into object-centric representation
        dec_ref, decs_source = self._decoder(feat_ref, pos_ref, feats_source, pos_source)

        with torch.cuda.amp.autocast(enabled=False):
            res_ref = self._downstream_head(1, [tok.float() for tok in dec_ref], shape_ref)
            ress_source = [self._downstream_head(2, [tok.float() for tok in decs_source[i]], shapes_source[i]) for i in range(len(decs_source))]

        for res_source in ress_source:
            res_source['pts3d_in_other_view'] = res_source.pop('pts3d')  # predict view2's pts3d in view1's frame
        return res_ref, ress_source
