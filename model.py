import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights
from models.modeling import VisionTransformer, CONFIGS
#from deit.models_v2 import deit_base_patch16_LS, vit_models
from models.models_v2 import vit_models
from timm.models.layers import to_2tuple

from models.HelixFormer.conv_helix_transformer import ConvHelixTransformer
from models.cpd import Attention, cross_Attention

from einops import rearrange
import math

from matplotlib import pyplot as plt
import cv2
from PIL import Image

from functools import partial
from deit.models_v2 import Layer_scale_init_Block

from models_emd.Network import DeepEMD

#pretrain_dist = '/mnt/HDD1/shih/OSV/vit/ViT/pretrain/imagenet21k/ViT-B_16.npz'

class my_deit_model(vit_models):
    def forward_features_all(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        x = x + self.pos_embed
        
        x = torch.cat((cls_tokens, x), dim=1)
            
        for i , blk in enumerate(self.blocks):
            x = blk(x)
            
        x = self.norm(x)
        return x
    
    def create_my_model(img_size=224):
        model = my_deit_model(img_size=img_size, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=Layer_scale_init_Block, Patch_layer=PatchEmbed_v0)
        checkpoint = torch.load('/mnt/HDD1/shih/OSV/deit_osv/pretrain/deit_3_base_224_21k.pth')
        model.load_state_dict(checkpoint["model"])

        return model


class PatchEmbed_v0(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            norm_layer=None,
            flatten=True,
            bias=True,
    ):
        super().__init__()
        #
        stride_size = (img_size - patch_size) // 13 # calculate stride size
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        #
        stride_size = to_2tuple(stride_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (((img_size[0]-patch_size[0]) // stride_size[0]) +1, ((img_size[1]-patch_size[1]) // stride_size[1]) +1)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

class ViT_for_OSV(nn.Module):
    def __init__(self, vis=False):
        super(ViT_for_OSV, self).__init__()
        '''
        ### torchvision pretrain ###
        self.model = vit_b_16(weights='DEFAULT')
        self.model.conv_proj = nn.Conv2d(1, 768, kernel_size=(16, 16), stride=(16, 16))
        self.model.heads = Identity()
        '''
        '''
        ### ViT jeonsworld ###
        config = CONFIGS['ViT-B_16']
        self.model = VisionTransformer(config)
        self.model = VisionTransformer(config, vis=vis)
        self.model.load_from(np.load(pretrain_dist))
        self.model.transformer.embeddings.patch_embeddings = nn.Conv2d(1, 768, kernel_size=(16, 16), stride=(16, 16))
        self.model.head = nn.Identity()
        '''
        ### deit facebook ###
        self.model = my_deit_model.create_my_model()
        self.model.patch_embed.proj = nn.Conv2d(1, 768, kernel_size=(16, 16), stride=(16, 16))
        self.model.head = nn.Identity()
        
        self.pdist = nn.PairwiseDistance(p=2, keepdim = True)
        #self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward_one(self, x):
        #x = self.model(x) # normal ver, return class token
        ### deit facebook ###
        x = self.model.forward_features_all(x) # return all token
        x = x[:, 0]
        return x
        
    def forward(self, x):
        batch_size, _, h, w = x.shape # [batch_size, 2, h, w]
        x1 = x[:, 0, :, :].unsqueeze(1)
        x2 = x[:, 1, :, :].unsqueeze(1)
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        
        out = self.pdist(out1, out2)
        #out = 1 - self.cos(out1, out2).unsqueeze(1)
        #out = self.out(torch.cat((out1, out2), 1))
        return out
    
    def forward_one_test(self, x):
        x, att_mat = self.model(x, all=True)
        
        att_mat = torch.stack(att_mat).squeeze(1)
        att_mat = torch.mean(att_mat, dim=1)
        #'''
        # To account for residual connections, we add an identity matrix to the
        # attention matrix and re-normalize the weights.
        residual_att = torch.eye(att_mat.size(1)).cuda()
        aug_att_mat = att_mat + residual_att
        aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

        # Recursively multiply the weight matrices
        joint_attentions = torch.zeros(aug_att_mat.size()).cuda()
        joint_attentions[0] = aug_att_mat[0]

        for n in range(1, aug_att_mat.size(0)):
            joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])
        
        # Attention from the output token to the input space.
        joint_attentions_ = torch.mean(joint_attentions, dim=0)
        #v = joint_attentions[-1]
        v = joint_attentions_
        grid_size = int(np.sqrt(aug_att_mat.size(-1)))
        mask = v[0, 1:].reshape(grid_size, grid_size).detach().cpu().numpy()
        #'''
        return x, mask
    
    def forward_test(self, x, path):
        batch_size, _, h, w = x.shape # [batch_size, 2, h, w]
        t = transforms.ToPILImage()
        x1 = x[:, 0, :, :].unsqueeze(1)
        x2 = x[:, 1, :, :].unsqueeze(1)

        out1, mask_1 = self.forward_one_test(x1)
        out2, mask_2 = self.forward_one_test(x2)
        #im_1 = Image.open(path['Anchor'][0])
        im_1 = t(x[:, 0, :, :])
        mask_1 = cv2.resize(mask_1 / mask_1.max(), im_1.size, interpolation=cv2.INTER_AREA)
        #im_2 = Image.open(path['ref'][0])
        im_2 = t(x[:, 1, :, :])
        mask_2 = cv2.resize(mask_2 / mask_2.max(), im_2.size, interpolation=cv2.INTER_AREA)

        result_1 = (mask_1 * im_1).astype("uint8")
        #plt.imsave('att_result_0.png', result_1)
        result_2 = (mask_2 * im_2).astype("uint8")
        #plt.imsave('att_result_1.png', result_2)
        #plt.close()
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows= 2, ncols = 2, figsize=(16, 8))

        ax1.set_title('att_result_0')
        ax2.set_title('att_result_1')
        z1_plot = ax1.imshow(result_1)
        z2_plot = ax2.imshow(result_2)

        z3_plot = ax3.imshow(mask_1)
        z4_plot = ax4.imshow(mask_2)

        plt.colorbar(z1_plot, ax=ax1)
        plt.colorbar(z2_plot, ax=ax2)
        plt.colorbar(z3_plot, ax=ax3)
        plt.colorbar(z4_plot, ax=ax4)
        plt.savefig('att_result.png')
        plt.close()

        out = self.pdist(out1[:, 0], out2[:, 0]) # class token
        return out
    
    def save_to_file(self, file_path: str) -> None:
        torch.save(self.state_dict(), file_path)

class ViT_for_OSV_emd(nn.Module):
    def __init__(self, opt, vis=False):
        super(ViT_for_OSV_emd, self).__init__()
        ### deit facebook ###
        self.model = my_deit_model.create_my_model()
        self.model.patch_embed.proj = nn.Conv2d(1, 768, kernel_size=(16, 16), stride=(16, 16))
        self.model.head = nn.Identity()
        
        self.pdist = nn.PairwiseDistance(p=2, keepdim = True)
        #self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        self.opt = opt

    def forward_one(self, x):
        #x = self.model(x) # normal ver, return class token
        ### deit facebook ###
        x = self.model.forward_features_all(x) # return all token
        return x
        
    def forward(self, x):
        batch_size, _, h, w = x.shape # [batch_size, 2, h, w]
        x1 = x[:, 0, :, :].unsqueeze(1)
        x2 = x[:, 1, :, :].unsqueeze(1)
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)

        class_token_1 = out1[:, 0]
        class_token_2 = out2[:, 0]

        out1_ = out1[:, 1:]
        out2_ = out2[:, 1:]

        data_shot = rearrange(out1_, 'b (h w) c -> b c h w', w=14).contiguous()
        data_query = rearrange(out2_, 'b (h w) c -> b c h w', w=14).contiguous()

        emd =  DeepEMD(self.opt)
        outs = []
        for i in range(data_shot.size(dim=0)):
            data_shot_ = data_shot[i].unsqueeze(0).unsqueeze(0)
            data_query_ = data_query[i].unsqueeze(0)
            pred = emd.emd_forward_1shot(data_shot_, data_query_, x[i])
            outs.append(pred.squeeze())
        preds_str = torch.stack(outs)[:,None].cuda()
        preds_gol = self.pdist(class_token_1, class_token_2)
        #out = 1 - self.cos(out1, out2).unsqueeze(1)
        #out = self.out(torch.cat((out1, out2), 1))
        
        #preds = 0.5 * (preds_str + preds_gol)
        preds = (preds_str)
        #preds = (preds_gol)
        return preds
    
    def forward_test(self, x, path):
        batch_size, _, h, w = x.shape # [batch_size, 2, h, w]
        x1 = x[:, 0, :, :].unsqueeze(1)
        x2 = x[:, 1, :, :].unsqueeze(1)
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)

        class_token_1 = out1[:, 0]
        class_token_2 = out2[:, 0]

        out1_ = out1[:, 1:]
        out2_ = out2[:, 1:]

        data_shot = rearrange(out1_, 'b (h w) c -> b c h w', w=14).contiguous()
        data_query = rearrange(out2_, 'b (h w) c -> b c h w', w=14).contiguous()

        emd =  DeepEMD(self.opt)
        outs = []
        for i in range(data_shot.size(dim=0)):
            data_shot_ = data_shot[i].unsqueeze(0).unsqueeze(0)
            data_query_ = data_query[i].unsqueeze(0)
            pred = emd.emd_forward_1shot_test(data_shot_, data_query_, x[i])
            outs.append(pred.squeeze())
        preds_str = torch.stack(outs)[:,None].cuda()
        preds_gol = self.pdist(class_token_1, class_token_2)
        #out = 1 - self.cos(out1, out2).unsqueeze(1)
        #out = self.out(torch.cat((out1, out2), 1))
        preds = (preds_str)
        return preds
    
    def save_to_file(self, file_path: str) -> None:
        torch.save(self.state_dict(), file_path)

class ViT_for_OSV_v2(nn.Module):
    def __init__(self):
        super(ViT_for_OSV_v2, self).__init__()
        '''
        config = CONFIGS['ViT-B_16']
        self.model = VisionTransformer(config)
        self.model.load_from(np.load(pretrain_dist))
        self.model.transformer.embeddings.patch_embeddings = nn.Conv2d(1, 768, kernel_size=(16, 16), stride=(16, 16))
        self.model.head = nn.Identity()
        '''
        ### deit facebook ###
        self.model = my_deit_model.create_my_model()
        self.model.patch_embed.proj = nn.Conv2d(1, 768, kernel_size=(16, 16), stride=(16, 16))
        self.model.head = nn.Identity()

        self.cross_model = ConvHelixTransformer(dim=768, emb_dim=768)
        self.pool = nn.AvgPool2d((14,14))
        
        self.pdist = nn.PairwiseDistance(p=2, keepdim = True)

    def forward_one(self, x):
        #x, attn_weights = self.model(x, all=True)
        ### deit facebook ###
        x = self.model.forward_features_all(x) # return all token
        return x
        
    def forward(self, x):
        batch_size, _, h, w = x.shape # [batch_size, 2, h, w]
        x1 = x[:, 0, :, :].unsqueeze(1)
        x2 = x[:, 1, :, :].unsqueeze(1)
        out1 = self.forward_one(x1) # batch_size * [h*w]+1 * channels
        out2 = self.forward_one(x2) # batch_size * [h*w]+1 * channels

        out1_ = torch.permute(out1, (1, 0, 2))[1:] # [h*w] * batch_size * channels
        out2_ = torch.permute(out2, (1, 0, 2))[1:] # [h*w] * batch_size * channels

        out1_ = rearrange(out1_, '(h w) b c -> b c h w', w=14).contiguous()
        out2_ = rearrange(out2_, '(h w) b c -> b c h w', w=14).contiguous()

        out_cross_1, out_cross_2 = self.cross_model(out1_, out2_)

        out_cross_1 = torch.flatten(self.pool(out_cross_1),start_dim=1)
        out_cross_2 = torch.flatten(self.pool(out_cross_2),start_dim=1)
        
        out = self.pdist(out1[:, 0], out2[:, 0]) # class token

        out_cross = self.pdist(out_cross_1, out_cross_2) # cross feature

        #p1 = 1.
        #total_out = out + p1*out_cross
        return out, out_cross
    
    def forward_test(self, x, path):
        batch_size, _, h, w = x.shape # [batch_size, 2, h, w]
        x1 = x[:, 0, :, :].unsqueeze(1)
        x2 = x[:, 1, :, :].unsqueeze(1)
        out1 = self.forward_one(x1) # batch_size * [h*w]+1 * channels
        out2 = self.forward_one(x2) # batch_size * [h*w]+1 * channels

        out1_ = torch.permute(out1, (1, 0, 2))[1:] # [h*w] * batch_size * channels
        out2_ = torch.permute(out2, (1, 0, 2))[1:] # [h*w] * batch_size * channels

        out1_ = rearrange(out1_, '(h w) b c -> b c h w', w=14).contiguous()
        out2_ = rearrange(out2_, '(h w) b c -> b c h w', w=14).contiguous()

        out_before_1_test =  torch.mean(out1_, dim=1).detach().cpu().numpy()
        out_before_2_test =  torch.mean(out2_, dim=1).detach().cpu().numpy()

        out_cross_1, out_cross_2 = self.cross_model(out1_, out2_)
        out_cross_1_test =  torch.mean(out_cross_1, dim=1).detach().cpu().numpy()
        out_cross_2_test =  torch.mean(out_cross_2, dim=1).detach().cpu().numpy()

        fig, ((ax00, ax01), (ax1, ax2), (ax3, ax4)) = plt.subplots(nrows= 3, ncols = 2, figsize=(8, 8))

        z00_plot = ax00.imshow(x[0, 0].detach().cpu().numpy())
        z01_plot = ax01.imshow(x[0, 1].detach().cpu().numpy())
        plt.colorbar(z00_plot, ax=ax00)
        plt.colorbar(z01_plot, ax=ax01)
        
        ax1.set_title('att_result_0')
        ax2.set_title('att_result_1')
        z1_plot = ax1.imshow(cv2.resize(out_cross_1_test[0], dsize=(224, 224)), vmin=-0.02, vmax=0.02)
        z2_plot = ax2.imshow(cv2.resize(out_cross_2_test[0], dsize=(224, 224)), vmin=-0.02, vmax=0.02)

        z3_plot = ax3.imshow(cv2.resize(out_before_1_test[0], dsize=(224, 224)))
        z4_plot = ax4.imshow(cv2.resize(out_before_2_test[0], dsize=(224, 224)))
        
        plt.colorbar(z1_plot, ax=ax1)
        plt.colorbar(z2_plot, ax=ax2)
        plt.colorbar(z3_plot, ax=ax3)
        plt.colorbar(z4_plot, ax=ax4)
        plt.savefig('att_result.png')
        plt.close()

        out_cross_1 = torch.flatten(self.pool(out_cross_1),start_dim=1)
        out_cross_2 = torch.flatten(self.pool(out_cross_2),start_dim=1)
        
        out = self.pdist(out1[:, 0], out2[:, 0]) # class token

        out_cross = self.pdist(out_cross_1, out_cross_2) # cross feature

        return out, out_cross
    
    def save_to_file(self, file_path: str) -> None:
        torch.save(self.state_dict(), file_path)

class ViT_for_OSV_v3(nn.Module):
    def __init__(self, vis=False):
        super(ViT_for_OSV_v3, self).__init__()
        hidden_dim = 768
        
        config = CONFIGS['ViT-B_16']
        self.model = VisionTransformer(config, vis=vis)
        self.model.load_from(np.load(pretrain_dist))
        self.model.transformer.embeddings.patch_embeddings = nn.Conv2d(1, hidden_dim, kernel_size=(16, 16), stride=(16, 16))
        self.model.head = nn.Identity()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        self.layernorm = nn.LayerNorm(hidden_dim)
        self.self_attn = Attention()
        self.cross_attn = cross_Attention(hidden_dim)

        self.pdist = nn.PairwiseDistance(p=2, keepdim = True)

    def forward_one(self, x):
        x, att_mat = self.model(x, all=True)
        return x

    def forward_one_test(self, x):
        x, att_mat = self.model(x, all=True)
        
        att_mat = torch.stack(att_mat).squeeze(1)
        att_mat = torch.mean(att_mat, dim=1)
        #'''
        # To account for residual connections, we add an identity matrix to the
        # attention matrix and re-normalize the weights.
        residual_att = torch.eye(att_mat.size(1)).cuda()
        aug_att_mat = att_mat + residual_att
        aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

        # Recursively multiply the weight matrices
        joint_attentions = torch.zeros(aug_att_mat.size()).cuda()
        joint_attentions[0] = aug_att_mat[0]

        for n in range(1, aug_att_mat.size(0)):
            joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])
        
        # Attention from the output token to the input space.
        v = joint_attentions[-1]
        grid_size = int(np.sqrt(aug_att_mat.size(-1)))
        mask = v[0, 1:].reshape(grid_size, grid_size).detach().cpu().numpy()
        #'''
        return x, mask

        
    def forward(self, x):
        batch_size, _, h, w = x.shape # [batch_size, 2, h, w]
        x1 = x[:, 0, :, :].unsqueeze(1)
        x2 = x[:, 1, :, :].unsqueeze(1)
        out1 = self.forward_one(x1) # batch_size * [h*w]+1 * channels
        out2 = self.forward_one(x2) # batch_size * [h*w]+1 * channels

        out1_ = out1[:,1:] # batch_size * [h*w] * channels
        out2_ = out2[:,1:] # batch_size * [h*w] * channels

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        out1_ = torch.cat((cls_tokens, out1_), dim=1)
        out2_ = torch.cat((cls_tokens, out2_), dim=1)

        # self-attention
        out_att_1, self_att_map_1 = self.self_attn(out1_, out1_, out1_) # import from cpd
        out_att_2, self_att_map_1 = self.self_attn(out2_, out2_, out2_) # import from cpd

        cls_tokens_1, cls_tokens_2 = out_att_1[:, 0].unsqueeze(dim=1), out_att_2[:, 0].unsqueeze(dim=1)
        out_att_1_, out_att_2_ = out_att_1[:,1:], out_att_2[:,1:]

        cls_tokens_1 = self.layernorm(cls_tokens_1)
        cls_tokens_2 = self.layernorm(cls_tokens_2)
        out_att_1_ = self.layernorm(out_att_1_)
        out_att_2_ = self.layernorm(out_att_2_)

        # cross-attention
        out_part_1, cross_att_map_1 = self.cross_attn(cls_tokens_1, out_att_2_, out_att_2_) # input: query, key, value
        out_part_2, cross_att_map_2 = self.cross_attn(cls_tokens_2, out_att_1_, out_att_1_) # input: query, key, value
        
        out_hol_1 = out1[:, 0]
        out_hol_2 = out2[:, 0]
        
        out = self.pdist(out_hol_1, out_hol_2) # class token
        out_part = self.pdist(out_part_1, out_part_2)

        return out, out_part
    
    def forward_test(self, x, path):
        batch_size, _, h, w = x.shape # [batch_size, 2, h, w]
        x1 = x[:, 0, :, :].unsqueeze(1)
        x2 = x[:, 1, :, :].unsqueeze(1)

        out1, mask_1 = self.forward_one_test(x1)
        out2, mask_2 = self.forward_one_test(x2)
        
        t = transforms.ToPILImage()
        
        im_1 = t(x[:, 0, :, :])
        mask_1 = cv2.resize(mask_1 / mask_1.max(), im_1.size)
        #im_2 = Image.open(path['ref'][0])
        im_2 = t(x[:, 1, :, :])
        mask_2 = cv2.resize(mask_2 / mask_2.max(), im_2.size)

        result_1 = (mask_1 * im_1).astype("uint8")
        #plt.imsave('att_result_0.png', result_1)
        result_2 = (mask_2 * im_2).astype("uint8")
        #plt.imsave('att_result_1.png', result_2)
        #plt.close()
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows= 2, ncols = 2, figsize=(16, 8))

        ax1.set_title('att_result_0')
        ax2.set_title('att_result_1')
        z1_plot = ax1.imshow(result_1)
        z2_plot = ax2.imshow(result_2)

        z3_plot = ax3.imshow(mask_1)
        z4_plot = ax4.imshow(mask_2)

        plt.colorbar(z1_plot, ax=ax1)
        plt.colorbar(z2_plot, ax=ax2)
        plt.colorbar(z3_plot, ax=ax3)
        plt.colorbar(z4_plot, ax=ax4)
        plt.savefig('att_result.png')
        plt.close()

        out1_ = out1[:,1:] # batch_size * [h*w] * channels
        out2_ = out2[:,1:] # batch_size * [h*w] * channels

        cls_tokens_1 = self.cls_token.expand(batch_size, -1, -1)
        cls_tokens_2 = self.cls_token.expand(batch_size, -1, -1)

        out1_ = torch.cat((cls_tokens_1, out1_), dim=1)
        out2_ = torch.cat((cls_tokens_2, out2_), dim=1)

        # self-attention
        out_att_1, self_att_map_1 = self.self_attn(out1_, out1_, out1_) # import from cpd
        out_att_2, self_att_map_2 = self.self_attn(out2_, out2_, out2_) # import from cpd
        '''
        im1 = self_att_map_1.squeeze().detach().cpu().numpy()
        plt.imsave('att_0.png', im1)
        im2 = self_att_map_2.squeeze().detach().cpu().numpy()
        plt.imsave('att_1.png', im2)
        plt.close()
        '''

        cls_tokens_1, cls_tokens_2 = out_att_1[:, 0].unsqueeze(dim=1), out_att_2[:, 0].unsqueeze(dim=1)
        out_att_1_, out_att_2_ = out_att_1[:,1:], out_att_2[:,1:]

        cls_tokens_1 = self.layernorm(cls_tokens_1)
        cls_tokens_2 = self.layernorm(cls_tokens_2)
        out_att_1_ = self.layernorm(out_att_1_)
        out_att_2_ = self.layernorm(out_att_2_)

        # cross-attention
        out_part_1, cross_att_map_1 = self.cross_attn(cls_tokens_1, out_att_2_, out_att_2_) # input: query, key, value
        out_part_2, cross_att_map_2 = self.cross_attn(cls_tokens_2, out_att_1_, out_att_1_) # input: query, key, value

        #print(cross_att_map_1, cross_att_map_2)
        #'''
        im1 = cross_att_map_1[0].reshape(14, 14).detach().cpu().numpy()
        im1 = cv2.resize(im1, (224,224))
        plt.imsave('att_0.png', im1)
        im2 = cross_att_map_2[0].reshape(14, 14).detach().cpu().numpy()
        im2 = cv2.resize(im2, (224,224))
        plt.imsave('att_1.png', im2)
        plt.close()
        #'''
        
        out_hol_1 = out1[:, 0]
        out_hol_2 = out2[:, 0]
        
        out = self.pdist(out_hol_1, out_hol_2) # class token
        out_part = self.pdist(out_part_1, out_part_2)

        #p1 = 1.
        #total_out = out + p1*out_part
        return out, out_part
    
    def save_to_file(self, file_path: str) -> None:
        torch.save(self.state_dict(), file_path)

class ViT_for_OSV_overlap(nn.Module):
    def __init__(self, vis=False): # size = 172
        super(ViT_for_OSV_overlap, self).__init__()
        ### deit facebook ###
        self.model = my_deit_model.create_my_model(img_size=172)
        self.model.patch_embed.proj = nn.Conv2d(1, 768, kernel_size=(16, 16), stride=(12, 12))
        self.model.head = nn.Identity()
        
        self.pdist = nn.PairwiseDistance(p=2, keepdim = True)
        #self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward_one(self, x):
        #x = self.model(x) # normal ver, return class token
        ### deit facebook ###
        x = self.model.forward_features_all(x) # return all token
        x = x[:, 0]
        return x
        
    def forward(self, x):
        batch_size, _, h, w = x.shape # [batch_size, 2, h, w]
        x1 = x[:, 0, :, :].unsqueeze(1)
        x2 = x[:, 1, :, :].unsqueeze(1)
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        
        out = self.pdist(out1, out2)
        #out = 1 - self.cos(out1, out2).unsqueeze(1)
        #out = self.out(torch.cat((out1, out2), 1))
        return out
    
    def forward_one_test(self, x):
        x, att_mat = self.model(x, all=True)
        
        att_mat = torch.stack(att_mat).squeeze(1)
        att_mat = torch.mean(att_mat, dim=1)
        #'''
        # To account for residual connections, we add an identity matrix to the
        # attention matrix and re-normalize the weights.
        residual_att = torch.eye(att_mat.size(1)).cuda()
        aug_att_mat = att_mat + residual_att
        aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

        # Recursively multiply the weight matrices
        joint_attentions = torch.zeros(aug_att_mat.size()).cuda()
        joint_attentions[0] = aug_att_mat[0]

        for n in range(1, aug_att_mat.size(0)):
            joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])
        
        # Attention from the output token to the input space.
        joint_attentions_ = torch.mean(joint_attentions, dim=0)
        #v = joint_attentions[-1]
        v = joint_attentions_
        grid_size = int(np.sqrt(aug_att_mat.size(-1)))
        mask = v[0, 1:].reshape(grid_size, grid_size).detach().cpu().numpy()
        #'''
        return x, mask
    
    def forward_test(self, x, path):
        batch_size, _, h, w = x.shape # [batch_size, 2, h, w]
        t = transforms.ToPILImage()
        x1 = x[:, 0, :, :].unsqueeze(1)
        x2 = x[:, 1, :, :].unsqueeze(1)

        out1, mask_1 = self.forward_one_test(x1)
        out2, mask_2 = self.forward_one_test(x2)
        #im_1 = Image.open(path['Anchor'][0])
        im_1 = t(x[:, 0, :, :])
        mask_1 = cv2.resize(mask_1 / mask_1.max(), im_1.size, interpolation=cv2.INTER_AREA)
        #im_2 = Image.open(path['ref'][0])
        im_2 = t(x[:, 1, :, :])
        mask_2 = cv2.resize(mask_2 / mask_2.max(), im_2.size, interpolation=cv2.INTER_AREA)

        result_1 = (mask_1 * im_1).astype("uint8")
        #plt.imsave('att_result_0.png', result_1)
        result_2 = (mask_2 * im_2).astype("uint8")
        #plt.imsave('att_result_1.png', result_2)
        #plt.close()
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows= 2, ncols = 2, figsize=(16, 8))

        ax1.set_title('att_result_0')
        ax2.set_title('att_result_1')
        z1_plot = ax1.imshow(result_1)
        z2_plot = ax2.imshow(result_2)

        z3_plot = ax3.imshow(mask_1)
        z4_plot = ax4.imshow(mask_2)

        plt.colorbar(z1_plot, ax=ax1)
        plt.colorbar(z2_plot, ax=ax2)
        plt.colorbar(z3_plot, ax=ax3)
        plt.colorbar(z4_plot, ax=ax4)
        plt.savefig('att_result.png')
        plt.close()

        out = self.pdist(out1[:, 0], out2[:, 0]) # class token
        return out
    
    def save_to_file(self, file_path: str) -> None:
        torch.save(self.state_dict(), file_path)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--norm", type=str, default='center')
    parser.add_argument("--metric", type=str, default='cosine')
    parser.add_argument('--solver', type=str, default='opencv')
    parser.add_argument('--temperature', type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device("cuda")

    x = torch.zeros([1, 2, 224, 224])
    #x = torch.zeros([8, 2, 172, 172])
    x = x.to(device)

    model = ViT_for_OSV_emd(args)
    #model = ViT_for_OSV_overlap()
    #model = ViT_for_OSV()
    model.cuda()
    out1 = model(x)

    print(out1.shape)