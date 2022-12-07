import torch
import torch.nn as nn
import torch.nn.functional as F
from .emd_utils import *

from matplotlib import pyplot as plt
from tqdm import tqdm
from PIL import Image
import numpy as np
from torchvision.transforms.functional import to_pil_image
from torchvision import transforms
from einops import rearrange

class DeepEMD(nn.Module):

    def __init__(self, args, mode='meta'):
        super().__init__()

        self.mode = mode
        self.args = args

        #self.encoder = ResNet(args=args)
        
        ### pool ###
        self.pool_pooling =  nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.pool_proj = nn.Linear(in_features=640, out_features=128)
        
    def forward(self, input):
        if self.mode == 'meta':
            support, query = input
            return self.emd_forward_1shot(support, query)
        
        elif self.mode == 'pool':
            support, query = input
            return self.pool_forward_1shot(support, query)

        elif self.mode == 'encoder':
            if self.args.deepemd == 'fcn':
                dense = True
            else:
                dense = False
            return self.encode(input, dense)
        else:
            raise ValueError('Unknown mode')

    def get_weight_vector(self, A, B):

        M = A.shape[0]
        N = B.shape[0]

        B = F.adaptive_avg_pool2d(B, [1, 1])
        B = B.repeat(1, 1, A.shape[2], A.shape[3])

        A = A.unsqueeze(1)
        B = B.unsqueeze(0)

        A = A.repeat(1, N, 1, 1, 1)
        B = B.repeat(M, 1, 1, 1, 1)

        combination = (A * B).sum(2)
        combination = combination.view(M, N, -1)
        combination = F.relu(combination) + 1e-3
        return combination

    def get_weight_vector_v2(self, x): # calulate marginal distribution by pixel value
        batch_size, _, h, w = x.shape # input images
        x1 = x[:, 0, :, :]
        x2 = x[:, 1, :, :]

        PATCH_SIZE = 16
        x1_patches = x1.unfold(1, PATCH_SIZE, PATCH_SIZE).unfold(2, PATCH_SIZE, PATCH_SIZE)
        x2_patches = x2.unfold(1, PATCH_SIZE, PATCH_SIZE).unfold(2, PATCH_SIZE, PATCH_SIZE)

        x1_patches = rearrange(x1_patches, 'b x y px py -> b (x y) (px py)').contiguous().sum(dim=-1)
        x2_patches = rearrange(x2_patches, 'b x y px py -> b (x y) (px py)').contiguous().sum(dim=-1)
        
        return (x1_patches / x1.sum()).unsqueeze(0), (x2_patches / x2.sum()).unsqueeze(0)
    
    def emd_forward_1shot(self, proto, query, x):
        proto = proto.squeeze(0)
        x = x.unsqueeze(0)

        if self.args.distribution == 'v1':
            weight_1 = self.get_weight_vector(query, proto)
            weight_2 = self.get_weight_vector(proto, query)
        elif self.args.distribution == 'v2':
            weight_1, weight_2 = self.get_weight_vector_v2(x)

        proto = self.normalize_feature(proto)
        query = self.normalize_feature(query)
        similarity_map = self.get_similiarity_map(proto, query)
        
        logits = self.get_emd_distance(similarity_map, weight_1, weight_2, solver='opencv') # opencv
        return logits
    
    def emd_forward_1shot_test(self, proto, query, x):
        x = x.unsqueeze(0)
        batch_size, _, h, w = x.shape # input images
        x1 = x[:, 0, :, :].detach().cpu()
        x2 = x[:, 1, :, :].detach().cpu()
        #print(x1.shape)
        #resize = transforms.Resize([224, 224])
        #x1 = resize(x1)
        #x2 = resize(x2)
        #print(x1.shape)
        x1 = x1.squeeze()
        x2 = x2.squeeze()
        '''
        PATCH_SIZE = 16
        STRIDE_SIZE = 16
        x1_patches = x1.unfold(0, PATCH_SIZE, STRIDE_SIZE).unfold(1, PATCH_SIZE, STRIDE_SIZE)
        x2_patches = x2.unfold(0, PATCH_SIZE, STRIDE_SIZE).unfold(1, PATCH_SIZE, STRIDE_SIZE)
        fig, ax = plt.subplots(28, 14, figsize=(8, 16))
        print("processing...")
        for i in range(14):
            for j in range(14):
                sub_img = x1_patches[i,j]
                ax[i][j].imshow(to_pil_image(sub_img))
                #ax[i][j].set_title('{}'.format(i * 14 + j))
                ax[i][j].axis('off')

                sub_img = x2_patches[i,j]
                ax[i + 14][j].imshow(to_pil_image(sub_img))
                #ax[i + 14][j].set_title('{}'.format(i * 14 + j))
                ax[i + 14][j].axis('off')
        plt.savefig('patches.png')
        plt.close()
        '''
        x1 = Image.fromarray(np.uint8(255 - x1.numpy()*255))
        x2 = Image.fromarray(np.uint8(255 - x2.numpy()*255))
        proto = proto.squeeze(0)

        weight_1 = self.get_weight_vector(query, proto)
        weight_2 = self.get_weight_vector(proto, query)

        #weight_1, weight_2 = self.get_weight_vector_v2(x)

        w1_ = weight_1.squeeze().detach().cpu().numpy()
        w1 = w1_.reshape(14, 14)
        w1 = cv2.resize(w1 / w1.max(), x1.size, interpolation=cv2.INTER_AREA) # interpolation=cv2.INTER_AREA
        result_1 = (w1 * x1).astype("uint8")

        w2_ = weight_2.squeeze().detach().cpu().numpy()
        w2 = w2_.reshape(14, 14)
        w2 = cv2.resize(w2 / w2.max(), x2.size, interpolation=cv2.INTER_AREA)
        result_2 = (w2 * x2).astype("uint8")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows= 2, ncols = 2, figsize=(16, 8))

        ax1.set_title('att_result_0')
        ax2.set_title('att_result_1')
        z1_plot = ax1.imshow(result_1)
        z2_plot = ax2.imshow(result_2)
        #z3_plot = ax3.imshow(w1_)
        #z4_plot = ax4.imshow(w2_)
        z3_plot = ax3.bar(np.arange(14 * 14), w1_)
        z4_plot = ax4.bar(np.arange(14 * 14), w2_)

        plt.colorbar(z1_plot, ax=ax1)
        plt.colorbar(z2_plot, ax=ax2)
        #plt.colorbar(z3_plot, ax=ax3)
        #plt.colorbar(z4_plot, ax=ax4)
        plt.savefig('att_result.png')
        plt.close()
        '''
        w1 = weight_1.squeeze().detach().cpu().numpy().reshape(14, 14)
        w2 = weight_2.squeeze().detach().cpu().numpy().reshape(14, 14)
        plt.imshow(w1)
        plt.colorbar()
        plt.savefig("weight_1.png")
        plt.close()
        plt.imshow(w2)
        plt.colorbar()
        plt.savefig("weight_2.png")
        plt.close()
        '''

        proto = self.normalize_feature(proto)
        query = self.normalize_feature(query)
        similarity_map = self.get_similiarity_map(proto, query)
        '''
        s = similarity_map.squeeze().detach().cpu().numpy()
        for idx, item in enumerate(s):
            item = item.reshape(14, 14)
            plt.imshow(item)
            plt.colorbar()
            plt.savefig("similarity_maps/{}.png".format(idx))
            plt.close()
        plt.imshow(s)
        plt.colorbar()
        plt.savefig("similarity_maps/similarity_map.png")
        plt.close()
        '''
        logits = self.get_emd_distance(similarity_map, weight_1, weight_2, solver='opencv') #opencv
        return logits

    def get_sfc(self, support):
        support = support.squeeze(0)
        # init the proto
        SFC = support.view(self.args.shot, -1, 640, support.shape[-2], support.shape[-1]).mean(dim=0).clone().detach()
        SFC = nn.Parameter(SFC.detach(), requires_grad=True)

        optimizer = torch.optim.SGD([SFC], lr=self.args.sfc_lr, momentum=0.9, dampening=0.9, weight_decay=0)

        # crate label for finetune
        label_shot = torch.arange(self.args.way).repeat(self.args.shot)
        label_shot = label_shot.type(torch.cuda.LongTensor)

        with torch.enable_grad():
            for k in range(0, self.args.sfc_update_step):
                rand_id = torch.randperm(self.args.way * self.args.shot).cuda()
                for j in range(0, self.args.way * self.args.shot, self.args.sfc_bs):
                    selected_id = rand_id[j: min(j + self.args.sfc_bs, self.args.way * self.args.shot)]
                    batch_shot = support[selected_id, :]
                    batch_label = label_shot[selected_id]
                    optimizer.zero_grad()
                    logits = self.emd_forward_1shot(SFC, batch_shot.detach())
                    loss = F.cross_entropy(logits, batch_label)
                    loss.backward()
                    optimizer.step()
        return SFC

    def get_emd_distance(self, similarity_map, weight_1, weight_2, solver='opencv'):
        num_query = similarity_map.shape[0]
        num_proto = similarity_map.shape[1]
        num_node=weight_1.shape[-1]

        if solver == 'opencv':  # use openCV solver

            for i in range(num_query):
                for j in range(num_proto):
                    _, flow = emd_inference_opencv(1 - similarity_map[i, j, :, :], weight_1[i, j, :], weight_2[j, i, :])

                    similarity_map[i, j, :, :] =(similarity_map[i, j, :, :])*torch.from_numpy(flow).cuda()
                    '''
                    s = similarity_map.squeeze().detach().cpu().numpy()
                    for idx, item in enumerate(s):
                        item = item.reshape(14, 14)
                        plt.imshow(item)
                        plt.colorbar()
                        plt.savefig("similarity_maps/{}.png".format(idx))
                        plt.close()
                    plt.imshow(s)
                    plt.colorbar()
                    plt.savefig("similarity_maps/similarity_map.png")
                    plt.close()
                    '''

            temperature=(self.args.temperature/num_node)
            #print(temperature)
            logitis = similarity_map.sum(-1).sum(-1) *  temperature
            ### similarity to distance ###
            return 1 - logitis

        elif solver == 'sinkhorn':

            for i in range(num_query):
                for j in range(num_proto):
                    _, flow = emd_inference_opencv(1 - similarity_map[i, j, :, :], weight_1[i, j, :], weight_2[j, i, :])

                    similarity_map[i, j, :, :] =(similarity_map[i, j, :, :])*torch.from_numpy(flow).cuda()

            temperature=(self.args.temperature/num_node)
            logitis = similarity_map.sum(-1).sum(-1) *  temperature
            ### similarity to distance ###
            return 1 - logitis
        
        elif solver == 'qpth':
            print("Warning: use qpth")
            weight_2 = weight_2.permute(1, 0, 2)
            similarity_map = similarity_map.view(num_query * num_proto, similarity_map.shape[-2],
                                                 similarity_map.shape[-1])
            weight_1 = weight_1.view(num_query * num_proto, weight_1.shape[-1])
            weight_2 = weight_2.reshape(num_query * num_proto, weight_2.shape[-1])

            _, flows = emd_inference_qpth(1 - similarity_map, weight_1, weight_2,form=self.args.form, l2_strength=self.args.l2_strength)

            logitis=(flows*similarity_map).view(num_query, num_proto,flows.shape[-2],flows.shape[-1])
            temperature = (self.args.temperature / num_node)
            logitis = logitis.sum(-1).sum(-1) *  temperature
        else:
            raise ValueError('Unknown Solver')

        return logitis

    def normalize_feature(self, x):
        if self.args.norm == 'center':
            x = x - x.mean(1).unsqueeze(1)
            return x
        else:
            return x


    def get_similiarity_map(self, proto, query):
        way = proto.shape[0]
        num_query = query.shape[0]
        query = query.view(query.shape[0], query.shape[1], -1)
        proto = proto.view(proto.shape[0], proto.shape[1], -1)

        proto = proto.unsqueeze(0).repeat([num_query, 1, 1, 1])
        query = query.unsqueeze(1).repeat([1, way, 1, 1])
        proto = proto.permute(0, 1, 3, 2)
        query = query.permute(0, 1, 3, 2)
        feature_size = proto.shape[-2]

        if self.args.metric == 'cosine':
            proto = proto.unsqueeze(-3)
            query = query.unsqueeze(-2)
            query = query.repeat(1, 1, 1, feature_size, 1)
            similarity_map = F.cosine_similarity(proto, query, dim=-1)
        if self.args.metric == 'l2':
            proto = proto.unsqueeze(-3)
            query = query.unsqueeze(-2)
            query = query.repeat(1, 1, 1, feature_size, 1)
            similarity_map = (proto - query).pow(2).sum(-1)
            similarity_map = 1 - similarity_map

        return similarity_map

    def encode(self, x, dense=True):

        if x.shape.__len__() == 5:  # batch of image patches
            num_data, num_patch = x.shape[:2]
            x = x.reshape(-1, x.shape[2], x.shape[3], x.shape[4])
            x = self.encoder(x)
            x = F.adaptive_avg_pool2d(x, 1)
            x = x.reshape(num_data, num_patch, x.shape[1], x.shape[2], x.shape[3])
            x = x.permute(0, 2, 1, 3, 4)
            x = x.squeeze(-1)
            return x

        else:
            x = self.encoder(x)
            if dense == False:
                x = F.adaptive_avg_pool2d(x, 1)
                return x
            if self.args.feature_pyramid is not None:
                x = self.build_feature_pyramid(x)
        return x

    def build_feature_pyramid(self, feature):
        feature_list = []
        for size in self.args.feature_pyramid:
            feature_list.append(F.adaptive_avg_pool2d(feature, size).view(feature.shape[0], feature.shape[1], 1, -1))
        feature_list.append(feature.view(feature.shape[0], feature.shape[1], 1, -1))
        out = torch.cat(feature_list, dim=-1)
        return out
    
    def pool_forward_1shot(self, proto, query):
        pdist = nn.PairwiseDistance(p=2, keepdim = True)
        
        proto_ = torch.flatten(self.pool_pooling(proto), 1)
        query_ = torch.flatten(self.pool_pooling(query), 1)
        
        out = pdist(proto_, query_)
        return out