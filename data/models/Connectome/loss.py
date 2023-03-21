import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h != ht and w != wt:  # upsample labels
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    loss = F.cross_entropy(
        input, target, weight=weight, size_average=size_average, ignore_index=250
    )
    return loss


def multi_scale_cross_entropy2d(input, target, weight=None, size_average=True, scale_weight=None):
    if not isinstance(input, tuple):
        return cross_entropy2d(input=input, target=target, weight=weight, size_average=size_average)

    # Auxiliary training for PSPNet [1.0, 0.4] and ICNet [1.0, 0.4, 0.16]
    if scale_weight is None:  # scale_weight: torch tensor type
        n_inp = len(input)
        scale = 0.4
        scale_weight = torch.pow(scale * torch.ones(n_inp), torch.arange(n_inp).float()).to(
            target.device
        )

    loss = 0.0
    for i, inp in enumerate(input):
        loss = loss + scale_weight[i] * cross_entropy2d(
            input=inp, target=target, weight=weight, size_average=size_average
        )

    return loss


def bootstrapped_cross_entropy2d(input, target, K, weight=None, size_average=True):

    batch_size = input.size()[0]

    def _bootstrap_xentropy_single(input, target, K, weight=None, size_average=True):

        n, c, h, w = input.size()
        input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        target = target.view(-1)
        loss = F.cross_entropy(
            input, target, weight=weight, reduce=False, size_average=False, ignore_index=250
        )

        topk_loss, _ = loss.topk(K)
        reduced_topk_loss = topk_loss.sum() / K

        return reduced_topk_loss

    loss = 0.0
    # Bootstrap from each image not entire batch
    for i in range(batch_size):
        loss += _bootstrap_xentropy_single(
            input=torch.unsqueeze(input[i], 0),
            target=torch.unsqueeze(target[i], 0),
            K=K,
            weight=weight,
            size_average=size_average,
        )
    return loss / float(batch_size)


class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG16FeatureExtractor, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        # print(vgg16.features)

        # RELU1_1
        self.enc_1 = nn.Sequential(*list(vgg16.features)[:2])
        # RELU2_1
        self.enc_2 = nn.Sequential(*list(vgg16.features)[2:7])
        # RELU2_1
        self.enc_3 = nn.Sequential(*list(vgg16.features)[7:12])
        # RELU2_1
        self.enc_4 = nn.Sequential(*list(vgg16.features)[12:19])
        # RELU2_1
        self.enc_5 = nn.Sequential(*list(vgg16.features)[19:26])

        # fix the encoder
        for i in range(1):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(1):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))

        return results[1:]


class L1Loss(nn.Module):
    def __init__(self, weight=1):
        super(L1Loss, self).__init__()
        self.weight = weight
        self.loss = torch.nn.L1Loss()

    def forward(self, pred, gt):
        return self.weight * self.loss(pred, gt)


class PercStyleLoss(nn.Module):
    def __init__(self, device, perc_weight=1, style_weight=1):
        super(PercStyleLoss, self).__init__()
        self.extractor = VGG16FeatureExtractor().to(device)
        self.perc_weight = perc_weight
        self.style_weight = style_weight

    def gram(self, input):
        a, b, c, d = input.size()
        features = input.view(a, b, -1)
        G = torch.bmm(features, features.permute(0, 2, 1))
        return G.div(b * c * d)

    # def gram(self, feat):
    #     a, b, c, d = feat.size()
    #     features = feat.view(a, -1)
    #     G = torch.mm(features, features.t())
    #     return G.div(b * c * d)

    def forward(self, pred, gt):
        feat_gt = self.extractor(gt)
        feat_pred = self.extractor(pred)
        L_perc = 0
        L_style = 0
        for layer in range(len(feat_gt)):
            L_perc += self.perc_weight * torch.mean(torch.abs(feat_pred[layer] - feat_gt[layer]))
            L_style = self.style_weight * torch.mean(torch.abs(self.gram(feat_pred[layer]) - self.gram(feat_gt[layer])))
        return L_perc, L_style


class SNDisLoss(nn.Module):
    """
    The loss for sngan discriminator
    """
    def __init__(self, weight=1):
        super(SNDisLoss, self).__init__()
        self.weight = weight

    def forward(self, pos, neg):
        #return self.weight * (torch.sum(F.relu(-1+pos)) + torch.sum(F.relu(-1-neg)))/pos.size(0)
        return self.weight * (torch.mean(F.relu(1.-pos)) + torch.mean(F.relu(1.+neg)))


# class SNGenLoss(nn.Module):
#     """
#     The loss for sngan generator
#     """
#     def __init__(self, weight=1):
#         super(SNGenLoss, self).__init__()
#         self.weight = weight
#
#     def forward(self, neg):
#         return - self.weight * torch.mean(neg)


# class RegionNCELoss(nn.Module):
#     def __init__(self, opt):
#         super(RegionNCELoss, self).__init__()
#         self.opt = opt
#         self.softmax = nn.Softmax(dim=1) # dim = 1?
#         self.CE_loss = torch.nn.CrossEntropyLoss(reduction='none')
#
#     def forward(self, feat_q, feat_k):
#         # feat_q B * S, feat_k B * 24 * S
#         feat_q = feat_q.unsqueeze(1)
#         # feat_q B * 1 * S
#
#         qk = (feat_q * feat_k).flatten(2,3)
#         # print('qk', qk.size())
#         # qk B * 24 * S
#
#         pred = -torch.log(self.softmax(qk)[:, 0, :])
#
#         return pred

# class RegionNCELoss(nn.Module):
#     def __init__(self, opt):
#         super(RegionNCELoss, self).__init__()
#         self.opt = opt
#         self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
#         self.mask_dtype = torch.uint8
#
#     def forward(self, feat_q, feat_k):
#         # (B * n_patch) * C
#         print('feat_q', feat_q.size())
#         batchSize = feat_q.shape[0]
#         dim = feat_q.shape[1]
#         feat_k = feat_k.detach()
#
#         # pos logit
#         # (B * n_patch) * 1 * C, (B * n_patch) * C * 1  -> (B * n_patch) * 1 * 1
#         l_pos = torch.bmm(feat_q.view(batchSize, 1, -1), feat_k.view(batchSize, -1, 1))
#         l_pos = l_pos.view(batchSize, 1)
#
#         # neg logit
#         batch_dim_for_bmm = self.opt.batch_size
#
#         # reshape features to batch size B * n_patch * C
#         feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
#         feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
#         npatches = feat_q.size(1)
#         # B * n_patch * C, B * C * n_patch -> B * n_patch * n_patch
#         l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))
#
#         # diagonal entries are similarity between same features, and hence meaningless.
#         # just fill the diagonal with very small number, which is exp(-10) and almost zero
#         diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
#         l_neg_curbatch.masked_fill_(diagonal, -10.0)
#         # (B * n_patch) * n_patch
#         l_neg = l_neg_curbatch.view(-1, npatches)
#
#         out = torch.cat((l_pos, l_neg), dim=1) / self.opt.nce_T
#
#         loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
#                                                         device=feat_q.device))
#         print(loss.size())
#
#         return loss

class PatchNCELoss(nn.Module):
    def __init__(self, opt):
        super(PatchNCELoss, self).__init__()
        self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8

    def forward(self, feat_q, feat_k):
        # (B * n_patch) * C
        # print('feat_q', feat_q.size())
        batchSize = feat_q.shape[0]
        dim = feat_q.shape[1]
        # feat_k = feat_k.detach()

        # pos logit
        # (B * n_patch) * 1 * C, (B * n_patch) * C * 1  -> (B * n_patch) * 1 * 1
        l_pos = torch.bmm(feat_q.view(batchSize, 1, -1), feat_k.view(batchSize, -1, 1))
        l_pos = l_pos.view(batchSize, 1)

        # neg logit
        batch_dim_for_bmm = self.opt.batch_size

        # reshape features to batch size B * n_patch * C
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        # B * n_patch * C, B * C * n_patch -> B * n_patch * n_patch
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        # (B * n_patch) * n_patch
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.opt.nce_T

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))
        # print(loss.size())

        return loss

class RegionNCELoss(nn.Module):
    def __init__(self, opt):
        super(RegionNCELoss, self).__init__()
        self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8


    def forward(self, feat_q, feat_k):
        # feat_q: B * 1 * C, feat_k, B * n_patch * C
        # print('feat_q', feat_q.size(), 'feat_k', feat_k.size())
        B = self.opt.batch_size

        # logits: B * n_patch * 1
        logits = torch.bmm(feat_k, feat_q.transpose(2, 1))
        # logits: B * n_patch
        logits = logits.view(B, -1)

        out = logits / self.opt.nce_T

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))
        print(loss.size())

        return loss


class TriConsistentLoss(nn.Module):
    def __init__(self, tri_weight=0.1):
        super(TriConsistentLoss, self).__init__()
        self.lambda_tri = tri_weight

    def gram(self, input):
        a, b, c, d = input.size()
        features = input.view(a, b, -1)
        G = torch.bmm(features, features.permute(0, 2, 1))
        return G.div(b * c * d)

    def forward(self, feat_masked, feat_inpaint, feat_gt):
        tri_loss = 0.0
        for f_m, f_i, f_g in zip(feat_masked, feat_inpaint, feat_gt):
            l_mi = torch.mean(torch.abs(f_m - f_i))
            l_mg = torch.mean(torch.abs(f_m - f_g))
            l_ig = torch.mean(torch.abs(f_i - f_g))

            l_perc = (l_mi + l_mg + l_ig)/3

            l_gram_mi = torch.mean(torch.abs(self.gram(f_m) - self.gram(f_i)))
            l_gram_mg = torch.mean(torch.abs(self.gram(f_m) - self.gram(f_g)))
            l_gram_ig = torch.mean(torch.abs(self.gram(f_i) - self.gram(f_g)))

            l_style = 100 * (l_gram_mi + l_gram_ig + l_gram_mg) / 3

            tri_loss = l_perc + l_style

        return self.lambda_tri * tri_loss


def set_requires_grad(self, nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad