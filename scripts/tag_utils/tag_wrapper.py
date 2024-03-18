import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from open_clip.transformer import VisionTransformer
from torch_kmeans import KMeans

from .gem_utils import SelfSelfAttention, GEMResidualBlock, modified_vit_forward
from scripts.dino_kmeans import dino_kmeans, extract_clip_features
from scripts.crf import batched_crf
from scripts.retrieve_utils.src.models.components.vocabularies import RetrievalVocabulary
from scripts.retrieve_utils.src.models.cased import CaSED


class TAGWrapper(nn.Module):
    def __init__(self, model, tokenizer, depth=7, ss_attn_iter=1, ss_attn_temp=None, classes=None, model_name="ViT-L/14", pretrained="openai", database_name="ViT-L-14_PMD_TOP5", n_clusters=15, vocab_transform=None, num_samples=10):
        super(TAGWrapper, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.depth = depth
        self.ss_attn_iter = ss_attn_iter
        self.ss_attn_temp = ss_attn_temp
        self.patch_size = self.model.visual.patch_size[0]
        self.apply_gem()
        if classes is not None:
            self.text_embeddings = self.pre_encode_text(classes)
        else:
            self.text_embeddings = None

        self.n_clusters = n_clusters
        self.k_means = KMeans(n_clusters=self.n_clusters, verbose=False)
        self.vocabulary = RetrievalVocabulary(database_name=database_name, databases_dict_fp="scripts/retrieve_utils/artifacts/models/retrieval/databases.json", num_samples = num_samples).cuda()
        self.cased = CaSED(vocabulary=self.vocabulary, model_name=model_name, pretrained=pretrained, vocab_transform=vocab_transform).cuda()

    def apply_gem(self):
        for i in range(1, self.depth):
            # Extract info from the original ViT
            num_heads = self.model.visual.transformer.resblocks[-i].attn.num_heads
            dim = int(self.model.visual.transformer.resblocks[-i].attn.head_dim * num_heads)
            qkv_bias = True
            # Init the self-self attention layer
            ss_attn = SelfSelfAttention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                        ss_attn_iter=self.ss_attn_iter, ss_attn_temp=self.ss_attn_temp)
            # Copy necessary weights
            ss_attn.qkv.weight.data = self.model.visual.transformer.resblocks[-i].attn.in_proj_weight.clone()
            ss_attn.qkv.bias.data = self.model.visual.transformer.resblocks[-i].attn.in_proj_bias.clone()
            ss_attn.proj.weight.data = self.model.visual.transformer.resblocks[-i].attn.out_proj.weight.clone()
            ss_attn.proj.bias.data = self.model.visual.transformer.resblocks[-i].attn.out_proj.bias.clone()
            # Swap the original Attention with our SelfSelfAttention
            self.model.visual.transformer.resblocks[-i].attn = ss_attn
            # Wrap Residual block to handle SelfSelfAttention outputs
            self.model.visual.transformer.resblocks[-i] = GEMResidualBlock(self.model.visual.transformer.resblocks[-i])
        # Modify ViT's forward function
        self.model.visual.forward = modified_vit_forward.__get__(self.model.visual, VisionTransformer)
        return

    def encode_text(self, text: list):
        prompts = [f'a photo of a {cls}.' for cls in text]
        tokenized_prompts = self.tokenizer(prompts).to(self.model.visual.proj.device)
        text_embedding = self.model.encode_text(tokenized_prompts)
        text_embedding = F.normalize(text_embedding, dim=-1)
        return text_embedding.unsqueeze(0)

    @torch.no_grad()
    def pre_encode_text(self, text: list):
        templates = ['a bad photo of a {}.', 'a photo of many {}.', 'a sculpture of a {}.', 'a photo of the hard to see {}.', 'a low resolution photo of the {}.', 'a rendering of a {}.', 'graffiti of a {}.', 'a bad photo of the {}.', 'a cropped photo of the {}.', 'a tattoo of a {}.', 'the embroidered {}.', 'a photo of a hard to see {}.', 'a bright photo of a {}.', 'a photo of a clean {}.', 'a photo of a dirty {}.', 'a dark photo of the {}.', 'a drawing of a {}.', 'a photo of my {}.', 'the plastic {}.', 'a photo of the cool {}.', 'a close-up photo of a {}.', 'a black and white photo of the {}.', 'a painting of the {}.', 'a painting of a {}.', 'a pixelated photo of the {}.', 'a sculpture of the {}.', 'a bright photo of the {}.', 'a cropped photo of a {}.', 'a plastic {}.', 'a photo of the dirty {}.', 'a jpeg corrupted photo of a {}.', 'a blurry photo of the {}.', 'a photo of the {}.', 'a good photo of the {}.', 'a rendering of the {}.', 'a {} in a video game.', 'a photo of one {}.', 'a doodle of a {}.', 'a close-up photo of the {}.', 'a photo of a {}.', 'the origami {}.', 'the {} in a video game.', 'a sketch of a {}.', 'a doodle of the {}.', 'a origami {}.', 'a low resolution photo of a {}.', 'the toy {}.', 'a rendition of the {}.', 'a photo of the clean {}.', 'a photo of a large {}.', 'a rendition of a {}.', 'a photo of a nice {}.', 'a photo of a weird {}.', 'a blurry photo of a {}.', 'a cartoon {}.', 'art of a {}.', 'a sketch of the {}.', 'a embroidered {}.', 'a pixelated photo of a {}.', 'itap of the {}.', 'a jpeg corrupted photo of the {}.', 'a good photo of a {}.', 'a plushie {}.', 'a photo of the nice {}.', 'a photo of the small {}.', 'a photo of the weird {}.', 'the cartoon {}.', 'art of the {}.', 'a drawing of the {}.', 'a photo of the large {}.', 'a black and white photo of a {}.', 'the plushie {}.', 'a dark photo of a {}.', 'itap of a {}.', 'graffiti of the {}.', 'a toy {}.', 'itap of my {}.', 'a photo of a cool {}.', 'a photo of a small {}.', 'a tattoo of the {}.', 'there is a {} in the scene.', 'there is the {} in the scene.', 'this is a {} in the scene.', 'this is the {} in the scene.', 'this is one {} in the scene.',]
        zeroshot_weights = []
        for cls in text:
            prompt = [template.format(cls) for template in templates]
            tokenized_prompt = self.tokenizer(prompt).to(self.model.visual.proj.device)
            text_embedding = self.model.encode_text(tokenized_prompt)
            text_embedding = F.normalize(text_embedding, dim=-1)
            text_embedding = text_embedding.mean(dim=0)
            text_embedding = F.normalize(text_embedding, dim=-1)
            zeroshot_weights.append(text_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
        return zeroshot_weights.permute((1, 0)).unsqueeze(0)

    def min_max(self, logits):
        B, num_prompt = logits.shape[:2]
        logits_min = logits.reshape(B, num_prompt, -1).min(dim=-1, keepdim=True)[0].unsqueeze(-1)
        logits_max = logits.reshape(B, num_prompt, -1).max(dim=-1, keepdim=True)[0].unsqueeze(-1)
        logits = (logits - logits_min) / (logits_max - logits_min)
        return logits

    def forward(self, image: torch.Tensor, text: list, normalize: bool = True, return_ori: bool =False):
        """
        :param image: torch.Tensor [1, 3, H, W]
        :param text: list[]
        :param normalize: bool - if True performs min-max normalization
        :param return_ori: bool - if True uses the features from the original visual encoder
        """
        # Image
        W, H = image.shape[-2:]
        feat_gem, feat_ori = self.model.visual(image)
        image_feat = feat_ori if return_ori else feat_gem
        image_feat = F.normalize(image_feat, dim=-1)  # [1, N, dim]

        # Text
        text_embeddings = self.encode_text(text)  # [1, num_prompt, dim]

        # Image-Text matching
        img_txt_matching = image_feat[:, 1:] @ text_embeddings.transpose(-1, -2)  # [1, N, num_prompt]
        img_txt_matching = rearrange(img_txt_matching, 'b (w h) c -> b c w h',
                                     w=W//self.patch_size, h=H//self.patch_size)  # [1, num_prompt, w, h]

        # Interpolate
        img_txt_matching = F.interpolate(img_txt_matching, size=(W, H), mode='bilinear')  # [1, num_prompt, W, H]

        # Heat Maps
        if normalize:
            img_txt_matching = self.min_max(img_txt_matching)
        return img_txt_matching

    def batched_forward(self, image: torch.Tensor, text: list, normalize: bool = True, return_ori: bool =False):
        """
        :param image: torch.Tensor [B, 3, H, W]
        :param text: list[list[]]
        :param normalize: bool - if True performs min-max normalization
        :param return_ori: bool - if True uses the features from the original visual encoder
        """
        B, _, W, H = image.shape
        L = len(text)
        if isinstance(text[0], list):
            cumm_idx = np.cumsum([len(t) for t in text]).tolist()
        elif isinstance(text[0], str):
            text = [text for i in range(B)]
            cumm_idx = np.cumsum([len(t) for t in text]).tolist()
        # assert B == L, f'Number of prompts L: {L} should be the same as number of images B: {B}.'

        # Image
        # torch.Size([4, 785, 512]) torch.Size([4, 785, 512])
        feat_gem, feat_ori = self.model.visual(image)
        image_feat = feat_ori if return_ori else feat_gem
        image_feat = F.normalize(image_feat, dim=-1)  # [B, N, dim]

        # Text
        # torch.Size([1, 108, 512])
        # Text
        if self.text_embeddings is None:
            flatten_text = [t for sub_text in text for t in sub_text]
            text_embeddings = self.encode_text(flatten_text)  # [B, num_prompt, dim]
        else:
            _, num_prompt, dim = self.text_embeddings.shape
            text_embeddings = self.text_embeddings.expand(B, num_prompt, dim)

        # Image-Text matching
        # torch.Size([4, 784, 108])
        img_txt_matching = 100 * image_feat[:, 1:] @ text_embeddings.transpose(-1, -2)  # [B, N, num_prompt]
        # torch.Size([4, 108, 28, 28])
        img_txt_matching = rearrange(img_txt_matching, 'b (w h) c -> b c w h',
                                     w=W // self.patch_size, h=H // self.patch_size)  # [B, num_prompt, w, h]

        # Interpolate
        img_txt_matching = F.interpolate(img_txt_matching, size=(W, H), mode='bilinear')  # [B,num_prompt, W, H]

        # Heat Maps
        if normalize:
            img_txt_matching = self.min_max(img_txt_matching)  # [B,num_prompt, W, H]

        return img_txt_matching
    
        # unflatten
        img_txt_matching = torch.tensor_split(img_txt_matching, cumm_idx[:-1], dim=1)
        img_txt_matching = [itm[i] for i, itm in enumerate(img_txt_matching)]
        return img_txt_matching
    
    def batched_forward_cased(self, image: torch.Tensor, normalize: bool = True, return_ori: bool =False, use_crf: bool = False):
        """
        :param image: torch.Tensor [B, 3, H, W]
        :param text: list[list[]]
        :param normalize: bool - if True performs min-max normalization
        :param return_ori: bool - if True uses the features from the original visual encoder
        """
        B, _, W, H = image.shape

        # Image
        # torch.Size([4, 785, 512]) torch.Size([4, 785, 512])
        feat_gem, feat_ori = self.model.visual(image)
        image_feat = feat_ori if return_ori else feat_gem
        # image_feat_l = self.model.visual(F.interpolate(image, size=(336, 336), mode='bilinear'))[0]
        # image_feat_l = rearrange(image_feat_l[:, 1:], 'b (w h) c -> b c w h', w=336 // self.patch_size, h=336 // self.patch_size)
        # image_feat_l = F.interpolate(image_feat_l, size=(W//self.patch_size, H//self.patch_size), mode='bilinear')
        # image_feat_l = torch.cat([image_feat[:, 0:1, :], rearrange(image_feat_l, 'b c w h -> b (w h) c')], dim=1)
        # image_feat = (image_feat + image_feat_l) / 2
        
        # image_feat = F.normalize(image_feat, dim=-1)  # [B, N, dim]
        dim = image_feat.shape[-1]
        # result = self.k_means(image_feat[:, 1:, :])
        # # torch.Size([8, 257])
        # # torch.Size([8, 10, 768])
        # images_label = result.labels
        # images_z = result.centers
        # images_z = images_z.reshape(-1, dim)
        
        dino_logits = dino_kmeans(image, self.k_means)
        # torch.Size([8, 15, 768]), torch.Size([8, 768, 448, 448])
        if use_crf:
            dino_logits = batched_crf(image, dino_logits).cuda()
        images_z, clip_features = extract_clip_features(image_feat, dino_logits)
        images_z = images_z.reshape(-1, dim)
        
        images_vocab = self.vocabulary(images_z=images_z)
        # print(images_vocab)
        # images_vocab = self.cased.batch_step(images_z, images_vocab, True)
        images_p, words, images_vocab = self.cased.batch_step(images_z, images_vocab)
        text = [list(set(sum(images_vocab[batch*self.n_clusters: (batch+1)*self.n_clusters], []))) for batch in range(B)]
        # street_map = torch.tensor([i=="street" for i in words])
        # images_p[:, street_map] = 0
        images_vocab = [words[i] for i in images_p.argmax(dim=1)]
        images_vocab = [images_vocab[batch*self.n_clusters: (batch+1)*self.n_clusters] for batch in range(B)]
        # images_vocab = [[vocab[0] for vocab in images_vocab[batch*self.n_clusters: (batch+1)*self.n_clusters]] for batch in range(B)]
        # print(images_vocab)
        clip_logits = torch.einsum("bnf,bfhw->bnhw", images_z.reshape(B, -1, dim), clip_features)
        return clip_logits, images_vocab, dino_logits

        L = len(text)
        if isinstance(text[0], list):
            cumm_idx = np.cumsum([len(t) for t in text]).tolist()
        elif isinstance(text[0], str):
            text = [text for i in range(B)]
            cumm_idx = np.cumsum([len(t) for t in text]).tolist()

        # Text
        # torch.Size([1, 108, 512])
        # Text
        if self.text_embeddings is None:
            flatten_text = [t for sub_text in text for t in sub_text]
            text_embeddings = self.encode_text(flatten_text)  # [B, num_prompt, dim]
        else:
            _, num_prompt, dim = self.text_embeddings.shape
            text_embeddings = self.text_embeddings.expand(B, num_prompt, dim)

        # Image-Text matching
        # torch.Size([4, 784, 108])
        img_txt_matching = 100 * image_feat[:, 1:] @ text_embeddings.transpose(-1, -2)  # [B, N, num_prompt]
        # torch.Size([4, 108, 28, 28])
        img_txt_matching = rearrange(img_txt_matching, 'b (w h) c -> b c w h',
                                     w=W // self.patch_size, h=H // self.patch_size)  # [B, num_prompt, w, h]

        # Interpolate
        img_txt_matching = F.interpolate(img_txt_matching, size=(W, H), mode='bilinear')  # [B,num_prompt, W, H]

        # Heat Maps
        if normalize:
            img_txt_matching = self.min_max(img_txt_matching)  # [B,num_prompt, W, H]

        # unflatten
        img_txt_matching = torch.tensor_split(img_txt_matching, cumm_idx[:-1], dim=1)
        img_txt_matching = [itm[i] for i, itm in enumerate(img_txt_matching)]
        return img_txt_matching, text
    
    def batched_forward_cased_crop(self, image: torch.Tensor, normalize: bool = True, return_ori: bool =False):
        """
        :param image: torch.Tensor [B, 3, H, W]
        :param text: list[list[]]
        :param normalize: bool - if True performs min-max normalization
        :param return_ori: bool - if True uses the features from the original visual encoder
        """
        low_image = self.create_low_images(image)
        _, images_vocab_l, dino_logits_l = self.batched_forward_cased(low_image)
        dino_logits = self.fix_dim(dino_logits_l)
        images_vocab = [sum(images_vocab_l[i*4: (i+1)*4], []) for i in range(len(images_vocab_l)//4)]
        return None, images_vocab, dino_logits
        
    
    def create_low_images(self, x):
        B, _, H, W = x.shape
        return torch.nn.functional.unfold(x, kernel_size=(H//2, W//2), stride=H//2).permute(0, 2, 1).contiguous().view(-1, 3, H//2, W//2)
    
    def fix_dim(self, x):
        B, _, H, W = x.shape
        x = rearrange(x, "(b n) c h w -> b n c h w", b = B//4, n = 4)
        x = torch.cat([torch.cat([x[:, 0, :, :, :], x[:, 1, :, :, :]], dim=3), torch.cat([x[:, 2, :, :, :], x[:, 3, :, :, :]], dim=3)], dim=2)
        return x
    
    def upsample_feat(self, feat, w, h):
        feat = rearrange(feat[:, 1:], 'b (w h) c -> b c w h', w=w // self.patch_size, h=h // self.patch_size)
        feat = F.interpolate(feat, size=(w, h), mode='bilinear')
        feat = rearrange(feat, 'b c w h -> b (w h) c')
        return feat
        
