import logging

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn.functional import cosine_similarity
from transformers import PreTrainedModel, AutoModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    SequenceClassifierOutput,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')

logger = logging.getLogger(__name__)


def concat_features(*features):
    return torch.cat(features, dim=0) if features[0] is not None else None


class QuadrupletLoss:
    def __init__(self, distance_function, margin=1.0):
        'A cosine distance margin quadruplet loss'
        self.margin = margin
        self.distance_function = distance_function

    def __call__(self, pos1, pos2, neg1, neg2):
        dist_pos = self.distance_function(pos1, pos2)
        dist_neg = self.distance_function(neg1, neg2)
        loss = torch.clamp_min(self.margin + dist_pos - dist_neg, 0)
        return loss.mean()


class Pooler(nn.Module):

    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type

    def forward(self, attention_mask, outputs, outputs_is_last_hidden_state=False):
        if outputs_is_last_hidden_state:
            if self.pooler_type not in ['avg', 'avg-wo-cls']:
                raise NotImplementedError

            last_hidden = outputs

        else:
            last_hidden = outputs.last_hidden_state
            pooler_output = outputs.pooler_output
            hidden_states = outputs.hidden_states

        if self.pooler_type == 'cls':
            return pooler_output
        elif self.pooler_type == 'cls_before_pooler':
            return last_hidden[:, 0]
        elif self.pooler_type == 'avg':
            return (last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
        elif self.pooler_type == 'avg-wo-cls':
            return (last_hidden[:, 1:] * attention_mask[:, 1:].unsqueeze(-1)).sum(1) / attention_mask[:, 1:].sum(-1).unsqueeze(-1)
        elif self.pooler_type == 'avg_first_last':
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == 'avg_top2':
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError


def init_cls_set_kwargs(cls, kwargs):
    cls.model_args = kwargs['model_args']
    cls.mask_token_id = kwargs['mask_token_id']


class CrossEncoderForClassification(PreTrainedModel):  # FIXME Not-implemented: pooler_type == hypernet
    """Encoder model with backbone and classification head."""

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.backbone = AutoModel.from_pretrained(
            config.model_name_or_path,
            from_tf=bool('.ckpt' in config.model_name_or_path),
            config=config,
            cache_dir=config.cache_dir,
            revision=config.model_revision,
            use_auth_token=True if config.use_auth_token else None,
            add_pooling_layer=False,
        ).base_model
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        if config.transform:
            self.transform = nn.Sequential(
                nn.Dropout(classifier_dropout),
                nn.Linear(config.hidden_size, config.hidden_size),
                ACT2FN[config.hidden_act],
            )
        else:
            self.transform = None
        self.classifier = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.num_labels),
        )
        self.pooler = Pooler(config.pooler_type)
        if config.pooler_type in {'avg_first_last', 'avg_top2'}:
            self.output_hidden_states = True
        else:
            self.output_hidden_states = False
        if config.num_labels == 1:
            self.reshape_function = lambda x: x.reshape(-1)
            if config.objective == 'mse':
                self.loss_fct_cls = nn.MSELoss
            elif config.objective in {'triplet', 'triplet_mse'}:
                raise NotImplementedError('Triplet loss is not implemented for CrossEncoderForClassification')
            else:
                raise ValueError(f'Only regression and triplet objectives are supported for CrossEncoderForClassification with num_labels=1. Got {config.objective}.')
        else:
            assert config.objective == 'classification'
            self.reshape_function = lambda x: x.reshape(-1, config.num_labels)
            self.loss_fct_cls = nn.CrossEntropyLoss
        self.post_init()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            **kwargs,
    ):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=self.output_hidden_states,
        )
        features = self.pooler(attention_mask, outputs)
        if self.transform is not None:
            features = self.transform(features)
        logits = self.classifier(features)
        reshaped_logits = self.reshape_function(logits)
        loss = None
        if labels is not None:
            loss = self.loss_fct_cls()(reshaped_logits, labels.view(-1))
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class HyperLinear(nn.Module):
    """
    The Hyper Network module, for creating weight that creates sentence representation from model input
    """

    def __init__(self, hyper_net_base_enc_emb, backbone_enc_emb, sr_emb_size=768):
        super().__init__()
        self.in_dim = hyper_net_base_enc_emb
        self.hidden_dim = hyper_net_base_enc_emb
        self.out_common = hyper_net_base_enc_emb

        self.sr_dim_size = sr_emb_size

        self.common = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.out_common),
            nn.LeakyReLU(),
        )

        self.weight_head = nn.Linear(self.out_common, backbone_enc_emb * sr_emb_size)
        self.bias_head = nn.Linear(self.out_common, 1 * sr_emb_size)

    def forward(self, inputs):
        """
        Args: inputs: (bs, backbone_enc_emb)
        Returns: the weight (bs, backbone_enc_emb, sr_emb_size) and bias (bs, sr_emb_size) of the classifier
        """

        features = self.common(inputs)
        batch_size = inputs.shape[0]
        weight = self.weight_head(features)
        weight = weight.view(batch_size, -1, self.sr_dim_size)
        bias = self.bias_head(features)
        bias = bias.view(batch_size, self.sr_dim_size)
        return weight, bias


class BiEncoderForClassification(PreTrainedModel):  # Note, model half applied
    """Encoder model with backbone and classification head."""

    def __init__(self, config, **kwargs):
        super().__init__(config)
        init_cls_set_kwargs(self, kwargs)

        auto_model = AutoModel.from_pretrained(
            config.model_name_or_path,
            from_tf=bool('.ckpt' in config.model_name_or_path),
            config=config,
            cache_dir=config.cache_dir,
            revision=config.model_revision,
            use_auth_token=True if config.use_auth_token else None,
            add_pooling_layer=False,
        )
        self.backbone = auto_model.base_model

        classifier_dropout = (
            config.classifier_dropout if hasattr(config, "classifier_dropout") else config.hidden_dropout_prob
        )

        if config.transform:
            self.transform = nn.Sequential(
                nn.Dropout(classifier_dropout),
                nn.Linear(config.hidden_size, config.hidden_size),
                ACT2FN[config.hidden_act]
            )
        else:
            self.transform = None

        if self.model_args.do_hypernet:
            self.hyper = HyperLinear(config.hidden_size, config.hidden_size, config.hidden_size)
            self.hypernet_base_enc = AutoModel.from_pretrained(config.model_name_or_path)
            for param in self.hypernet_base_enc.parameters():
                param.requires_grad = False

        if config.pooler_type in {'avg_first_last', 'avg_top2'}:
            self.output_hidden_states = True
        else:
            self.output_hidden_states = False

        if config.objective == 'mse':
            self.loss_fct_cls = nn.MSELoss
            self.loss_fct_kwargs = {}
        elif config.objective in {'triplet', 'triplet_mse'}:
            self.loss_fct_cls = QuadrupletLoss
            self.loss_fct_kwargs = {'distance_function': lambda x, y: 1.0 - cosine_similarity(x, y)}
        else:
            raise ValueError('Only regression and triplet objectives are supported for BiEncoderForClassification')

        self.post_init()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            input_ids_2=None,
            attention_mask_2=None,
            token_type_ids_2=None,
            position_ids_2=None,
            head_mask_2=None,
            inputs_embeds_2=None,
            input_ids_cond=None,
            attention_mask_cond=None,
            token_type_ids_cond=None,
            labels=None,
            **kwargs,
    ):
        device = input_ids.device
        bsz = input_ids.shape[0]
        input_ids_3 = concat_features(input_ids, input_ids_2)
        attention_mask_3 = concat_features(attention_mask, attention_mask_2)
        token_type_ids_3 = concat_features(token_type_ids, token_type_ids_2)
        position_ids_3 = concat_features(position_ids, position_ids_2)
        head_mask_3 = concat_features(head_mask, head_mask_2)
        inputs_embeds_3 = concat_features(inputs_embeds, inputs_embeds_2)

        outputs = self.backbone(
            input_ids=input_ids_3,
            attention_mask=attention_mask_3,
            token_type_ids=token_type_ids_3,
            position_ids=position_ids_3,
            head_mask=head_mask_3,
            inputs_embeds=inputs_embeds_3,
            output_hidden_states=self.output_hidden_states,
        )

        # Final output should be features_1, features_2
        if self.model_args.do_hypernet:
            def hn_pool(input_ids_, attention_mask_, token_type_ids_, outputs_lhs_, outputs_attention_mask_, main_base_pooling_method, hn_base_pooling_method=None):
                with torch.no_grad():
                    pooling_method = hn_base_pooling_method if hn_base_pooling_method is not None else main_base_pooling_method
                    hyper_input = self.hypernet_base_enc(input_ids=input_ids_, attention_mask=attention_mask_, token_type_ids=token_type_ids_)
                    hyper_input = Pooler(pooling_method)(attention_mask_, hyper_input)

                weight, bias = self.hyper(hyper_input)
                weight, bias = weight.to(device), bias.to(device)

                # pool sr from base enc output
                features_ = Pooler(main_base_pooling_method)(outputs_attention_mask_, outputs_lhs_, outputs_is_last_hidden_state=True)
                features_ = features_.to(weight.dtype)
                features_ = features_.unsqueeze(1)

                # forward mlp to make sr' using hn
                temp = torch.bmm(features_, weight)
                temp = temp.view(bsz, -1)
                features_ = temp + bias

                return features_

            outputs_lhs, outputs_lhs2 = torch.split(outputs.last_hidden_state, bsz, dim=0)

            if self.model_args.pooler_type == "hypernet":
                features_1 = hn_pool(input_ids, attention_mask, token_type_ids, outputs_lhs, attention_mask, "avg")
                features_2 = hn_pool(input_ids_2, attention_mask_2, token_type_ids_2, outputs_lhs2, attention_mask_2, "avg")

            elif self.model_args.pooler_type == "hypernet2":
                features_1 = hn_pool(input_ids, attention_mask, token_type_ids, outputs_lhs, attention_mask, "avg-wo-cls")
                features_2 = hn_pool(input_ids_2, attention_mask_2, token_type_ids_2, outputs_lhs2, attention_mask_2, "avg-wo-cls")

            elif self.model_args.pooler_type == "hypernet3":
                features_1 = hn_pool(input_ids_cond, attention_mask_cond, token_type_ids_cond, outputs_lhs, attention_mask, "avg")
                features_2 = hn_pool(input_ids_cond, attention_mask_cond, token_type_ids_cond, outputs_lhs2, attention_mask_2, "avg", "cls")

            else:
                raise NotImplementedError

        else:
            if self.model_args.do_prompt:
                features = outputs.last_hidden_state[input_ids_3 == self.mask_token_id]
            else:
                features = Pooler(self.model_args.pooler_type)(attention_mask_3, outputs)

            if self.transform is not None:
                features = self.transform(features)

            features_1, features_2 = torch.split(features, bsz, dim=0)  # [sentence1, condtion], [sentence2, condition]

        loss = None
        if self.config.objective in {'triplet', 'triplet_mse'}:
            positives1, negatives1 = torch.split(features_1, bsz // 2, dim=0)
            positives2, negatives2 = torch.split(features_2, bsz // 2, dim=0)

            if labels is not None:
                loss = self.loss_fct_cls(**self.loss_fct_kwargs)(positives1, positives2, negatives1, negatives2)

            logits = cosine_similarity(features_1, features_2, dim=1)

            if self.config.objective in {'triplet_mse'} and labels is not None:
                loss += nn.MSELoss()(logits, labels)
            else:
                logits = logits.detach()

        else:
            logits = cosine_similarity(features_1, features_2, dim=1)
            if labels is not None:
                loss = self.loss_fct_cls(**self.loss_fct_kwargs)(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )


class TriEncoderForClassification(PreTrainedModel):  # FIXME Not-implemented: pooler_type == hypernet
    def __init__(self, config, **kwargs):
        super().__init__(config)
        init_cls_set_kwargs(self, kwargs)

        self.backbone = AutoModel.from_pretrained(
            config.model_name_or_path,
            from_tf=bool('.ckpt' in config.model_name_or_path),
            config=config,
            cache_dir=config.cache_dir,
            revision=config.model_revision,
            use_auth_token=True if config.use_auth_token else None,
            add_pooling_layer=False,
        ).base_model
        self.triencoder_head = config.triencoder_head
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        if config.transform:
            self.transform = nn.Sequential(
                nn.Dropout(classifier_dropout),
                nn.Linear(config.hidden_size, config.hidden_size),
                ACT2FN[config.hidden_act],
            )
        else:
            self.transform = None

        self.condition_transform = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        if self.triencoder_head == 'concat':
            self.concat_transform = nn.Sequential(
                nn.Dropout(classifier_dropout),
                nn.Linear(config.hidden_size * 2, config.hidden_size),
                ACT2FN[config.hidden_act],
            )
        elif self.triencoder_head == 'hadamard':
            self.concat_transform = None

        self.pooler = Pooler(config.pooler_type)
        if config.pooler_type in {'avg_first_last', 'avg_top2'}:
            self.output_hidden_states = True
        else:
            self.output_hidden_states = False
        if config.num_labels == 1:
            self.reshape_function = lambda x: x.reshape(-1)
            if config.objective == 'mse':
                self.loss_fct_cls = nn.MSELoss
                self.loss_fct_kwargs = {}
            elif config.objective in {'triplet', 'triplet_mse'}:
                self.loss_fct_cls = QuadrupletLoss
                self.loss_fct_kwargs = {'distance_function': lambda x, y: 1.0 - cosine_similarity(x, y)}
            else:
                raise ValueError('Only regression and triplet objectives are supported for TriEncoderForClassification')
        else:
            self.reshape_function = lambda x: x.reshape(-1, config.num_labels)
            self.loss_fct_cls = nn.CrossEntropyLoss
        self.post_init()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            input_ids_2=None,
            attention_mask_2=None,
            token_type_ids_2=None,
            position_ids_2=None,
            head_mask_2=None,
            inputs_embeds_2=None,
            input_ids_3=None,
            attention_mask_3=None,
            token_type_ids_3=None,
            position_ids_3=None,
            head_mask_3=None,
            inputs_embeds_3=None,
            labels=None,
            **kwargs,
    ):
        bsz = input_ids.shape[0]
        input_ids = concat_features(input_ids, input_ids_2, input_ids_3)
        attention_mask = concat_features(attention_mask, attention_mask_2, attention_mask_3)
        token_type_ids = concat_features(token_type_ids, token_type_ids_2, token_type_ids_3)
        position_ids = concat_features(position_ids, position_ids_2, position_ids_3)
        head_mask = concat_features(head_mask, head_mask_2, head_mask_3)
        inputs_embeds = concat_features(inputs_embeds, inputs_embeds_2, inputs_embeds_3)
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=self.output_hidden_states,
        )

        if self.model_args.do_prompt:
            features = outputs.last_hidden_state[input_ids == self.mask_token_id]
        else:
            features = self.pooler(attention_mask, outputs)

        features_1, features_2, features_3 = torch.split(features, bsz, dim=0)
        features_3 = self.condition_transform(features_3)
        # do we need positional embeddings?
        loss = None
        if self.transform is not None:
            features_1 = self.transform(features_1)
            features_2 = self.transform(features_2)
        if self.triencoder_head == 'concat':
            features_1 = torch.cat([features_1, features_3], dim=-1)
            features_2 = torch.cat([features_2, features_3], dim=-1)
            features_1 = self.concat_transform(features_1)
            features_2 = self.concat_transform(features_2)
        elif self.triencoder_head == 'hadamard':
            features_1 = features_1 * features_3
            features_2 = features_2 * features_3
        if self.config.objective in {'triplet', 'triplet_mse'}:
            positive_idxs = torch.arange(0, features_1.shape[0] // 2)
            negative_idxs = torch.arange(features_1.shape[0] // 2, features_1.shape[0])
            positives1 = features_1[positive_idxs]
            positives2 = features_2[positive_idxs]
            negatives1 = features_1[negative_idxs]
            negatives2 = features_2[negative_idxs]
            if labels is not None:
                loss = self.loss_fct_cls(**self.loss_fct_kwargs)(positives1, positives2, negatives1, negatives2)
            logits = cosine_similarity(features_1, features_2, dim=1)
            if self.config.objective == 'triplet_mse' and labels is not None:
                loss += nn.MSELoss()(logits, labels)
            else:
                logits = logits.detach()
        else:
            logits = cosine_similarity(features_1, features_2, dim=1)
            if labels is not None:
                loss = self.loss_fct_cls(**self.loss_fct_kwargs)(logits, labels)
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )
