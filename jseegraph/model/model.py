#!/usr/bin/env python3
# coding=utf-8

import torch
import torch.nn as nn

from model.module.encoder import Encoder

from model.module.transformer import Decoder
from model.head.labeled_edge_head import LabeledEdgeHead
from model.head.evt_head import EvtHead
from model.head.evt_ent_head import EvtEntHead
from utility.utils import create_padding_mask
from model.module.module_wrapper import ModuleWrapper
from data.batch import Batch


class Model(nn.Module):
    def __init__(self, dataset, args, initialize=True):
        super(Model, self).__init__()
        self.encoder = Encoder(args, dataset)
        if args.n_layers > 0:
            self.decoder = Decoder(args)
        else:
            self.decoder = lambda x, *args: x  # identity function, which ignores all arguments except the first one



        head_dict = {
            ("ace_p_evt", "en"): EvtHead, ("ace_p_evt", "zh"): EvtHead,
            ("ace_p_evt_ent", "en"): EvtEntHead, ("ace_p_evt_ent", "zh"): EvtEntHead,
            ("ace_pp_evt", "en"): EvtHead, ("ace_pp_evt", "zh"): EvtHead,
            ("ace_pp_evt_ent", "en"): EvtEntHead, ("ace_pp_evt_ent", "zh"): EvtEntHead,
            ("ere_p_evt", "en"): EvtHead, ("ere_p_evt", "zh"): EvtHead, ("ere_p_evt", "es"): EvtHead,
            ("ere_p_evt_ent", "en"): EvtEntHead, ("ere_p_evt_ent", "zh"): EvtEntHead, ("ere_p_evt_ent", "es"): EvtEntHead,
            ("ere_pp_evt", "en"): EvtHead, ("ere_pp_evt", "zh"): EvtHead, ("ere_pp_evt", "es"): EvtHead,
            ("ere_pp_evt_ent", "en"): EvtEntHead, ("ere_pp_evt_ent", "zh"): EvtEntHead, ("ere_pp_evt_ent", "es"): EvtEntHead,
        }
        

        frameworks = [v[0] for k,v in dataset.id_to_framework.items()]
        assert len(set(frameworks)) == 1, "Too many framworks to unpack"
        framework = frameworks[0]
        self.head = head_dict[dataset.id_to_framework[0]](dataset.child_datasets[dataset.id_to_framework[0]], args, framework, initialize)
        self.total_epochs = args.epochs    
        self.query_length = args.query_length
        self.dataset = dataset
        self.args = args


    def forward(self, batch, inference=False, **kwargs):
        every_input, word_lens = batch["every_input"]
        decoder_lens = self.query_length * word_lens
        batch_size, input_len = every_input.size(0), every_input.size(1)
        device = every_input.device

        encoder_mask = create_padding_mask(batch_size, input_len, word_lens, device)
        decoder_mask = create_padding_mask(batch_size, self.query_length * input_len, decoder_lens, device)

        encoder_output, decoder_input = self.encoder(batch["input"], batch["char_form_input"], batch["input_scatter"], input_len, batch['framework'])

        decoder_output = self.decoder(decoder_input, encoder_output, decoder_mask, encoder_mask)


        def select_inputs(indices):
            return (
                encoder_output.index_select(0, indices),
                decoder_output.index_select(0, indices),
                encoder_mask.index_select(0, indices),
                decoder_mask.index_select(0, indices),
                Batch.index_select(batch, indices),
            )


        if inference:
            output = {}
            for i in range(len(self.dataset.child_datasets)):
                indices = (batch["framework"] == i).nonzero(as_tuple=False).flatten()
                if indices.size(0) == 0:
                    continue
                output[self.dataset.id_to_framework[i]] = self.head.predict(*select_inputs(indices), **kwargs)

            return output
        else:
            return self.head(encoder_output, decoder_output, encoder_mask, decoder_mask, batch)


    def get_params_for_optimizer(self, args):
        encoder_decay, encoder_no_decay = self.get_encoder_parameters(args.n_encoder_layers)
        decoder_decay, decoder_no_decay = self.get_decoder_parameters()

        parameters = [{"params": p, "weight_decay": args.encoder_weight_decay} for p in encoder_decay]
        parameters += [{"params": p, "weight_decay": 0.0} for p in encoder_no_decay]
        parameters += [
            {"params": decoder_decay, "weight_decay": args.decoder_weight_decay},
            {"params": decoder_no_decay, "weight_decay": 0.0},
        ]
        return parameters

    def get_decoder_parameters(self):
        no_decay = ["bias", "LayerNorm.weight", "_norm.weight"]
        decay_params = (p for name, p in self.named_parameters() if not any(nd in name for nd in no_decay) and not name.startswith("encoder.bert") and p.requires_grad)
        no_decay_params = (p for name, p in self.named_parameters() if any(nd in name for nd in no_decay) and not name.startswith("encoder.bert") and p.requires_grad)

        return decay_params, no_decay_params

    def get_encoder_parameters(self, n_layers):
        no_decay = ["bias", "LayerNorm.weight", "_norm.weight"]
        decay_params = [
            [p for name, p in self.named_parameters() if not any(nd in name for nd in no_decay) and name.startswith(f"encoder.bert.encoder.layer.{n_layers - 1 - i}.") and p.requires_grad] for i in range(n_layers)
        ]
        no_decay_params = [
            [p for name, p in self.named_parameters() if any(nd in name for nd in no_decay) and name.startswith(f"encoder.bert.encoder.layer.{n_layers - 1 - i}.") and p.requires_grad] for i in range(n_layers)
        ]

        return decay_params, no_decay_params


    def get_dummy_batch(self, head, device):
        encoder_output = torch.zeros(1, 1, self.args.hidden_size, device=device)
        decoder_output = torch.zeros(1, self.query_length, self.args.hidden_size, device=device)
        encoder_mask = torch.zeros(1, 1, dtype=torch.bool, device=device)
        decoder_mask = torch.zeros(1, self.query_length, dtype=torch.bool, device=device)

        batch = {
            "every_input": (torch.zeros(1, 1, dtype=torch.long, device=device), torch.ones(1, dtype=torch.long, device=device)),
            "input": (torch.zeros(1, 1, dtype=torch.long, device=device), torch.ones(1, dtype=torch.long, device=device)),
            "labels": ([torch.zeros(1, 1, len(head.dataset.label_field.vocab) + 1, device=device)], [torch.ones(1, dtype=torch.long, device=device)]),
            "edge_presence": torch.zeros(1, 1, 1, dtype=torch.long, device=device),
            "edge_labels": (torch.zeros(1, 1, 1, head.dataset.edge_label_freqs.size(0), dtype=torch.long, device=device), torch.zeros(1, 1, 1, dtype=torch.bool, device=device)),
            "anchor": (torch.zeros(1, 1, 1, dtype=torch.long, device=device), torch.zeros(1, 1, dtype=torch.bool, device=device))
        }

        return encoder_output, decoder_output, encoder_mask, decoder_mask, batch                