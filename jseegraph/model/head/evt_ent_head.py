#!/usr/bin/env python3
# coding=utf-8

import torch
import torch.nn as nn

from model.head.abstract_head import AbstractHead
from data.parser.to_mrp.evt_ent_parser import EvtEntParser
from utility.cross_entropy import binary_cross_entropy
from utility.hungarian_matching import match_label


class EvtEntHead(AbstractHead):
    def __init__(self, dataset, args, framework, initialize):
        config = {
            "label": True,
            "edge presence": True,
            "edge label": True,
            "anchor": True
        }
        super(EvtEntHead, self).__init__(dataset, args, framework, config, initialize)

        self.top_node = nn.Parameter(torch.randn(1, 1, args.hidden_size), requires_grad=True)
        self.parser = EvtEntParser(dataset)


    def forward_edge(self, decoder_output):
        top_node = self.top_node.expand(decoder_output.size(0), -1, -1)
        decoder_output = torch.cat([top_node, decoder_output], dim=1)
        return self.edge_classifier(decoder_output)
