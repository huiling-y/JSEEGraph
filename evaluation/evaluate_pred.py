#!/usr/bin/env python3
# coding: utf-8
from __future__ import print_function
import sys
import os
import json

from nltk.tokenize.simple import SpaceTokenizer

tk = SpaceTokenizer()

from collections import defaultdict

def convert_char_offsets_to_token_idxs(char_offsets, token_offsets):
    """
    char_offsets: list of str
    token_offsets: list of tuples

    >>> text = "I think the new uni ( ) is a great idea"
    >>> char_offsets = ["8:19"]
    >>> token_offsets =
    [(0,1), (2,7), (8,11), (12,15), (16,19), (20,21), (22,23), (24,26), (27,28), (29,34), (35,39)]

    >>> convert_char_offsets_to_token_idxs(char_offsets, token_offsets)
    >>> (2,3,4)
    """
    token_idxs = []
    #
    for char_offset in char_offsets:
        bidx, eidx = char_offset.split(":")
        bidx, eidx = int(bidx), int(eidx)
        intoken = False
        for i, (b, e) in enumerate(token_offsets):
            if b == bidx:
                intoken = True
            if intoken:
                token_idxs.append(i)
            if e == eidx:
                intoken = False
    return frozenset(token_idxs)

def convert_entity_to_tuple(sentence, lang='en'):
    text = sentence['text']
    entities = sentence['entities']
    entity_tuples = set()
    if lang == 'zh':
        token_offsets = [(i, i+1) for i in range(len(text))]
    else:
        token_offsets = list(tk.span_tokenize(text))

    if len(entities) > 0:
        for entity in entities:

            entity_char_idxs = entity[1]
            _entity = convert_char_offsets_to_token_idxs(entity_char_idxs, token_offsets)
            _entity_type = entity[-1].lower()

            entity_tuples.add((_entity, _entity_type))
    return list(entity_tuples)

def convert_event_to_tuple(sentence, lang='en'):
    """
    >>> sentence 
    {'sent_id': 'nw/APW_ENG_20030322.0119/001',
    'text': 'U.S. and British troops were moving on the strategic southern port city of Basra Saturday after a massive aerial assault pounded Baghdad at dawn',
    'events': [
        {
            'event_type': 'Attack',
            'trigger': [['pounded'], ['121:128']],
            'arguments': [[['Baghdad'], ['129:136'], 'Place'], [['dawn'], ['140:144'], 'Time-Starting']]},
        {
            'event_type': 'Transport',
            'trigger': [['moving'], ['29:35']],
            'arguments': [[['Saturday'], ['81:89'], 'Time-Within'], [['the strategic southern port city of Basra'], ['39:80'], 'Destination'], [['U.S. and British troops'], ['0:23'], 'Artifact']]}
        
                ]
    
    }

    >>> event_tupels 
    [
        ((frozenset({20}), 'attack'), (frozenset({21}), 'place'), (frozenset({23}), 'time-starting')),
        ((frozenset({5}), 'transport'), (frozenset({14}), 'time-within'), (frozenset({7, 8, 9, 10, 11, 12, 13}), 'destination'), (frozenset({0, 1, 2, 3}), 'artifact'))
    ]

    >>> trigger_tupels 
    {
        (frozenset({20}), 'attack'), ((frozenset({5}), 'transport')
    }

    >>> argument_tupels -> (offset, arg_role, event_type)
    {
        (frozenset({21}), 'place', 'attack'), (frozenset({23}), 'time-starting', 'attack'), (frozenset({14}), 'time-within', 'transport'),
        (frozenset({7, 8, 9, 10, 11, 12, 13}), 'destination', 'transport'), (frozenset({0, 1, 2, 3}), 'artifact', 'transport')
    }

    
    """

    text = sentence['text']
    events = sentence['events']

    trigger_tuples = set()
    argument_tuples = set()

    if lang == 'zh':
        token_offsets = [(i, i+1) for i in range(len(text))]
    else:
        token_offsets = list(tk.span_tokenize(text))

    if len(events) > 0:

        for event in events:

            trigger_char_idxs = event['trigger'][1]
            trigger = convert_char_offsets_to_token_idxs(trigger_char_idxs, token_offsets)
            event_type = event['event_type'].lower() if event['event_type'] else "none"

            trigger_tuples.add((trigger, event_type))

            if len(event['arguments']) > 0:
                for argument in event['arguments']:
                    arg_role = argument[-1]
                    argument_char_idxs = argument[1]
                    arg = convert_char_offsets_to_token_idxs(argument_char_idxs, token_offsets)

                    argument_tuples.add((arg, arg_role, event_type))
    return list(trigger_tuples), list(argument_tuples)


def trigger_tuple_in_list(trigger_tuple, trigger_tuple_list, classification=False):
    if classification:
        if trigger_tuple in trigger_tuple_list:
            return True
        else:
            return False
    else:
        for trigger in trigger_tuple_list:
            if trigger_tuple[0] == trigger[0]:
                return True
        return False

def arg_tuple_in_list(arg_tuple, arg_tuple_list, classification=False):
    if classification:
        if arg_tuple in arg_tuple_list:
            return True
        else:
            return False
    else:
        for arg in arg_tuple_list:
            if arg_tuple[0] == arg[0]:
                return True
        return False

def entity_tuple_in_list(entity_tuple, entity_tuple_list, classification=False):
    if classification:
        if entity_tuple in entity_tuple_list:
            return True
        else:
            return False
    else:
        for entity in entity_tuple_list:
            if entity_tuple[0] == entity[0]:
                return True
        return False

def trigger_precision(gold, pred, classification=False):
    """
    triggers in list of tuples
    [(frozenset({20}), 'attack'), ((frozenset({5}), 'transport')]

    both gold, pred
    """
    tp = []
    fp = []
    for sent_idx in pred.keys():
        ptuples = pred[sent_idx]
        gtuples = gold[sent_idx]

        for stuple in ptuples:
            if trigger_tuple_in_list(stuple, gtuples, classification=classification):
                tp.append(1)
            else:
                fp.append(1)
    return sum(tp) / (sum(tp) + sum(fp) + 0.0000000000000001)

def trigger_recall(gold, pred, classification=False):
    tp = []
    fn = []
    for sent_idx in pred.keys():
        ptuples = pred[sent_idx]
        gtuples = gold[sent_idx]

        for stuple in gtuples:
            if trigger_tuple_in_list(stuple, ptuples, classification=classification):
                tp.append(1)
            else:
                fn.append(1)
    return sum(tp) / (sum(tp) + sum(fn) + 0.0000000000000001)

def trigger_f1(gold, pred, classification=False):
    precision = trigger_precision(gold, pred, classification=classification)
    recall = trigger_recall(gold, pred, classification=classification)

    f1 = 2 * (precision * recall) / (precision + recall + 0.00000000000000001)

    return precision, recall, f1



def argument_precision(gold, pred, classification=False):
    """
    arguments in list of tuples
    {
        (frozenset({21}), 'place', 'attack'), (frozenset({23}), 'time-starting', 'attack'), (frozenset({14}), 'time-within', 'transport'),
        (frozenset({7, 8, 9, 10, 11, 12, 13}), 'destination', 'transport'), (frozenset({0, 1, 2, 3}), 'artifact', 'transport')
    }

    each arg: (offset, arg_role, entity_type)
    """

    tp = []
    fp = []
    for sent_idx in pred.keys():
        ptuples = pred[sent_idx]
        gtuples = gold[sent_idx]

        for stuple in ptuples:
            if arg_tuple_in_list(stuple, gtuples, classification=classification):
                tp.append(1)
            else:
                fp.append(1)
    return sum(tp) / (sum(tp) + sum(fp) + 0.0000000000000001)

def argument_recall(gold, pred, classification=False):
    tp = []
    fn = []
    for sent_idx in pred.keys():
        ptuples = pred[sent_idx]
        gtuples = gold[sent_idx]

        for stuple in gtuples:
            if arg_tuple_in_list(stuple, ptuples, classification=classification):
                tp.append(1)
            else:
                fn.append(1)
    return sum(tp) / (sum(tp) + sum(fn) + 0.0000000000000001)

def argument_f1(gold, pred, classification=False):
    precision = argument_precision(gold, pred, classification=classification)
    recall = argument_recall(gold, pred, classification=classification)

    f1 = 2 * (precision * recall) / (precision + recall + 0.00000000000000001)

    return precision, recall, f1

def entity_precision(gold, pred, classification=False):
    tp = []
    fp = []

    for sent_idx in pred.keys():
        ptuples = pred[sent_idx]
        gtuples = gold[sent_idx]

        for stuple in ptuples:
            if entity_tuple_in_list(stuple, gtuples, classification=classification):
                tp.append(1)
            else:
                fp.append(1)
    return sum(tp) / (sum(tp) + sum(fp) + 0.0000000000000001)

def entity_recall(gold, pred, classification=False):
    tp = []
    fn = []

    for sent_idx in pred.keys():
        ptuples = pred[sent_idx]
        gtuples = gold[sent_idx]

        for stuple in gtuples:
            if entity_tuple_in_list(stuple, ptuples, classification=classification):
                tp.append(1)
            else:
                fn.append(1)

    return sum(tp) / (sum(tp) + sum(fn) + 0.0000000000000001)

def entity_f1(gold, pred, classification=False):
    precision = entity_precision(gold, pred, classification=classification)
    recall = entity_recall(gold, pred, classification=classification)

    f1 = 2 * (precision * recall) / (precision + recall + 0.00000000000000001)

    return precision, recall, f1  