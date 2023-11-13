import os
import re
import json
import glob
import tqdm
import random
import numpy as np
import copy
from lxml import etree
from typing import List, Dict, Any, Tuple
from stanfordcorenlp import StanfordCoreNLP
from bs4 import BeautifulSoup
from dataclasses import dataclass
from argparse import ArgumentParser
import stanza

from preprocess_ace import (
    Span,
    Entity,
    RelationArgument,
    Relation,
    EventArgument,
    Event,
    Sentence,
    Document,
    StanfordCoreNLPv2,
    new_span,
    span_to_string,
    load_json,
    dump_json,
    recover_sent,
    retokenize,
    process_entities,
    process_events,
    process_relation,
    find_index,
    update_offset,
    split_data
)

DIRS = ['df', 'nw']
SPLIT_TAGS = {
    'df':['<headline>','</headline>', '<post', '</post>', '<quote', '</quote>', '<a', '</a>'],
    'nw':['<TEXT>','</TEXT>','<HEADLINE>','</HEADLINE>', '<DATE_TIME>', '</DATE_TIME>', '<AUTHOR>', '</AUTHOR>', '<P>', '</P>']
}
MASK_TAGS = ['<img'] #, '<a', '</a>']

def get_dir(path):
    patterns = ['/df/', '/nw/']
    for i,pattern in enumerate(patterns):
        if pattern in path:
            return DIRS[i]

def read_annotate_file(path: str,
                  time_and_val: bool = False,
                  head_only: bool = True
                 ) -> Tuple[str, str, List[Entity], List[Relation], List[Event]]:
    """Reads an annotation file.

    Args:
        path (str): path to the input file.
        time_and_val (bool): extract times and values or not.
        head_only: use only the head of an entity or the full span
    
    Returns:
        doc_id (str): document ID.
        source (str): document source.
        entity_list (List[Entity]): a list of Entity instances.
        relation_list (List[Relation]): a list of Relation instances.
        event_list (List[Event]): a list of Events instances.
    """

    data = open(path, 'r', encoding='utf-8').read()
    soup = BeautifulSoup(data, 'lxml-xml')

    # metadata
    root = soup.find('deft_ere')
    source = root['source_type']
    doc_id = root['doc_id']
    #doc = root.find('document')

    entity_list, relation_list, event_list = [], [], []    

    # entities
    for entity in root.find_all('entity'):
        entity_id = entity['id']
        entity_type = entity['type']
        
        for entity_mention in entity.find_all('entity_mention'):
            mention_id = entity_mention['id']
            mention_type = entity_mention['noun_type']
            if head_only and entity_mention.find('nom_head'):
                head = entity_mention.find('nom_head')
                text = head.text
                start = int(head['offset'])
                #end = start + int(head['length']) - 1
                end = start + len(text) - 1
            else:
                mention = entity_mention.find('mention_text')
                text = mention.text
                start = int(entity_mention['offset'])
                #end = start + int(entity_mention['length']) - 1
                end = start + len(text) - 1
            entity_list.append(
                Entity(start, end, text, entity_id, mention_id, entity_type, entity_type, mention_type)
            )
    
    if time_and_val:
        # fillers 
        for entity in root.find_all('fillers'):
            for entity_mention in entity.find_all('filler'):
                entity_id = entity_mention['id']
                entity_type = entity_mention['type']
                mention_id = entity_mention['id']
                mention_type = 'null'
                text = entity_mention.text
                start = int(entity_mention['offset'])
                #end = start + int(entity_mention['length']) - 1
                end = start + len(text) - 1

                entity_list.append(Entity(start, end, text,
                                        entity_id, mention_id, entity_type,
                                        entity_type, mention_type))
    # relations
    for relation in root.find_all('relation'):
        relation_id = relation['id']
        relation_type = relation['type']
        relation_subtype = relation['subtype']
        for relation_mention in relation.find_all('relation_mention'):
            mention_id = relation_mention['id']
            _arg1 = _arg2 = None
            
            _arg1 = relation_mention.find('rel_arg1')
            _arg2 = relation_mention.find('rel_arg2')

            if 'entity_id' in _arg1.attrs:
                _arg1_mention_id = _arg1['entity_mention_id']
            else:
                _arg1_mention_id = _arg1['filler_id']

            if 'entity_id' in _arg2.attrs:
                _arg2_mention_id = _arg2['entity_mention_id']
            else:
                _arg2_mention_id = _arg2['filler_id']            


            if _arg1 and _arg2:
                arg1 = RelationArgument(_arg1_mention_id, 'Arg-1', _arg1.text)
                arg2 = RelationArgument(_arg2_mention_id, 'Arg-2', _arg2.text)
                relation_list.append(
                    Relation(mention_id, relation_type, relation_subtype, arg1, arg2)
                )
    
    # events
    for event in root.find_all('hopper'):
        event_id = event['id']

        for event_mention in event.find_all('event_mention'):
            mention_id = event_mention['id']
            event_type = event_mention['type']
            event_subtype = event_mention['subtype']
            trigger = event_mention.find('trigger')
            trigger_start = int(trigger['offset'])
            #trigger_end = trigger_start + int(trigger['length']) - 1
            trigger_end = trigger_start + len(text) - 1
            trigger_text = trigger.text
            event_args = []
            for arg in event_mention.find_all('em_arg'):
                if 'entity_id' in arg.attrs:
                    arg_mention_id = arg['entity_mention_id']
                    arg_role = arg['role']
                    arg_text = arg.text
                if 'filler_id' in arg.attrs:
                    arg_mention_id = arg['filler_id']
                    arg_role = arg['role']
                    arg_text = arg.text
                if arg_text:
                    event_args.append(EventArgument(arg_mention_id, arg_role, arg_text))
            event_list.append(
                Event(
                    event_id, mention_id,
                    event_type, event_subtype,
                    Span(trigger_start, trigger_end, trigger_text),
                    event_args
                ))
    return doc_id, source, entity_list, relation_list, event_list 


def find_index(offsets, offset):
    idx_start = -1
    idx_end = -1
    for j, _offset in enumerate(offsets):
        if idx_start == -1 and _offset[0] <= offset[0] and _offset[1] > offset[0]:
            idx_start = j 
        if idx_end == -1 and _offset[0] <= offset[1] and _offset[1] > offset[1]:
            idx_end = j
            break
    assert idx_start != -1 and idx_end != -1
    return idx_start, idx_end

def correct_offsets(sents, offsets):
    new_offsets = copy.deepcopy(offsets)
    for i, (sent, offset) in enumerate(zip(sents, offsets)):
        for j, _offset in enumerate(offset):
            if sent[j].startswith('<'):
                new_offsets[i][j] = (0, 0)
                if sent[j].startswith('<img'):
                    new_offsets[i][j] = (-1, -1)
    return sents, new_offsets

def filter_mask(sents, offsets):
    new_sents = []
    new_offsets = []
    for i, (sent, offset_per_sent) in enumerate(zip(sents, offsets)):
        _subsent = []
        _subsent_offset = []
        for j, (token, offset) in enumerate(zip(sent, offset_per_sent)):
            if offset != (-1, -1):
                _subsent.append(token)
                _subsent_offset.append(offset)
            else:
                if len(_subsent) > 0:
                    new_sents.append(_subsent)
                    new_offsets.append(_subsent_offset)

                    _subsent = []
                    _subsent_offset = []

        if len(_subsent) > 0:
            new_sents.append(_subsent)
            new_offsets.append(_subsent_offset)            


    return new_sents, new_offsets



def read_source_file(path: str,
                  corenlp_path: str,
                  language: str = 'en') -> Tuple[List, List]:
    """Reads a SGM text file.
    
    Args:
        path (str): path to the input file.
        language (str): document language. Valid values: "en" or "zh", for English and Chinese respectively

    Returns:
        List of sentences; List of token offsets.
    """
    nlp = StanfordCoreNLPv2(corenlp_path)

    with open(path, 'r', encoding='utf-8') as f:
        special_tokens = ['%25', '%3B', '%3A', '%2F', '%7C',\
             '%26', '%3D', '%3F', '%E2', '%80', '%9C',\
                '%20', '%2b', '%2f', '%3d', '%C3', '%AD', '%A9', '%9D']
        text = f.read()#.replace('%25', '###').replace('%3B', '***').replace("%3A", '***').replace("%2F", "***").replace('%7C', '***').replace('%26', '***').replace('%3D', '***').replace('%3F', '***').replace('%E2', '***').replace('%80', '***').replace('%9C', '***').replace('%9D', '***').replace('%20', '***').replace('%2b', '***').replace('%2f', '***').replace('%3d', '***')
        for spec_tok in special_tokens:
            text = text.replace(spec_tok, '***')
    sents, offsets = nlp.sent_tokenize(text)
    nlp.close()  

    sents, offsets = correct_offsets(sents, offsets)

    _dir = get_dir(path)
    mark_split_tag = SPLIT_TAGS[_dir]

    new_sents = []
    new_offsets = []

    for i, sent in enumerate(sents):
        offset_per_sentence = offsets[i]
        select = True

        start_posi = 0
        for j, token in enumerate(sent):
            if bool(sum([token.startswith(e) for e in mark_split_tag])):
                subsent = sent[start_posi:j]
                suboffset = offset_per_sentence[start_posi:j]
                if select and len(subsent) > 0:
                    assert (0, 0) not in suboffset
                    new_sents.append(subsent)
                    new_offsets.append(suboffset)
                start_posi = j+1
                select = True 
            elif token.startswith('<') and (not token.startswith('<img')): #and not bool(sum([token.startswith(e) for e in MASK_TAGS])):
                select = False 
        
        subsent = sent[start_posi:]
        suboffset = offset_per_sentence[start_posi:]
        if select and len(subsent) > 0:
            assert (0, 0) not in suboffset
            new_sents.append(subsent)
            new_offsets.append(suboffset)
    
    new_sents, new_offsets = filter_mask(new_sents, new_offsets)

    if language == 'zh':
        _new_sents, _new_offsets = retokenize(new_sents, new_offsets, corenlp_path, language)
        return _new_sents, _new_offsets

    return new_sents, new_offsets


            
def convert(src_file: str,
            annotate_file: str,
            corenlp_path: str,
            time_and_val: bool = False,
            head_only: bool = True,
            language: str = 'en') -> Document:
    """Converts a document.

    Args:
        sgm_file (str): path to a SGM file.
        apf_file (str): path to a APF file.
        time_and_val (bool, optional): extracts times and values or not.
            Defaults to False.
        language (str, optional): document language. Available options: english,
            chinese. Defaults to 'english'.

    Returns:
        Document: a Document instance.
    """
    sentences = read_source_file(src_file, corenlp_path=corenlp_path, language=language)
    doc_id, source, entities, relations, events = read_annotate_file(
        annotate_file, time_and_val=time_and_val, head_only=head_only)
    
    # Process entities, relations, and events
    sentence_entities, unassigned = process_entities(entities, sentences)
    sentence_relations = process_relation(
        relations, sentence_entities, sentences)
    sentence_events = process_events(events, sentence_entities, sentences)
    
    doc_unassigned = {
        'doc_id': doc_id,
        'unassigned_ets': [e.to_dict() for e in unassigned]
    }

    sents, offsets = sentences
    # Convert span character offsets to token indices 
    sentence_objs = []
    for i, (sent_tok, offset, ents, evts, rels) in enumerate(zip(
            sents, offsets, sentence_entities, sentence_events,
            sentence_relations)):
        for entity in ents:
            entity.char_offsets_to_token_offsets(offset, sent_tok)
            
        for event in evts:
            # the trigger for event_mention with id="em-931" has wrong offset annotation, thus processed manually, the start offset should be 419
            if doc_id == "cb156ad2a5458fabc9e093b6b5e0f97f" and event.trigger.start == 418 and event.trigger.end == 426:
                event.trigger.start = 419
            if doc_id == 'ENG_DF_001471_20131206_G00A0FP48':
                if event.trigger.start == 354 and event.trigger.end == 359:
                    event.trigger.start = 355
                    event.trigger.end = 358
                if event.trigger.start == 722 and event.trigger.end == 728:
                    event.trigger.start = 723
                    event.trigger.end = 727 

            event.trigger.char_offsets_to_token_offsets(offset, sent_tok)             
           
        sent_id = '{}-{}'.format(doc_id, i)
        if language == 'zh':
            sent_text = ''.join(sent_tok)
        else:
            sent_text = ' '.join(sent_tok)
        sentence_objs.append(Sentence(start=offset[0][0],
                                      end=offset[-1][-1],
                                      text=sent_text,
                                      sent_id=sent_id,
                                      tokens=sent_tok,
                                      entities=ents,
                                      relations=rels,
                                      events=evts))
    return Document(doc_id, sentence_objs), doc_unassigned


def convert_batch(input_path: str,
                  output_path: str,
                  corenlp_path: str,
                  time_and_val: bool = False,
                  head_only: bool = True,
                  language: str = 'en'):
    """Converts a batch of documents.

    Args:
        input_path (str): path to the input directory. Usually, it is the path 
            to the LDC2006T06/data/English or LDC2006T06/data/Chinese folder.
        output_path (str): path to the output JSON file.
        time_and_val (bool, optional): extracts times and values or not.
            Defaults to False.
        language (str, optional): document language. Available options: english,
            chinese. Defaults to 'english'.
    """
    all_files = glob.glob(
        os.path.join(input_path, '**', '*.xml')
    )

    src_files = []
    annotate_files = []
    for i in range(len(all_files)):
        if all_files[i].endswith('rich_ere.xml'):
            annotate_files.append(all_files[i])
        else:
            src_files.append(all_files[i])

    src_files = sorted(src_files)
    annotate_files = sorted(annotate_files)

    assert len(src_files) == len(annotate_files)
    
    files = [f[:-13] for f in annotate_files]

    print('Converting the dataset to JSON format')
    print('#XML files: {}'.format(len(src_files)))

    all_unassigned = []
    with open(output_path, 'w', encoding='utf-8') as w:
        for file in tqdm.tqdm(files):

            annotate_file = f"{file}.rich_ere.xml"
            src_file = f"{file}.xml"
            doc, unassigned = convert(src_file, annotate_file, corenlp_path, time_and_val=time_and_val, head_only=head_only,
                          language=language)
            all_unassigned.append(unassigned)
            w.write(json.dumps(doc.to_dict(), ensure_ascii=False) + '\n')
    
    with open(f'{output_path}-unassigned', 'w', encoding='utf-8') as w:
        json.dump(all_unassigned, w, ensure_ascii=False)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', help='Path to the input folder', default='~/Downloads/ERE/all_data')
    parser.add_argument('-o', '--output', help='Path to the output folder', default='../test_preprocess_ere')
    parser.add_argument('-s', '--split', default=None,
                        help='Path to the split folder')
    parser.add_argument('-d',
                        '--dataset',
                        help='Dataset type: ere_evt_ent, ere_evt, ere_ent',
                        default='ere_evt_ent')
    parser.add_argument('-l', '--lang', default='en',
                        help='Document language')
    parser.add_argument('--time_and_val', action='store_true',
                        help='Extracts times and values', default=True)
    parser.add_argument('--head_only', action='store_true',
                        help='Extracts only the head of entity', default=False)
    parser.add_argument('--corenlp', action='store_true',
                        help='corenlp path', default='~/Downloads/ace_2005_td_v7/stanford-corenlp-full-2018-02-27')

    args = parser.parse_args()
    if args.lang not in ['zh', 'en', 'es']:
        raise ValueError('Unsupported language: {}'.format(args.lang))

    input_dir = os.path.join(args.input, args.lang)
    print(input_dir)

    # Convert to doc-level JSON format
    json_path = os.path.join(args.output, '{}.json'.format(args.lang))
    convert_batch(input_dir, json_path, args.corenlp, time_and_val=args.time_and_val,
                  head_only=args.head_only, language=args.lang)   

    raw_path = os.path.join(args.output, '{}.{}.raw.json'.format(args.lang,
                                                                     args.dataset))
    update_offset(json_path, raw_path)
    split_data(raw_path, args.output, args.split)