import os
import re
import json
import glob
import tqdm
import random
import stanza
import numpy as np
import copy
from lxml import etree
from typing import List, Dict, Any, Tuple
from stanfordcorenlp import StanfordCoreNLP
from bs4 import BeautifulSoup
from dataclasses import dataclass
from argparse import ArgumentParser

DIRS = ['bc', 'bn', 'cts', 'nw', 'un', 'wl']


SPLIT_TAGS = {
    'bc':["<SPEAKER>", "</SPEAKER>",'</TURN>','<HEADLINE>','</HEADLINE>', '<DATETIME>', '</DATETIME>'],
    'bn':["<TURN>","</TURN>", '<DATETIME>', '</DATETIME>', '<ENDTIME>' '</ENDTIME>'],
    "cts":["<SPEAKER>", "</SPEAKER>","</TURN>", '<DATETIME>', '</DATETIME>'],
    'nw':['<TEXT>','</TEXT>','<HEADLINE>','</HEADLINE>', '<DATETIME>', '</DATETIME>'],
    'un':['</SUBJECT>','<HEADLINE>','</HEADLINE>','<SUBJECT>','</POST>','<QUOTE', '<POSTER>', '</POSTER>', '<POSTDATE>','</POSTDATE>', '<DATETIME>','</DATETIME>'],
    'wl':['</POSTDATE>','</POST>','<HEADLINE>','</HEADLINE>','<TEXT>','</TEXT>', '<POSTER>', '</POSTER>',  '<DATETIME>', '</DATETIME>']
}



class StanfordCoreNLPv2(StanfordCoreNLP):
    def __init__(self, path, lang='en'):
        super(StanfordCoreNLPv2, self).__init__(path, lang=lang)
    
    def sent_tokenize(self, sentence):
        r_dict = self._request('ssplit, tokenize', sentence)
        tokens = [[token['originalText'] for token in s['tokens']] for s in r_dict['sentences']]
        spans = [[(token['characterOffsetBegin'], token['characterOffsetEnd']) for token in s['tokens']] for s in r_dict['sentences']]
        return tokens, spans


@dataclass
class Span:
    start: int
    end: int
    text: str

    def __post_init__(self):
        self.start = int(self.start)
        self.end = int(self.end)
        self.text = self.text.replace('\n', ' ')
    
    def to_dict(self) -> Dict[str, Any]:
        """Converts instance variables to a dict.
        
        Returns:
            dict: a dict of instance variables.
        """
        return {
            'text': self.text,
            'start': self.start,
            'end': self.end
        }
    
    def char_offsets_to_token_offsets(self, offsets, toks):
        """Converts self.start and self.end from character offsets to token
        offsets.

        Args:
            offsets (List[int, int]): a list of char offsets of the tokens,
            each with the start and end
        """
        #tokens_offsets = [(e[0], e[1]-1) for e in offsets] # convert to inclusive span
        idx_start = -1
        idx_end = -1
        matched_span = self.string_match(offsets, toks)
        for j, _offset in enumerate(offsets):
            #if idx_start == -1 and _offset[0] <= self.start and _offset[1] > self.start:
            if idx_start == -1 and _offset[0] <= matched_span[0] and _offset[1] > matched_span[0]:
                idx_start = j 
            #if idx_end == -1 and _offset[0] <= self.end and _offset[1] > self.end:
            if idx_end == -1 and _offset[0] <= (matched_span[1]-1) and _offset[1] > (matched_span[1]-1):
                idx_end = j
                break
        assert idx_start != -1 and idx_end != -1, f"Failed to locate {self.text} {matched_span} in {toks} {offsets}"
        #return idx_start, idx_end
        self.start, self.end = idx_start, idx_end      

    def string_match(self, offsets, toks):
        sent_text = recover_sent(toks, offsets)
        try:
            string_pattern = copy.deepcopy(self.text).replace('(', '\(').replace(')', '\)').replace('$', '\$').replace('*', '\*').replace('[', '\[').replace('+', '\+')
            indices_object = re.finditer(pattern=string_pattern, string=sent_text)
            indices = [(index.start(),index.end()) for index in indices_object]
            indices = [(s+offsets[0][0],e+offsets[0][0]) for (s,e) in indices]
            distances = [np.abs(s-self.start) for (s,e) in indices]
            if len(distances) > 0:
                matached_span = indices[np.argmin(distances)]
                return matached_span
            else:
                return self.start, self.end + 1
        except:
            print(f"Failed to match {self.text} in {sent_text}")
            

    
    def copy(self):
        """Makes a copy of itself.

        Returns:
            Span: a copy of itself."""
        return Span(self.start, self.end, self.text)

@dataclass
class Entity(Span):
    entity_id: str
    mention_id: str
    entity_type: str
    entity_subtype: str
    mention_type: str
    value: str = None

    def to_dict(self) -> Dict[str, Any]:
        """Converts instance variables to a dict.
        
        Returns:
            Dict: a dict of instance variables.
        """
        entity_dict = {
            'text': self.text,
            'entity_id': self.entity_id,
            'mention_id': self.mention_id,
            'start': self.start,
            'end': self.end,
            'entity_type': self.entity_type,
            'entity_subtype': self.entity_subtype,
            'mention_type': self.mention_type
        }
        if self.value:
            entity_dict['value'] = self.value
        return entity_dict  

@dataclass
class RelationArgument:
    mention_id: str
    role: str
    text: str

    def to_dict(self) -> Dict[str, Any]:
        """Converts instance variables to a dict.
        
        Returns:
            Dict[str, Any]: a dict of instance variables.
        """
        return {
            'mention_id': self.mention_id,
            'role': self.role,
            'text': self.text
        }

@dataclass
class Relation:
    relation_id: str
    relation_type: str
    relation_subtype: str
    arg1: RelationArgument
    arg2: RelationArgument

    def to_dict(self) -> Dict[str, Any]:
        """Converts instance variables to a dict.
        
        Returns:
            Dict[str, Any]: a dict of instance variables.
        """
        return {
            'relation_id': self.relation_id,
            'relation_type': self.relation_type,
            'relation_subtype': self.relation_subtype,
            'arg1': self.arg1.to_dict(),
            'arg2': self.arg2.to_dict(),
        }

@dataclass
class EventArgument:
    mention_id: str
    role: str
    text: str

    def to_dict(self) -> Dict[str, Any]:
        """Converts instance variables to a dict.
        
        Returns:
            Dict[str, Any]: a dict of instance variables.
        """
        return {
            'mention_id': self.mention_id,
            'role': self.role,
            'text': self.text,
        }

@dataclass
class Event:
    event_id: str
    mention_id: str
    event_type: str
    event_subtype: str
    #start: int
    #end: int
    #text: str
    trigger: Span
    arguments: List[EventArgument]

    def to_dict(self) -> Dict[str, Any]:
        """Converts instance variables to a dict.
        
        Returns:
            Dict[str, Any]: a dict of instance variables.
        """
        return {
            'event_id': self.event_id,
            'mention_id': self.mention_id,
            'event_type': self.event_type,
            'event_subtype': self.event_subtype,
            #'start': self.start,
            #'end': self.end,
            'trigger': self.trigger.to_dict(),
            'arguments': [arg.to_dict() for arg in self.arguments],
        }

@dataclass
class Sentence(Span):
    sent_id: str
    tokens: List[str]
    entities: List[Entity]
    relations: List[Relation]
    events: List[Event]

    def to_dict(self) -> Dict[str, Any]:
        """Converts instance variables to a dict.
        
        Returns:
            Dict[str, Any]: a dict of instance variables.
        """
        return {
            'sent_id': self.sent_id,
            'tokens': [t for t in self.tokens],
            'entities': [entity.to_dict() for entity in self.entities],
            'relations': [relation.to_dict() for relation in self.relations],
            'events': [event.to_dict() for event in self.events],
            'start': self.start,
            'end': self.end,
            'text': self.text.replace('\t', ' '),
        }

@dataclass
class Document:
    doc_id: str
    sentences: List[Sentence]

    def to_dict(self) -> Dict[str, Any]:
        """Converts instance variables to a dict.
        
        Returns:
            Dict[str, Any]: a dict of instance variables.
        """
        return {
            'doc_id': self.doc_id,
            'sentences': [sent.to_dict() for sent in self.sentences]
        }

def read_apf_file(path: str,
                  time_and_val: bool = False,
                  head_only: bool = True
                 ) -> Tuple[str, str, List[Entity], List[Relation], List[Event]]:
    """Reads an APF file.

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
    root = soup.find('source_file')
    source = root['SOURCE']
    doc = root.find('document')
    doc_id = doc['DOCID']

    entity_list, relation_list, event_list = [], [], []

    # entities: nam, nom, pro
    for entity in doc.find_all('entity'):
        entity_id = entity['ID']
        entity_type = entity['TYPE']
        entity_subtype = entity['SUBTYPE']
        for entity_mention in entity.find_all('entity_mention'):
            mention_id = entity_mention['ID']
            mention_type = entity_mention['TYPE']
            if head_only:
                head = entity_mention.find('head').find('charseq')
                start, end, text = int(head['START']), int(head['END']), head.text
            else:
                extent = entity_mention.find('extent').find('charseq')
                start, end, text = int(extent['START']), int(extent['END']), extent.text
            entity_list.append(Entity(start, end, text,
                                      entity_id, mention_id, entity_type,
                                      entity_subtype, mention_type))

    if time_and_val:
        # entities: value
        for entity in doc.find_all('value'):
            entity_id = entity['ID']
            entity_type = entity['TYPE']
            entity_subtype = entity.get('SUBTYPE', None)
            for entity_mention in entity.find_all('value_mention'):
                mention_id = entity_mention['ID']
                mention_type = 'VALUE'
                extent = entity_mention.find('extent').find('charseq')
                start, end, text = int(extent['START']), int(extent['END']), extent.text
                entity_list.append(Entity(start, end, text,
                                          entity_id, mention_id, entity_type,
                                          entity_subtype, mention_type))

        # entities: timex
        for entity in doc.find_all('timex2'):
            entity_id = entity['ID']
            enitty_type = entity_subtype = 'TIME'
            value = entity.get('VAL', None)
            for entity_mention in entity.find_all('timex2_mention'):
                mention_id = entity_mention['ID']
                mention_type = 'TIME'
                extent = entity_mention.find('extent').find('charseq')
                start, end, text = int(extent['START']), int(extent['END']), extent.text
                entity_list.append(Entity(start, end, text,
                                          entity_id, mention_id, entity_type,
                                          entity_subtype, mention_type,
                                          value=value))

    # relations
    for relation in doc.find_all('relation'):
        relation_id = relation['ID']
        relation_type = relation['TYPE']
        if relation_type == 'METONYMY':
            continue
        relation_subtype = relation['SUBTYPE']
        for relation_mention in relation.find_all('relation_mention'):
            mention_id = relation_mention['ID']
            arg1 = arg2 = None
            for arg in relation_mention.find_all('relation_mention_argument'):
                arg_mention_id = arg['REFID']
                arg_role = arg['ROLE']
                arg_text = arg.find('extent').find('charseq').text
                if arg_role == 'Arg-1':
                    arg1 = RelationArgument(arg_mention_id, arg_role, arg_text)
                elif arg_role == 'Arg-2':
                    arg2 = RelationArgument(arg_mention_id, arg_role, arg_text)
            if arg1 and arg2:
                relation_list.append(Relation(mention_id, relation_type,
                                              relation_subtype, arg1, arg2))

    # events
    for event in doc.find_all('event'):
        event_id = event['ID']
        event_type = event['TYPE']
        event_subtype = event['SUBTYPE']
        event_modality = event['MODALITY']
        event_polarity = event['POLARITY']
        event_genericity = event['GENERICITY']
        event_tense = event['TENSE']
        for event_mention in event.find_all('event_mention'):
            mention_id = event_mention['ID']
            trigger = event_mention.find('anchor').find('charseq')
            trigger_start, trigger_end = int(trigger['START']), int(trigger['END'])
            trigger_text = trigger.text
            event_args = []
            for arg in event_mention.find_all('event_mention_argument'):
                arg_mention_id = arg['REFID']
                arg_role = arg['ROLE']
                arg_text = arg.find('extent').find('charseq').text
                event_args.append(EventArgument(arg_mention_id, arg_role, arg_text))
            event_list.append(Event(event_id, mention_id,
                                    event_type, event_subtype,
                                    Span(trigger_start, trigger_end, trigger_text),
                                    event_args))


    return doc_id, source, entity_list, relation_list, event_list

def correct_offsets(sents, offsets):
    """
    Update the offets after sentence tokenization with corenlp
    Args:
        sents: tokenized sents
        offsets: sentence offsets
    
    Return:
        sents
        offsets: updated offsets of token starting with "<" to (0, 0), for further processing
    
    """
    new_offsets = []
    minus = 0
    for i, offsets_per_sentence in enumerate(offsets):
        sentence = sents[i]
        new_offsets_per_sentence = []
        for j, offset in enumerate(offsets_per_sentence):
            if sentence[j].startswith('<'):
                new_offsets_per_sentence.append((0, 0))
                minus += len(sentence[j])
            else:
                new_offsets_per_sentence.append((offset[0]-minus, offset[1]-minus))
        new_offsets.append(new_offsets_per_sentence)
    return sents, new_offsets

def recover_sent(tokens, offsets):
    """
    recover sentence from a list of tokens and a list of token offsets
    """
    sent = ''
    for i in range(len(tokens)-1):
        sent += (tokens[i] + ' '*(offsets[i+1][0]-offsets[i][1]))
    sent += tokens[-1]
    return sent

def retokenize(sentences, offsets, corenlp_path, lang):
    nlp = StanfordCoreNLPv2(corenlp_path, lang=lang)
    new_sents = []
    new_offsets = []
    for i, (sent, offset) in enumerate(zip(sentences, offsets)):
        start = offset[0][0]
        sent_text = recover_sent(sent, offset)
        _subsents, _suboffsets = nlp.sent_tokenize(sent_text)
        for i, (_subsent, _suboffset) in enumerate(zip(_subsents, _suboffsets)):
            _suboffset = [(s+start, e+start) for (s,e) in _suboffset]
            new_sents.append(_subsent)
            new_offsets.append(_suboffset)
    nlp.close()
    return new_sents, new_offsets

def get_dir(path):
    patterns = ['/bc/', '/bn/', '/cts/', '/nw/', '/un/', '/wl/']
    for i,pattern in enumerate(patterns):
        if pattern in path:
            return DIRS[i]


def read_sgm_file(path: str,
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
        text = f.read()
    sents, offsets = nlp.sent_tokenize(text)
    nlp.close()

    sents, offsets = correct_offsets(sents, offsets)


    _dir = get_dir(path)
    mark_split_tag = SPLIT_TAGS[_dir]

    if _dir == 'cst':
        sents = sents[1:]
        offsets = offsets[1:]

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
            elif token.startswith('<'):
                select = False 
        
        subsent = sent[start_posi:]
        suboffset = offset_per_sentence[start_posi:]
        if select and len(subsent) > 0:
            assert (0, 0) not in suboffset
            new_sents.append(subsent)
            new_offsets.append(suboffset)
    
    if language != 'en':
        _new_sents, _new_offsets = retokenize(new_sents, new_offsets, corenlp_path, language)
        return _new_sents, _new_offsets

    return new_sents, new_offsets

def recover_tags(sent, offset):
    new_sent = []
    new_offset = []
    
    found_tags = []
    
    left = None
    
    for i in range(len(sent)):
        
        if sent[i] == '>' and left != None:
            found_tags.append((left, i))
            left = None
            
        if sent[i] == '<' or sent[i] == '</':
            left = i
    
    found_tags = sorted(found_tags, key=lambda x: x[0])
    
    l = 0
    
    for s,t in found_tags:
        
        for i in range(l, s):
            new_sent.append(sent[i])
            new_offset.append(offset[i])
        
        _tok = recover_sent(sent[s: t+1], offset[s: t+1])
        _offset = (offset[s][0], offset[t][-1])
        new_sent.append(_tok)
        new_offset.append(_offset)
        
        l = t+1
    
    if l <= len(sent)-1:
        for i in range(l, len(sent)):
            new_sent.append(sent[i])
            new_offset.append(offset[i])
    
    return new_sent, new_offset

def recover_tags_all(sents, offsets):
    new_sents = []
    new_offsets = []
    for i, (sent, offset) in enumerate(zip(sents, offsets)):
        new_sent, new_offset = recover_tags(sent, offset)
        new_sents.append(new_sent)
        new_offsets.append(new_offset)
    return new_sents, new_offsets


def process_entities(entities: List[Entity],
                     sentences: Tuple[List, List]
                    ) -> List[List[Entity]]:
    """Cleans entities and splits them into lists

    Args:
        entities (List[Entity]): a list of Entity instances.
        sentences (Tuple(list, list)): sentences, offsets

    Returns:
        List[List[Entity]]: a list of sentence entity lists.
    """

    sents, offsets = sentences
    sentence_entities = [[] for _ in range(len(sents))]
    unassigned = []
    # assign each entity to the sentence where it appears
    for entity in entities:
        assigned = False
        start, end = entity.start, entity.end
        #for i in range(len(sents)):
        for i, (sent, offset) in enumerate(zip(sents, offsets)):
            if start >= offset[0][0] and end < offset[-1][-1] and entity.text in recover_sent(sent, offset):
                sentence_entities[i].append(entity)
                assigned = True
                break 
        if not assigned:
            unassigned.append(entity)
    # remove duplicate entities
    sentence_entities_cleaned = [[] for _ in range(len(sents))]
    map_offset = [[] for _ in range(len(sentence_entities))]
    for i, entities in enumerate(sentence_entities):
        if not entities:
            continue
        for entity in entities:
            _offset = (entity.start, entity.end)
            if _offset not in map_offset[i]:
                map_offset[i].append(_offset)
                sentence_entities_cleaned[i].append(entity)

    return sentence_entities, unassigned


def process_events(events: List[Event],
                   sentence_entities: List[List[Entity]],
                   sentences:  Tuple[List, List]
                  ) -> List[List[Event]]:
    """Cleans and assigns events.

    Args:
        events (List[Event]): A list of Event objects
        entence_entities (List[List[Entity]]): A list of sentence entity lists.
        sentences (Tuple(list, list)): sentences, offsets
    
    Returns:
        List[List[Event]]: a list of sentence event lists.
    """
    sents, offsets = sentences
    sentence_events = [[] for _ in range(len(sents))]
    # assign each event mention to the sentence where it appears
    for event in events:
        start, end = event.trigger.start, event.trigger.end
        for i in range(len(sents)):
            s = offsets[i][0][0]
            e = offsets[i][-1][-1]
            sent_entities = sentence_entities[i]
            if start >= s and end <= e and event.trigger.text in recover_sent(sents[i], offsets[i]):
                # clean the argument list
                arguments = []
                for argument in event.arguments:
                    # entity_id = argument.entity_id
                    mention_id = argument.mention_id
                    for entity in sent_entities:
                        if entity.mention_id == mention_id:
                            arguments.append(argument)
                            break
                event_cleaned = Event(event.event_id, event.mention_id,
                                      event.event_type, event.event_subtype,
                                      trigger=event.trigger.copy(),
                                      arguments=arguments)
                sentence_events[i].append(event_cleaned)
    


    return sentence_events

def process_relation(relations: List[Relation],
                     sentence_entities: List[List[Entity]],
                     sentences:  Tuple[List, List]
                    ) -> List[List[Relation]]:
    """Cleans and assigns relations

    Args:
        relations (List[Relation]): a list of Relation instances.
        sentence_entities (List[List[Entity]]): a list of sentence entity lists.
        sentences (Tuple(list, list)): sentences, offsets

    Returns:
        List[List[Relation]]: a list of sentence relation lists.
    """
    sents, offsets = sentences
    sentence_relations = [[] for _ in range(len(sents))]
    for relation in relations:
        mention_id1 = relation.arg1.mention_id
        mention_id2 = relation.arg2.mention_id
        for i, entities in enumerate(sentence_entities):
            arg1_in_sent = any([mention_id1 == e.mention_id for e in entities])
            arg2_in_sent = any([mention_id2 == e.mention_id for e in entities])
            if arg1_in_sent and arg2_in_sent:
                sentence_relations[i].append(relation)
                break
            elif arg1_in_sent != arg2_in_sent:
                break
    return sentence_relations

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


def convert(sgm_file: str,
            apf_file: str,
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
    sentences = read_sgm_file(sgm_file, corenlp_path=corenlp_path, language=language)
    doc_id, source, entities, relations, events = read_apf_file(
        apf_file, time_and_val=time_and_val, head_only=head_only)
    
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
    if language == 'en':
        sgm_files = glob.glob(os.path.join(
            input_path, '**', 'timex2norm', '*.sgm'))
    elif language == 'zh':
        sgm_files = glob.glob(os.path.join(
            input_path, '**', 'adj', '*.sgm'))      
    else:
        raise ValueError('Unknown language: {}'.format(language))

    print('Converting the dataset to JSON format')
    print('#SGM files: {}'.format(len(sgm_files)))       
       

    all_unassigned = []
    with open(output_path, 'w', encoding='utf-8') as w:
        for sgm_file in tqdm.tqdm(sgm_files):

            apf_file = sgm_file.replace('.sgm', '.apf.xml')
            doc, unassigned = convert(sgm_file, apf_file, corenlp_path, time_and_val=time_and_val, head_only=head_only,
                          language=language)
            all_unassigned.append(unassigned)
            w.write(json.dumps(doc.to_dict(), ensure_ascii=False) + '\n')
    
    with open(f'{output_path}-unassigned', 'w', encoding='utf-8') as w:
        json.dump(all_unassigned, w, ensure_ascii=False)




def update_offset(input_file: str,
                 output_file: str):
    """Update offset from token offset to char offset

    Args:
        input_file (str): path to the input file.
        output_file (str): path to the output file.
    """
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:

        for line in f:
            _doc = json.loads(line)
            for sentence in _doc['sentences']:
                sent = {
                    'sent_id': sentence['sent_id'],
                    'doc_id': _doc['doc_id'],
                    'text': sentence['text']
                }

                lens = [len(t) for t in sentence['tokens']]
                _entities = sentence['entities']
                _relations = sentence['relations']
                _events = sentence['events']
                
                entities = []
                for _entity in _entities:
                    _s, _e = new_span(lens, _entity['start'], _entity['end'])
                    _entity['start'] = _s
                    _entity['end'] = _e
                    entities.append([[_entity['text']], [span_to_string(_s, _e)], _entity['entity_type']])
                
                relations = []
                for _relation in _relations:
                    _arg1 = [e for e in _entities if _relation['arg1']['mention_id'] == e['mention_id']][0]
                    _arg2 = [e for e in _entities if _relation['arg2']['mention_id'] == e['mention_id']][0]
                    relations.append([[span_to_string(_arg1['start'], _arg1['end'])], [span_to_string(_arg2['start'], _arg2['end'])], _relation['relation_type']])
                
                events = []
                for _event in _events:
                    event = {
                        'event_type': _event['event_subtype']
                    }
                    _s, _e = new_span(lens, _event['trigger']['start'], _event['trigger']['end'])

                    event['trigger'] = [[_event['trigger']['text']], [span_to_string(_s, _e)]]
                    event['arguments'] = []

                    for _arg in _event['arguments']:
                        loc_ent = [e for e in _entities if _arg['mention_id'] == e['mention_id']][0]
                        _s, _e = loc_ent['start'], loc_ent['end']
                        event['arguments'].append([[loc_ent['text']], [span_to_string(_s, _e)], _arg['role']])
                    
                    events.append(event)
                
                sent['entities'] = entities
                sent['relations'] = relations
                sent['events'] = events
                

                data.append(sent)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
                        

def new_span(lens, start, end):
    # update the span from token idx (inclusive)  to char idx
    # lens: list of length of each token
    new_start = sum(lens[:start]) + start
    new_end = new_start + sum(lens[start:end+1]) + (end-start)
    return new_start, new_end

def span_to_string(start, end):
    return f"{start}:{end}"

def load_json(file):
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def dump_json(file, data, indent=0):
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def split_data(input_file, output_dir, split_path):
    """Splits the input file into train/dev/test sets.

    Args:
        input_file (str): path to the input file.
        output_dir (str): path to the output directory.
        split_path (str): path to the split directory that contains three files,
            train.doc.txt, dev.doc.txt, and test.doc.txt . Each line in these
            files is a document ID.
    """
    print('Splitting the dataset into train/dev/test sets')
    train_docs, dev_docs, test_docs = set(), set(), set()
    # Load doc ids
    with open(os.path.join(split_path, 'train.doc.txt')) as r:
        train_docs.update(r.read().strip('\n').split('\n'))
    with open(os.path.join(split_path, 'dev.doc.txt')) as r:
        dev_docs.update(r.read().strip('\n').split('\n'))
    with open(os.path.join(split_path, 'test.doc.txt')) as r:
        test_docs.update(r.read().strip('\n').split('\n'))
    
    # split the dataset
    train_set = []
    dev_set = []
    test_set = []

    data = load_json(input_file)
    for sentence in data:
        if sentence['doc_id'] in train_docs:
            train_set.append(sentence)
        elif sentence['doc_id'] in dev_docs:
            dev_set.append(sentence)
        else:
            test_set.append(sentence)
    
    dump_json(os.path.join(output_dir, 'train.json'), train_set)
    dump_json(os.path.join(output_dir, 'dev.json'), dev_set)
    dump_json(os.path.join(output_dir, 'test.json'), test_set)

    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', help='Path to the input folder', default='~/Downloads/ace_2005_td_v7/data')
    parser.add_argument('-o', '--output', help='Path to the output folder', default='../test_preprocess')
    parser.add_argument('-s', '--split', default=None,
                        help='Path to the split folder')
    parser.add_argument('-d',
                        '--dataset',
                        help='Dataset type: ace_evt, ace_ent, ere',
                        default='ace_evt')
    parser.add_argument('-l', '--lang', default='en',
                        help='Document language')
    parser.add_argument('--time_and_val', action='store_true',
                        help='Extracts times and values', default=False)
    parser.add_argument('--head_only', action='store_true',
                        help='Extracts only the head of entity', default=True)
    parser.add_argument('--corenlp', action='store_true',
                        help='corenlp path', default='~/Downloads/ace_2005_td_v7/stanford-corenlp-full-2018-02-27')

    args = parser.parse_args()
    if args.lang not in ['zh', 'en']:
        raise ValueError('Unsupported language: {}'.format(args.lang))
    langs={
        'en': 'English',
        'zh': 'Chinese'
    }
    input_dir = os.path.join(args.input, langs[args.lang])
    print(input_dir)

    # Convert to doc-level JSON format
    json_path = os.path.join(args.output, '{}.json'.format(args.lang))
    convert_batch(input_dir, json_path, args.corenlp, time_and_val=args.time_and_val,
                  head_only=args.head_only, language=args.lang)   

    raw_path = os.path.join(args.output, '{}.{}.raw.json'.format(args.lang,
                                                                     args.dataset))
    update_offset(json_path, raw_path)
    split_data(raw_path, args.output, args.split)
