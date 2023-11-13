#!/usr/bin/env python3
# coding=utf-8

from data.parser.to_mrp.abstract_parser import AbstractParser
from itertools import chain

# ACE specific
# Event types: 33 types

ace_event_types = ['Be-Born', 'Die', 'Marry', 'Divorce', 'Injure', 'Transfer-Ownership', 'Transfer-Money', \
    'Transport', 'Start-Org', 'End-Org', 'Declare-Bankruptcy', 'Merge-Org', 'Attack', 'Demonstrate', 'Meet', \
        'Phone-Write', 'Start-Position', 'End-Position', 'Nominate', 'Elect', 'Arrest-Jail', 'Release-Parole', \
            'Charge-Indict', 'Trial-Hearing', 'Sue', 'Convict', 'Sentence', 'Fine', 'Execute', 'Extradite', \
                'Acquit', 'Pardon', 'Appeal']

# Event type argument pair: full argument set

ace_pairs_full = {
    "Attack": ['Instrument', 'Time-Before', 'Victim', 'Time-At-End', 'Time-After', 'Time-Starting', 'Place', 'Agent', 'Target', 'Time-Within', 'Attacker', 'Time-Ending', 'Time-At-Beginning', 'Time-Holds'],
    "Transport": ['Artifact', 'Vehicle', 'Victim', 'Time-Before', 'Origin', 'Time-At-End', 'Destination', 'Time-After', 'Time-Starting', 'Agent', 'Place', 'Time-Within', 'Time-Ending', 'Time-At-Beginning', 'Time-Holds'],
    "Die": ['Instrument', 'Victim', 'Time-Before', 'Time-After', 'Time-Starting', 'Agent', 'Place', 'Time-Within', 'Time-Ending', 'Person', 'Time-At-Beginning', 'Time-Holds'],
    "Meet": ['Entity', 'Time-After', 'Time-Starting', 'Place', 'Time-Within', 'Time-Ending', 'Time-At-Beginning', 'Time-Holds'],
    "End-Position": ['Entity', 'Time-Before', 'Time-At-End', 'Position', 'Time-After', 'Time-Starting', 'Place', 'Time-Within', 'Time-Ending', 'Person', 'Time-Holds'],
    "Transfer-Money": ['Time-Before', 'Time-After', 'Recipient', 'Money', 'Giver', 'Time-Starting', 'Place', 'Time-Within', 'Beneficiary', 'Time-Holds'],
    "Elect": ['Entity', 'Time-Before', 'Position', 'Time-Starting', 'Place', 'Time-Within', 'Person', 'Time-At-Beginning', 'Time-Holds'],
    "Injure": ['Victim', 'Instrument', 'Agent', 'Place', 'Time-Within'],
    "Transfer-Ownership": ['Artifact', 'Time-Before', 'Time-Ending', 'Buyer', 'Place', 'Time-Within', 'Seller', 'Price', 'Beneficiary', 'Time-At-Beginning'],
    "Phone-Write": ['Entity', 'Time-Before', 'Time-After', 'Time-Starting', 'Place', 'Time-Within', 'Time-Holds'],
    "Start-Position": ['Entity', 'Time-Before', 'Position', 'Time-After', 'Time-Starting', 'Place', 'Time-Within', 'Person', 'Time-At-Beginning', 'Time-Holds'],
    "Trial-Hearing": ['Defendant', 'Crime', 'Time-At-End', 'Time-Starting', 'Prosecutor', 'Place', 'Time-Within', 'Adjudicator', 'Time-Holds'],
    "Charge-Indict": ['Defendant', 'Crime', 'Time-Before', 'Prosecutor', 'Place', 'Time-Within', 'Adjudicator', 'Time-Ending'],
    "Sentence": ['Defendant', 'Crime', 'Time-At-End', 'Time-Starting', 'Place', 'Time-Within', 'Sentence', 'Adjudicator'],
    "Arrest-Jail": ['Crime', 'Time-Before', 'Time-Starting', 'Agent', 'Place', 'Time-Within', 'Time-Ending', 'Person', 'Time-At-Beginning', 'Time-Holds'],
    "Marry": ['Time-Before', 'Place', 'Time-Within', 'Person', 'Time-Holds'],
    "Demonstrate": ['Entity', 'Time-At-End', 'Time-Starting', 'Place', 'Time-Within'],
    "Sue": ['Defendant', 'Crime', 'Plaintiff', 'Place', 'Time-Within', 'Adjudicator', 'Time-Holds'],
    "Convict": ['Defendant', 'Crime', 'Place', 'Time-Within', 'Adjudicator', 'Time-At-Beginning'],
    "Be-Born": ['Place', 'Time-Within', 'Person', 'Time-Holds'],
    "Start-Org": ['Time-Before', 'Org', 'Time-After', 'Time-Starting', 'Agent', 'Place', 'Time-Within'],
    "Release-Parole": ['Entity', 'Crime', 'Time-After', 'Place', 'Time-Within', 'Person'],
    "Declare-Bankruptcy": ['Org', 'Time-After', 'Place', 'Time-Within', 'Time-At-Beginning'],
    "Appeal": ['Crime', 'Plaintiff', 'Place', 'Time-Within', 'Adjudicator', 'Time-Holds'],
    "End-Org": ['Org', 'Time-After', 'Place', 'Time-Within', 'Time-At-Beginning', 'Time-Holds'],
    "Divorce": ['Place', 'Time-Within', 'Person'],
    "Fine": ['Entity', 'Crime', 'Money', 'Place', 'Time-Within', 'Adjudicator'],
    "Execute": ['Crime', 'Time-After', 'Agent', 'Place', 'Time-Within', 'Person', 'Time-At-Beginning'],
    "Merge-Org": ['Time-Ending', 'Org'],
    "Nominate": ['Agent', 'Position', 'Time-Within', 'Person'],
    "Extradite": ['Origin', 'Destination', 'Agent', 'Time-Within', 'Person'],
    "Acquit": ['Time-Within', 'Defendant', 'Adjudicator', 'Crime'],
    "Pardon": ['Place', 'Defendant', 'Time-At-End', 'Adjudicator']
}

# Time and value arguments excluded

ace_pairs_short = {
    'Attack': ['Instrument', 'Victim', 'Place', 'Agent', 'Target', 'Attacker'],
    'Transport': ['Artifact', 'Vehicle', 'Victim', 'Origin', 'Destination', 'Agent', 'Place'],
    'Die': ['Instrument', 'Victim', 'Agent', 'Place', 'Person'],
    'Meet': ['Entity', 'Place'],
    'End-Position': ['Entity', 'Place', 'Person'],
    'Transfer-Money': ['Recipient', 'Giver', 'Place', 'Beneficiary'],
    'Elect': ['Entity', 'Place', 'Person'],
    'Injure': ['Victim', 'Instrument', 'Agent', 'Place'],
    'Transfer-Ownership': ['Artifact', 'Buyer', 'Place', 'Seller', 'Beneficiary'],
    'Phone-Write': ['Entity', 'Place'],
    'Start-Position': ['Entity', 'Place', 'Person'],
    'Trial-Hearing': ['Defendant', 'Prosecutor', 'Place', 'Adjudicator'],
    'Charge-Indict': ['Defendant', 'Prosecutor', 'Place', 'Adjudicator'],
    'Sentence': ['Defendant', 'Place', 'Adjudicator'],
    'Arrest-Jail': ['Agent', 'Place', 'Person'],
    'Marry': ['Place', 'Person'],
    'Demonstrate': ['Entity', 'Place'],
    'Sue': ['Defendant', 'Plaintiff', 'Place', 'Adjudicator'],
    'Convict': ['Defendant', 'Place', 'Adjudicator'],
    'Be-Born': ['Place', 'Person'],
    'Start-Org': ['Org', 'Agent', 'Place'],
    'Release-Parole': ['Entity', 'Place', 'Person'],
    'Declare-Bankruptcy': ['Org', 'Place'],
    'Appeal': ['Plaintiff', 'Place', 'Adjudicator'],
    'End-Org': ['Org', 'Place'],
    'Divorce': ['Place', 'Person'],
    'Fine': ['Entity', 'Place', 'Adjudicator'],
    'Execute': ['Agent', 'Place', 'Person'],
    'Merge-Org': ['Org'],
    'Nominate': ['Agent', 'Person'],
    'Extradite': ['Origin', 'Destination', 'Agent', 'Person'],
    'Acquit': ['Defendant', 'Adjudicator'],
    'Pardon': ['Place', 'Defendant', 'Adjudicator']
}

# ERE specific

# events: 38 types, 37 (wo declarebankruptcy)
# events: 18 (a shorter list of event types that exist in all three languages)
ere_event_types = ['arrestjail', 'artifact', 'attack', 'broadcast', 'contact', 'correspondence', 'demonstrate', 'die', 'elect', \
                   'endposition', 'injure', 'meet', 'startposition', 'transaction', 'transfermoney', 'transferownership', \
                    'transportartifact', 'transportperson']

# event type - argument role pairs
ere_pairs = {
    #'acquit': ['place', 'defendant', 'crime', 'time', 'adjudicator'],
    #'appeal': ['defendant', 'adjudicator', 'place', 'prosecutor'],
    'arrestjail': ['agent', 'place', 'time', 'crime', 'person'],
    'artifact': ['instrument', 'agent', 'place', 'time', 'artifact'],
    'attack': ['instrument', 'place', 'time', 'attacker', 'target'],
    #'beborn': ['time', 'place', 'person'],
    'broadcast': ['time', 'place', 'audience', 'entity'],
    #'chargeindict': ['prosecutor','place','defendant','crime','time','adjudicator'],
    'contact': ['time', 'place', 'entity'],
    #'convict': ['place', 'defendant', 'crime', 'time', 'adjudicator'],
    'correspondence': ['time', 'place', 'entity'],
    #'declarebankruptcy': ['time', 'org'],
    'demonstrate': ['time', 'place', 'entity'],
    'die': ['instrument', 'agent', 'place', 'time', 'victim'],
    #'divorce': ['time', 'person'],
    'elect': ['position', 'agent', 'place', 'time', 'person'],
    #'endorg': ['time', 'place', 'org'],
    'endposition': ['position', 'place', 'entity', 'time', 'person'],
    #'execute': ['agent', 'place', 'crime', 'time', 'person'],
    #'extradite': ['agent', 'destination', 'crime', 'time', 'person', 'origin'],
    #'fine': ['money', 'place', 'entity', 'crime', 'time', 'adjudicator'],
    'injure': ['instrument', 'agent', 'place', 'time', 'victim'],
    #'marry': ['time', 'place', 'person'],
    'meet': ['time', 'place', 'entity'],
    #'mergeorg': ['org'],
    #'nominate': ['agent', 'time', 'position', 'person'],
    #'pardon': ['place', 'defendant', 'crime', 'time', 'adjudicator'],
    #'releaseparole': ['agent', 'place', 'time', 'crime', 'person'],
    #'sentence': ['place','defendant', 'crime','time', 'adjudicator', 'sentence'],
    #'startorg': ['agent', 'time', 'place', 'org'],
    'startposition': ['position', 'place', 'entity', 'time', 'person'],
    #'sue': ['place', 'defendant', 'crime', 'time', 'adjudicator', 'plaintiff'],
    'transaction': ['place', 'time', 'giver', 'recipient', 'beneficiary'],
    'transfermoney': ['money', 'place', 'time', 'giver', 'recipient', 'beneficiary'],
    'transferownership': ['thing', 'place', 'time', 'giver', 'recipient', 'beneficiary'],
    'transportartifact': ['instrument', 'agent', 'destination', 'time', 'artifact', 'origin'],
    'transportperson': ['instrument', 'agent', 'destination', 'time', 'person', 'origin'],
    #'trialhearing': ['prosecutor', 'place', 'defendant', 'time', 'crime', 'adjudicator']
    }




class EvtParser(AbstractParser):
    def __init__(self, *args):
        super().__init__(*args)

        if len(self.dataset.edge_label_field.vocab) == 55:
            self.event_types = ace_event_types
            pairs = ace_pairs_short
        else:
            self.event_types = ere_event_types
            pairs = ere_pairs
        
        arg_roles = set(chain(*pairs.values()))
        for arg_role in arg_roles.copy():
            if arg_role in self.event_types:
                arg_roles.remove(arg_role)

        self.argument_roles = list(arg_roles)

        self.pairs = {}
        for k,v in pairs.items():
            self.pairs[k] = [self.dataset.edge_label_field.vocab.stoi[item] for item in v]
        
        self.argument_ids = [self.dataset.edge_label_field.vocab.stoi[arg] for arg in self.argument_roles]
        self.event_type_ids = [self.dataset.edge_label_field.vocab.stoi[e] for e in self.event_types]

    def parse(self, prediction):
        output = {}

        output["id"] = self.dataset.id_field.vocab.itos[prediction["id"].item()]
        output["nodes"] = self.create_nodes(prediction)
        output["nodes"] = self.create_anchors(prediction, output["nodes"], join_contiguous=True, at_least_one=True)
        output["nodes"] = [{"id": 0}] + output["nodes"]
        output["edges"] = self.create_edges(prediction, output["nodes"])

        return output

    def create_nodes(self, prediction):
        return [{"id": i + 1} for i, l in enumerate(prediction["labels"])]

    def create_edges(self, prediction, nodes):
        N = len(nodes)
        edge_prediction = prediction["edge presence"][:N, :N]

        edges = []
        event_nodes = []
        event_t = []
        for target in range(1, N):
            if edge_prediction[0, target] >= 0.5:
                for j in self.argument_ids:
                    prediction['edge labels'][0, target, j] = float('-inf')
                self.create_edge(0, target, prediction, edges, nodes)
                event_nodes.append(target)
                event_t.append(edges[-1]['label'])                

        for source in range(1, N):
            for target in range(1, N):
                if source == target:
                    continue
                if edge_prediction[source, target] < 0.5:
                    continue

                if source in event_nodes:
                    per_etype = event_t[event_nodes.index(source)]
                    candidates = self.pairs[per_etype]
                    for j in range(len(self.dataset.edge_label_field.vocab)):
                        if j not in candidates:
                            prediction['edge labels'][source, target, j] = float('-inf')

                    self.create_edge(source, target, prediction, edges, nodes)

        return edges

    def get_edge_label(self, prediction, source, target):
        return self.dataset.edge_label_field.vocab.itos[prediction["edge labels"][source, target].argmax(-1).item()]
