from collections import Counter

import torch
from data.field.mini_torchtext.vocab import Vocab

from data.dataset import Dataset, Collate
from data.batch import Batch
from data.concat_dataset import ConcatDataset



class SharedDataset:
    def __init__(self, args):
        self.child_datasets = {
            (framework, language): Dataset(args) for framework, language in args.frameworks
        }
        self.framework_to_id = {(f, l): i for i, (f, l) in enumerate(args.frameworks)}
        self.id_to_framework = {i: (f, l) for i, (f, l) in enumerate(args.frameworks)}

    def log(self, text):
        print(text, flush=True)

    def load_state_dict(self, args, d):
        for key, dataset in self.child_datasets.items():
            dataset.load_state_dict(args, d[key])
        self.share_chars()

    def state_dict(self):
        return {key: dataset.state_dict() for key, dataset in self.child_datasets.items()}

    def load_sentences(self, sentences, args, framework: str, language: str):
        def switch(f, l, s):
            return s if (framework == f and language == l) else []

        datasets = [
            dataset.load_sentences(switch(f, l, sentences), args, language)
            for (f, l), dataset in self.child_datasets.items()
        ]
        return torch.utils.data.DataLoader(ConcatDataset(datasets), batch_size=1, shuffle=False, collate_fn=Collate())

    def load_datasets(self, args):
        for (framework, language), dataset in self.child_datasets.items():
            dataset.load_dataset(args, framework, language)

        self.share_chars()
        self.share_vocabs(args)

        # print each dataset stats after sharing vocab
        for i in range(len(self.child_datasets)):
            print(self.id_to_framework[i])
            print((f"{len(self.child_datasets[self.id_to_framework[i]].edge_label_field.vocab)} words in the edge label vocabulary"))
            print(list(self.child_datasets[self.id_to_framework[i]].edge_label_field.vocab.freqs.keys()), flush=True)
            print((f"{len(self.child_datasets[self.id_to_framework[i]].label_field.vocab)} words in the label vocabulary"))
            print(list(self.child_datasets[self.id_to_framework[i]].label_field.vocab.freqs.keys()), flush=True)
            print((f"{len(self.child_datasets[self.id_to_framework[i]].char_form_field.vocab)} characters in the vocabulary"))
            #print(list(self.child_datasets[self.id_to_framework[i]].char_form_field.vocab.freqs.keys()), flush=True)
        print('---------------------------------------------\n')

        train_datasets = [self.child_datasets[self.id_to_framework[i]].train for i in range(len(self.child_datasets))]
        self.train = torch.utils.data.DataLoader(
            ConcatDataset(train_datasets),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0, #args.workers,
            collate_fn=Collate(),
            pin_memory=True,
            drop_last=True
        )
        self.train_size = len(self.train.dataset)
        self.mean_label_length = sum(dataset.node_count for dataset in self.child_datasets.values()) / self.train_size

        val_datasets = [self.child_datasets[self.id_to_framework[i]].val for i in range(len(self.child_datasets))]
        self.val = torch.utils.data.DataLoader(
            ConcatDataset(val_datasets),
            batch_size=1,
            shuffle=False,
            num_workers=0, #args.workers,
            collate_fn=Collate(),
            pin_memory=True,
        )
        self.val_size = len(self.val.dataset)

        test_datasets = [self.child_datasets[self.id_to_framework[i]].test for i in range(len(self.child_datasets))]
        self.test = torch.utils.data.DataLoader(
            ConcatDataset(test_datasets),
            batch_size=1,
            shuffle=False,
            num_workers=0,#args.workers,
            collate_fn=Collate(),
            pin_memory=True,
        )
        self.test_size = len(self.test.dataset)

        self.log(f"\n{self.train_size} sentences in the train split")
        self.log(f"{self.val_size} sentences in the validation split")
        self.log(f"{self.test_size} sentences in the test split")

        #if gpu == 0:
        batch = next(iter(self.train))
        print(f"\nBatch content: {Batch.to_str(batch)}\n")
        print(flush=True)

    def share_chars(self):
        sos, eos, unk, pad = "<sos>", "<eos>", "<unk>", "<pad>"

        form_counter= Counter()
        for dataset in self.child_datasets.values():
            form_counter += dataset.char_form_field.vocab.freqs

        form_vocab = Vocab(form_counter, min_freq=1, specials=[pad, unk, sos, eos])


        for dataset in self.child_datasets.values():
            dataset.char_form_field.vocab = dataset.char_form_field.nesting_field.vocab = form_vocab


        self.char_form_vocab_size = len(form_vocab)


    def share_vocabs(self, args):


        ace_p_evt_datasets = [dataset for (f,l), dataset in self.child_datasets.items() if f == "ace_p_evt"]
        if len(ace_p_evt_datasets) == 2:
            print("Sharing ace_p_evt vocabs...")
            self.share_vocabs_(ace_p_evt_datasets, args, share_edges=True, share_anchors=True, share_labels=True)

        ace_p_evt_ent_datasets = [dataset for (f,l), dataset in self.child_datasets.items() if f == "ace_p_evt_ent"]
        if len(ace_p_evt_ent_datasets) == 2:
            print("Sharing ace_p_evt_ent vocabs...")
            self.share_vocabs_(ace_p_evt_ent_datasets, args, share_edges=True, share_anchors=True, share_labels=True)


        ace_pp_evt_datasets = [dataset for (f,l), dataset in self.child_datasets.items() if f == "ace_pp_evt"]
        if len(ace_pp_evt_datasets) == 2:
            print("Sharing ace_pp_evt vocabs...")
            self.share_vocabs_(ace_pp_evt_datasets, args, share_edges=True, share_anchors=True, share_labels=True)

        ace_pp_evt_ent_datasets = [dataset for (f,l), dataset in self.child_datasets.items() if f == "ace_pp_evt_ent"]
        if len(ace_pp_evt_ent_datasets) == 2:
            print("Sharing ace_pp_evt_ent vocabs...")
            self.share_vocabs_(ace_pp_evt_ent_datasets, args, share_edges=True, share_anchors=True, share_labels=True)


        ere_p_evt_datasets = [dataset for (f,l), dataset in self.child_datasets.items() if f == "ere_p_evt"]
        if len(ere_p_evt_datasets) == 3:
            print("Sharing ere_p_evt vocabs...")
            self.share_vocabs_(ere_p_evt_datasets, args, share_edges=True, share_anchors=True, share_labels=True)

        ere_p_evt_ent_datasets = [dataset for (f,l), dataset in self.child_datasets.items() if f == "ere_p_evt_ent"]
        if len(ere_p_evt_ent_datasets) == 3:
            print("Sharing ere_p_evt_ent vocabs...")
            self.share_vocabs_(ere_p_evt_ent_datasets, args, share_edges=True, share_anchors=True, share_labels=True)


        ere_pp_evt_datasets = [dataset for (f,l), dataset in self.child_datasets.items() if f == "ere_pp_evt"]
        if len(ere_pp_evt_datasets) == 3:
            print("Sharing ere_pp_evt vocabs...")
            self.share_vocabs_(ere_pp_evt_datasets, args, share_edges=True, share_anchors=True, share_labels=True)

        ere_pp_evt_ent_datasets = [dataset for (f,l), dataset in self.child_datasets.items() if f == "ere_pp_evt_ent"]
        if len(ere_pp_evt_ent_datasets) == 3:
            print("Sharing ere_pp_evt_ent vocabs...")
            self.share_vocabs_(ere_pp_evt_ent_datasets, args, share_edges=True, share_anchors=True, share_labels=True)


    def share_vocabs_(self, datasets, args, share_edges=False, share_anchors=False, share_labels=False):

        all_node_count = sum([dataset.node_count for dataset in datasets])
        all_token_count = sum([dataset.token_count for dataset in datasets])
        all_edge_count = sum([dataset.edge_count for dataset in datasets])
        all_no_edge_count = sum([dataset.no_edge_count for dataset in datasets])
        all_anchor_freq = sum([dataset.anchor_freq for dataset in datasets])


        for dataset in datasets:
            dataset.node_count = all_node_count
            dataset.token_count = all_token_count

        #share id field
        id_counter = datasets[0].id_field.vocab.freqs
        for dataset in datasets[1:]:
            id_counter += dataset.id_field.vocab.freqs
        
        for dataset in datasets:
            dataset.id_field.vocab = Vocab(id_counter, specials=[])

        if share_edges:

            edge_label_counter = datasets[0].edge_label_field.vocab.freqs
            for dataset in datasets[1:]:
                edge_label_counter += dataset.edge_label_field.vocab.freqs

            for dataset in datasets:
                dataset.edge_count = all_edge_count
                dataset.no_edge_count = all_no_edge_count            
                dataset.edge_label_field.vocab = Vocab(edge_label_counter, specials=[])

            for dataset in datasets:
                dataset.create_edge_freqs(args)

        if share_anchors:
            all_anchor_freq = sum([dataset.train_size * dataset.anchor_freq for dataset in datasets]) / sum([dataset.train_size for dataset in datasets])
            for datast in datasets:
                dataset.anchor_freq = all_anchor_freq


        if share_labels:
            label_counter = datasets[0].label_field.vocab.freqs
            for dataset in datasets[1:]:
                label_counter += dataset.label_field.vocab.freqs
            
            for dataset in datasets:
                dataset.label_field.vocab = Vocab(label_counter, specials=[])
                dataset.anchored_label_field.vocab = dataset.label_field.vocab
            
            for dataset in datasets:
                dataset.create_label_freqs(args)
                


