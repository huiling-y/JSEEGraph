import yaml
import copy

class Params:
    def __init__(self):
        self.graph_mode = "evt-ent"             # possibilities: {evt-ent, evt, ent}
        self.accumulation_steps = 1                  # number of gradient accumulation steps for achieving a bigger batch_size
        self.activation = "relu"                     # transformer (decoder) activation function, supported values: {'relu', 'gelu', 'sigmoid', 'mish'}
        self.balance_loss_weights = True             # use weight loss balancing (GradNorm)        self.predict_intensity = False
        self.batch_size = 16                         # batch size (further divided into multiple GPUs)
        self.beta_2 = 0.98                           # beta 2 parameter for Adam(W) optimizer
        self.blank_weight = 1.0                      # weight of cross-entropy loss for predicting an empty label
        self.char_embedding = False                   # use character embedding in addition to bert
        self.char_embedding_size = 128               # dimension of the character embedding layer in the character embedding module
        self.decoder_delay_steps = 0                 # number of initial steps with frozen decoder
        self.decoder_learning_rate = 6e-4            # initial decoder learning rate
        self.decoder_weight_decay = 1.2e-6           # amount of weight decay
        self.dropout_anchor = 0.5                    # dropout at the last layer of anchor classifier
        self.dropout_edge_label = 0.5                # dropout at the last layer of edge label classifier
        self.dropout_edge_presence = 0.5             # dropout at the last layer of edge presence classifier
        self.dropout_label = 0.5                     # dropout at the last layer of label classifier
        self.dropout_transformer = 0.5               # dropout for the transformer layers (decoder)
        self.dropout_transformer_attention = 0.1     # dropout for the transformer's attention (decoder)
        self.dropout_word = 0.1                      # probability of dropping out a whole word from the encoder (in favour of char embedding)
        self.encoder = "xlm-roberta-base"            # pretrained encoder model
        self.encoder_delay_steps = 2000              # number of initial steps with frozen XLM-R
        self.encoder_freeze_embedding = True         # freeze the first embedding layer in XLM-R
        self.encoder_learning_rate = 6e-5            # initial encoder learning rate
        self.encoder_weight_decay = 1e-2             # amount of weight decay
        self.lr_decay_multiplier = 100
        self.epochs = 100                            # number of epochs for train
        self.focal = True                            # use focal loss for the label prediction
        self.grad_norm_alpha = 1.5                   # grad-norm sensitivity
        self.grad_norm_lr = 1e-3                     # learning rate for the grad-norm optimizer
        self.group_ops = False                       # group 'opN' edge labels into one
        self.hidden_size_ff = 4 * 768                # hidden size of the transformer feed-forward submodule
        self.hidden_size_anchor = 128                # hidden size anchor biaffine layer
        self.hidden_size_edge_label = 256            # hidden size for edge label biaffine layer
        self.hidden_size_edge_presence = 512         # hidden size for edge label biaffine layer
        self.layerwise_lr_decay = 1.0                # layerwise decay of learning rate in the encoder
        self.n_attention_heads = 8                   # number of attention heads in the decoding transformer
        self.n_layers = 3                            # number of layers in the decoder
        self.query_length = 4                        # number of queries genereted for each word on the input
        self.pre_norm = True                         # use pre-normalized version of the transformer (as in Transformers without Tears)
        self.warmup_steps = 6000                     # number of the warm-up steps for the inverse_sqrt scheduler
    

    def init_data_paths(self):
        #directory_1 = {
        #    "evt-ent": "evt_ent",
        #    "evt": "evt",
        #    "ent": "ent"
        #}[self.graph_mode]
        #
        #directory_2 = {
        #    'ace_p': 'ace_p',
        #    'ace_pp': 'ace_pp',           
        #    'ere_p': 'ere_p',           
        #    'ere_pp': 'ere_pp',           
#
        #}[self.framework]
#
        ##raw_dir = {
        ##    "labeled-edge": "raw"
        ##}[self.graph_mode]
#
        #self.training_data = f"{self.data_directory}/ie_graph_mrp/{directory_2}_{directory_1}/{self.language}/train.mrp"
        #self.validation_data = f"{self.data_directory}/ie_graph_mrp/{directory_2}_{directory_1}/{self.language}/dev.mrp"       
        #self.test_data = f"{self.data_directory}/ie_graph_mrp/{directory_2}_{directory_1}/{self.language}/test.mrp"
#
        #self.raw_training_data = f"{self.data_directory}/raw/{directory_2}/{self.language}/train.json"
        #self.raw_validation_data = f"{self.data_directory}/raw/{directory_2}/{self.language}/dev.json"
        #self.raw_testing_data = f"{self.data_directory}/raw/{directory_2}/{self.language}/test.json"

        framework_lang_pairs = [
            ('ace_p_ent', 'en'), ('ace_p_ent', 'zh'), 
            ('ace_p_evt', 'en'), ('ace_p_evt', 'zh'),
            ('ace_p_evt_ent', 'en'), ('ace_p_evt_ent', 'zh'),
            ('ace_pp_ent', 'en'), ('ace_pp_ent', 'zh'), 
            ('ace_pp_evt', 'en'), ('ace_pp_evt', 'zh'),
            ('ace_pp_evt_ent', 'en'), ('ace_pp_evt_ent', 'zh'),  
            ('ere_p_ent', 'en'), ('ere_p_ent', 'zh'), ('ere_p_ent', 'es'),
            ('ere_p_evt', 'en'), ('ere_p_evt', 'zh'), ('ere_p_evt', 'es'),
            ('ere_p_evt_ent', 'en'), ('ere_p_evt_ent', 'zh'), ('ere_p_evt_ent', 'es'),
            ('ere_pp_ent', 'en'), ('ere_pp_ent', 'zh'), ('ere_pp_ent', 'es'),
            ('ere_pp_evt', 'en'), ('ere_pp_evt', 'zh'), ('ere_pp_evt', 'es'),
            ('ere_pp_evt_ent', 'en'), ('ere_pp_evt_ent', 'zh'), ('ere_pp_evt_ent', 'es')
        ]

        framework_lang_raw_dir = {
            ('ace_p_ent', 'en'): "ace_p", ('ace_p_ent', 'zh'): "ace_p", 
            ('ace_p_evt', 'en'): "ace_p", ('ace_p_evt', 'zh'): "ace_p",
            ('ace_p_evt_ent', 'en'): "ace_p", ('ace_p_evt_ent', 'zh'): "ace_p",
            ('ace_pp_ent', 'en'): "ace_pp", ('ace_pp_ent', 'zh'): "ace_pp", 
            ('ace_pp_evt', 'en'): "ace_pp", ('ace_pp_evt', 'zh'): "ace_pp",
            ('ace_pp_evt_ent', 'en'): "ace_pp", ('ace_pp_evt_ent', 'zh'): "ace_pp",  
            ('ere_p_ent', 'en'): "ere_p", ('ere_p_ent', 'zh'): "ere_p", ('ere_p_ent', 'es'): "ere_p",
            ('ere_p_evt', 'en'): "ere_p", ('ere_p_evt', 'zh'): "ere_p", ('ere_p_evt', 'es'): "ere_p",
            ('ere_p_evt_ent', 'en'): "ere_p", ('ere_p_evt_ent', 'zh'): "ere_p", ('ere_p_evt_ent', 'es'): "ere_p",
            ('ere_pp_ent', 'en'): "ere_pp", ('ere_pp_ent', 'zh'): "ere_pp", ('ere_pp_ent', 'es'): "ere_pp",
            ('ere_pp_evt', 'en'): "ere_pp", ('ere_pp_evt', 'zh'): "ere_pp", ('ere_pp_evt', 'es'): "ere_pp",
            ('ere_pp_evt_ent', 'en'): "ere_pp", ('ere_pp_evt_ent', 'zh'): "ere_pp", ('ere_pp_evt_ent', 'es'): "ere_pp"
        }

        # path to the training datasets

        self.training_data = {}
        self.validation_data = {}
        self.test_data = {}
        self.raw_training_data = {}
        self.raw_validation_data = {}
        self.raw_test_data = {}

        for f, l in framework_lang_pairs:
            raw_dir = framework_lang_raw_dir[(f, l)]
            self.training_data[(f, l)] = f"{self.data_directory}/ie_graph_mrp/{f}/{l}/train.mrp"
            self.validation_data[(f, l)] = f"{self.data_directory}/ie_graph_mrp/{f}/{l}/dev.mrp"
            self.test_data[(f, l)] = f"{self.data_directory}/ie_graph_mrp/{f}/{l}/test.mrp"

            self.raw_training_data[(f, l)] = f"{self.data_directory}/raw/{raw_dir}/{l}/train.json"
            self.raw_validation_data[(f, l)] = f"{self.data_directory}/raw/{raw_dir}/{l}/dev.json"
            self.raw_test_data[(f, l)] = f"{self.data_directory}/raw/{raw_dir}/{l}/test.json"
        return self

    def load_state_dict(self, d):
        for k, v in d.items():
            setattr(self, k, v)
        return self

    def state_dict(self):
        members = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
        return {k: self.__dict__[k] for k in members}

    def load(self, args):
        with open(args.config, "r", encoding="utf-8") as f:
            params = yaml.safe_load(f)
            self.load_state_dict(params)
        self.init_data_paths()

    def save(self, json_path):
        with open(json_path, "w", encoding="utf-8") as f:
            d = self.state_dict()
            yaml.dump(d, f)

    def get_hyperparameters(self):
        clone = copy.copy(self)
        del clone.training_data
        del clone.validation_data
        del clone.test_data
        del clone.raw_training_data
        del clone.raw_validation_data
        del clone.raw_test_data
        del clone.frameworks
        return clone