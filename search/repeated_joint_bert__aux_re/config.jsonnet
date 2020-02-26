local SEED = std.parseInt(std.extVar("ALLENNLP_SEED"));

local char_num_filters = 128;
local bert_model = "bert-base-uncased";
local ner_emb_dim = 100;
local encoder_hidden_dim = 187;
local pos_emb_dim = 30;

{
  "numpy_seed": SEED,
  "pytorch_seed": SEED,
  "random_seed": SEED,
  "dataset_reader": {
    "type": "jsonl_reader",
    "subtasks": [2, 3],
    "aux_re_task": true,
    "token_indexers": {
      "tokens": {
        "type": "bert-pretrained",
        "pretrained_model": bert_model,
        "do_lowercase": true,
        "use_starting_offsets": true,
        "truncate_long_sequences": false,
      },
      "characters": {
        "type": "characters",
        "min_padding_length": 3,
      },
      "pos_tokens": {
        "type": "pos_tag",
      }
    },
  },
  "train_data_path": "~/code/semeval_2020_task_6/data/deft_split/jsonl/train.jsonl",
  "validation_data_path": "~/code/semeval_2020_task_6/data/deft_split/jsonl/dev.jsonl",
  "model": {
    "type": "joint_classifier",
    "text_field_embedder": {
      "allow_unmatched_keys": true,
      "embedder_to_indexer_map": {
        "tokens": ["tokens", "tokens-offsets"],
        "characters": ["characters"],
        "pos_tokens": ["pos_tokens"],
      },
      "token_embedders": {
        "tokens": {
          "type": "bert-pretrained",
          "pretrained_model": bert_model,
        },
        "characters": {
          "type": "character_encoding",
          "embedding": {
            "embedding_dim": 16,
          },
          "encoder": {
            "type": "cnn",
            "embedding_dim": 16,
            "num_filters": char_num_filters,
            "ngram_filter_sizes": [3],
            "conv_layer_activation": "relu",
          },
        },
        "pos_tokens": {
          "type": "embedding",
          "embedding_dim": pos_emb_dim,
          "trainable": true,
          "vocab_namespace": "pos_tokens",
        },
      },
    },
    "ner_tag_embedder": {
      "type": "embedding",
      "embedding_dim": ner_emb_dim,
      "trainable": true,
      "vocab_namespace": "tags",
    },
    "encoder": {
      "type": "lstm",
      "input_size": 768 + char_num_filters + pos_emb_dim,
      "hidden_size": encoder_hidden_dim,
      "num_layers": 2,
      "bidirectional": true,
      "dropout": 0.3,
    },
    "relation_scorer": {
      "type": "auxiliary_relation_scorer",
      "input_size": 2 * encoder_hidden_dim + ner_emb_dim,
      "hidden_size": 100,
      "label_namespace": "relation_labels",
    },
    "evaluated_ner_labels": ['Term', 'Alias-Term', 'Referential-Term', 'Definition', 'Referential-Definition', 'Qualifier'],
  },
  "iterator": {
    "type": "basic",
    "batch_size": 25,
  },
  "trainer": {
    "optimizer": {
      "type": "adam",
      "lr": "0.0016610248112331916",
    },
    "num_serialized_models_to_keep": 1,
    "patience": 20,
    "cuda_device": 0,
    "validation_metric": "+ner_f1",
    "num_epochs": 100,
    "grad_clipping": 5.0,
  },
}
