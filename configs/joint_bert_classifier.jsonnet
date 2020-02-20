local bert_model = "bert-base-uncased";
local ner_emb_dim = 100;
local encoder_hidden_dim = 200;
local SEED = 0;

{
  "numpy_seed": SEED,
  "pytorch_seed": SEED,
  "random_seed": SEED,
  "dataset_reader": {
    "type": "jsonl_reader",
    "subtasks": [2, 3],
    "token_indexers": {
      "tokens": {
        "type": "bert-pretrained",
        "pretrained_model": bert_model,
        "do_lowercase": true,
        "use_starting_offsets": true,
        "truncate_long_sequences": false,
      },
    },
  },
  "train_data_path": "data/deft_split/jsonl/train.jsonl",
  "validation_data_path": "data/deft_split/jsonl/dev.jsonl",
  "model": {
    "type": "joint_classifier",
    "text_field_embedder": {
      "allow_unmatched_keys": true,
      "embedder_to_indexer_map": {
        "tokens": ["tokens", "tokens-offsets"],
      },
      "token_embedders": {
        "tokens": {
          "type": "bert-pretrained",
          "pretrained_model": bert_model,
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
      "input_size": 768,
      "hidden_size": encoder_hidden_dim,
      "num_layers": 2,
      "bidirectional": true,
      "dropout": 0.3,
    },
    "relation_scorer": {
      "input_size": 2 * encoder_hidden_dim + ner_emb_dim,
      "hidden_size": 100,
      "label_namespace": "relation_labels",
      "negative_label": "0",
      "evaluated_labels": ['Direct-Defines', 'Indirect-Defines', 'AKA', 'Refers-To', 'Supplements'],
    },
    "evaluated_ner_labels": ['Term', 'Definition', 'Alias-Term', 'Referential-Definition', 'Referential-Term', 'Qualifier'],
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
    "cuda_device": std.parseInt(std.extVar("ALLENNLP_DEVICE")),
    "validation_metric": "-loss",
    "num_epochs": 100,
    "grad_clipping": 5.0,
  },
}
