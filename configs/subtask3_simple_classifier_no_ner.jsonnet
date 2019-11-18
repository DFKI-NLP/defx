local token_emb_dim = 50;
local encoder_hidden_dim = 50;

{
  "dataset_reader": {
    "type": "jsonl_reader",
    "subtasks": [3],
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true,
      },
    },
  },
  "train_data_path": "data/deft_split/jsonl/train.jsonl",
  "validation_data_path": "data/deft_split/jsonl/dev.jsonl",
  "model": {
    "type": "subtask3_classifier_with_gold_ner",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": token_emb_dim,
          "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.50d.txt.gz",
          "trainable": false,
        },
      },
    },
    "ner_tag_embedder": {
      "type": "embedding",
      "embedding_dim": 2,
      "trainable": true,
      "vocab_namespace": "tags",
    },
    "encoder": {
      "type": "lstm",
      "input_size": token_emb_dim,
      "hidden_size": encoder_hidden_dim,
      "num_layers": 1,
      "bidirectional": true,
    },
    "relation_scorer": {
      "input_size": 2 * encoder_hidden_dim,
      "hidden_size": 50,
      "label_namespace": "relation_labels",
      "negative_label": "0",
      "evaluated_labels": ['Direct-Defines', 'Indirect-Defines', 'Refers-To', 'AKA', 'Qualifies'],
    },
    "ignore_ner": true,
  },
  "iterator": {
    "type": "basic",
    "batch_size": 25,
  },
  "trainer": {
    "optimizer": {
      "type": "adam",
      "lr": 1e-3,
    },
    "num_serialized_models_to_keep": 1,
    "patience": 20,
    "cuda_device": std.parseInt(std.extVar("ALLENNLP_DEVICE")),
    "validation_metric": "-loss",
    "num_epochs": 100,
    "grad_clipping": 5.0,
  },
}
