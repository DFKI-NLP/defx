{
  local token_emb_dim = 2,
  local ner_emb_dim = 4,
  local encoder_hidden_dim = 3,

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
  "train_data_path": "tests/fixtures/jsonl_format_samples.jsonl",
  "validation_data_path": "tests/fixtures/jsonl_format_samples.jsonl",
  "model": {
    "type": "subtask3_classifier_with_gold_ner",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": token_emb_dim,
          "trainable": false,
        },
      },
    },
    "ner_tag_embedder": {
      "type": "embedding",
      "embedding_dim": ner_emb_dim,
      "trainable": true
    },
    "encoder": {
      "type": "lstm",
      "input_size": token_emb_dim,
      "hidden_size": encoder_hidden_dim,
      "num_layers": 1,
      "bidirectional": true,
    },
    "relation_scorer": {
      "input_size": 2 * encoder_hidden_dim + ner_emb_dim,
      "hidden_size": 5,
      "label_namespace": "relation_labels",
      "negative_label": "0",
      "evaluated_labels": ['Direct-Defines', 'Indirect-Defines', 'Refers-To', 'AKA', 'Qualifies'],
    }
  },
  "iterator": {
    "type": "basic",
    "batch_size": 2,
  },
  "trainer": {
    "optimizer": {
      "type": "adam",
      "lr": 1e-3,
    },
    "patience": 1,
    "cuda_device": -1,
    "validation_metric": "-loss",
    "num_epochs": 1,
    "grad_clipping": 5.0,
  },
}
