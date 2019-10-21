{
  local token_emb_dim = 300,

  "dataset_reader": {
    "type": "subtask1_reader",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true,
      },
    },
  },
  "train_data_path": "data/subtask1/split/train.tsv",
  "validation_data_path": "data/subtask1/split/dev.tsv",
  "model": {
    "type": "subtask1_classifier_wrapper",
    "model": {
      "type": "basic_classifier",
      "text_field_embedder": {
        "token_embedders": {
          "tokens": {
            "type": "embedding",
            "embedding_dim": token_emb_dim,
            "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.840B.300d.txt.gz",
            "trainable": false,
          },
        },
      },
      "seq2vec_encoder": {
        "type": "cnn",
        "embedding_dim": token_emb_dim,
        "num_filters": 500,
        "ngram_filter_sizes": [2, 3, 4],
      },
      "dropout": 0.5
    },
  },
  "iterator": {
    "type": "basic",
    "batch_size": 32,
  },
  "trainer": {
    "optimizer": {
      "type": "adam",
      "lr": 1e-3,
    },
    "patience": 10,
    "validation_metric": "+f1-measure",
    "num_epochs": 100,
    "num_serialized_models_to_keep": 1,
    "grad_clipping": 5.0,
    "cuda_device": -1,
    "learning_rate_scheduler": {
      "type": "reduce_on_plateau",
      "mode": "min",
      "factor": 0.5,
      "patience": 3,
      "verbose": false,
    },
  },
}