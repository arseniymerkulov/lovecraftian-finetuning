class HyperParams:
    device = 'cpu'
    model_name = 'gpt2-medium'

    dataset_path = 'data'
    output_path = 'trained_models'

    dot_stopwords = {'Dr', 'Ms', 'Mr', 'Mrs', 'Prof', 'Inc', 'Fr', 'St'}

    batch_size = 16
    epochs = 8
    learning_rate = 3e-4
    warmup_steps = 100
    max_seq_len = 400
