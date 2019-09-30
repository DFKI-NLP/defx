from sklearn.model_selection import train_test_split


print('Splitting subtask 1 train set into train and dev set...')

examples = []
labels = []
with open('data/subtask1/split/train_dev.tsv') as train_dev_file:
    for line in train_dev_file.readlines():
        text, label = line.split('\t')
        examples.append(line)
        labels.append(label)

train, dev = train_test_split(examples, test_size=.1, stratify=labels)

with open('data/subtask1/split/train.tsv', 'w') as train_file:
    for example in train:
        train_file.write(example)

with open('data/subtask1/split/dev.tsv', 'w') as dev_file:
    for example in dev:
        dev_file.write(example)
