from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch
import nltk
import random
import glob
import os


from hyperparams import HyperParams


class TextUtils:
    @staticmethod
    def preprocess_text(tokenizer, text):
        text = text.replace('\n', ' ')

        for stop_word in HyperParams.dot_stopwords:
            text = text.replace(stop_word + '. ', stop_word + '._')

        tokenized = tokenizer.tokenize(text)
        random.shuffle(tokenized)

        sentences = []
        for sentence in tokenized[:10]:
            for stop_word in HyperParams.dot_stopwords:
                sentence = sentence.replace(stop_word + '._', stop_word + '. ')

            sentences.append(sentence)

        return sentences

    @staticmethod
    def preprocess_title(title):
        return os.path.basename(title).replace('_', ' ').replace('.txt', '')

    @staticmethod
    def load_data(tokenizer, path):
        title = TextUtils.preprocess_title(path)

        with open(path, 'r', encoding='utf8') as file:
            text = file.read()
            sentences = TextUtils.preprocess_text(tokenizer, text)
            labels = [title for sentence in sentences]

            return labels, sentences

    @staticmethod
    def process_output(output, string_length=64):
        counter = 0
        processed = ''

        for i in range(len(output)):
            counter += 1

            if output[i] == ' ' and counter > string_length:
                processed = f'{processed}\n'
                counter = 0
            else:
                processed = f'{processed}{output[i]}'

        return processed


class LovecraftDataset(Dataset):
    def __init__(self, x, y, tokenizer, max_length=1024):
        super().__init__()
        self.sentences = [torch.tensor(tokenizer.encode(f'{y[i]}\n{x[i][:max_length]}')) for i in range(len(y))]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item):
        return self.sentences[item]

    @staticmethod
    def get(tokenizer):
        nltk.download('punkt')
        sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

        books = glob.glob(f'{HyperParams.dataset_path}/*.txt')
        y = []
        x = []

        for book in books:
            labels, sentences = TextUtils.load_data(sentence_tokenizer, book)

            y += labels
            x += sentences

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        train_dataset = LovecraftDataset(x_train, y_train, tokenizer)
        train_data_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

        test_dataset = LovecraftDataset(x_test, y_test, tokenizer)

        return train_data_loader, test_dataset
