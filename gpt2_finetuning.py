from generator.dataset import LovecraftDataset
from generator.generator import Generator


if __name__ == '__main__':
    model = Generator()
    train_data_loader, _ = LovecraftDataset.get(model.tokenizer)
    model.train(train_data_loader)
