from generator.generator import Generator
from generator.dataset import TextUtils


if __name__ == '__main__':
    model = Generator()
    model.evaluate(1)

    output = model.generate('nameless', 1, 64, 128)
    print(TextUtils.process_output(output[0], 64))
