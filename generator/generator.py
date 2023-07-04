from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import tqdm
import torch
import os


from hyperparams import HyperParams


class Generator:
    def __init__(self):
        self.model = GPT2LMHeadModel.from_pretrained(HyperParams.model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(HyperParams.model_name)

    def train(self, train_data_loader):
        if not os.path.exists(HyperParams.output_path):
            os.mkdir(HyperParams.output_path)

        self.model = self.model.to(HyperParams.device)
        self.model.train()

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=HyperParams.learning_rate)

        tmp_items_tens = None
        for epoch in range(HyperParams.epochs):
            proc_seq_count = 0
            sum_loss = 0.0

            for _, item in tqdm(enumerate(train_data_loader), total=len(train_data_loader)):
                item_tens = item.to(HyperParams.device)

                if item_tens.size()[1] > HyperParams.max_seq_len:
                    continue

                if not torch.is_tensor(tmp_items_tens):
                    tmp_items_tens = item_tens
                    continue
                else:
                    if tmp_items_tens.size()[1] + item_tens.size()[1] > HyperParams.max_seq_len:
                        work_items_tens = tmp_items_tens
                        tmp_items_tens = item_tens
                    else:
                        tmp_items_tens = torch.cat([tmp_items_tens, item_tens[:, 1:]], dim=1)
                        continue

                outputs = self.model(work_items_tens, labels=work_items_tens)
                loss, logits = outputs[:2]
                loss.backward()
                sum_loss += loss.detach().data

                if proc_seq_count % HyperParams.batch_size == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    self.model.zero_grad()

                proc_seq_count += 1

            print(f"Epoch {epoch + 1} | Train loss: {sum_loss}")
            torch.save(self.model.state_dict(), f'{HyperParams.output_path}/{HyperParams.model_name}-{epoch + 1}.pt')

    def generate(self, prompt_text, n_seqs=1, min_length=16, max_length=32):
        encoded_prompt = self.tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")

        output_sequences = self.model.generate(
            input_ids=encoded_prompt,
            max_length=max_length,
            min_length=min_length,
            temperature=0.8,
            num_beams=None,
            top_k=0,
            top_p=0.8,
            repetition_penalty=1.2,
            do_sample=True,
            num_return_sequences=n_seqs,
            pad_token_id=self.tokenizer.eos_token_id
        )

        return [self.tokenizer.decode(sequence.tolist()) for sequence in output_sequences]

    def evaluate(self, revision):
        self.model.load_state_dict(torch.load(f'{HyperParams.output_path}/{HyperParams.model_name}-{revision}.pt',
                                              map_location=torch.device(HyperParams.device)))
        self.model.eval()
        self.model.to(HyperParams.device)
