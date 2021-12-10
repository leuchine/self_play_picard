from transformers import AdamW, T5Tokenizer, T5ForConditionalGeneration
import torch
from torch.utils.data import Dataset
import json
from torch.utils.data import DataLoader
import sacrebleu
import os
from transformers.models.t5.configuration_t5 import T5Config, T5_PRETRAINED_CONFIG_ARCHIVE_MAP
import tqdm
from seq2seq.preprocess_sql2text import preprocess_dataset
from seq2seq.utils.args import parse_args


class SQL2TextDataset(Dataset):

    def __init__(self, path):
        if path.endswith(".json"):
            data = json.load(open(path))
        else:
            assert path.endswith(".jsonl")
            with open(path) as f:
                json_lines = f.readlines()
                data = [json.loads(line) for line in json_lines]
        self.processed_data = []
        for item in data:
            context = ' | '.join([item['goal'].strip(),
                                  item['question'].strip(),
                                  item['query'].strip(),
                                  item['schema'].strip()])
            target = item['target']
            self.processed_data.append({'context': context.lower(),
                                   'target': target.lower()})

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        return self.processed_data[idx]

class Trainer:

    def __init__(self, model_args, data_args, training_args, sql2text_args, use_self_play):
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.sql2text_args = sql2text_args
        self.use_self_play = use_self_play
        path = os.path.join(model_args.cache_dir, data_args.dataset + "_sql2text")

        # data
        if use_self_play:
            pretrain_data = SQL2TextDataset(os.path.join(path, sql2text_args.pretrain_file))
            self.pretrain_loader = DataLoader(
                pretrain_data,
                batch_size=training_args.per_device_train_batch_size * torch.cuda.device_count(),
                shuffle=True
            )
        train_data = SQL2TextDataset(os.path.join(path, 'train.json'))
        val_data = SQL2TextDataset(os.path.join(path, 'validation.json'))
        self.train_loader = DataLoader(
            train_data,
            batch_size=training_args.per_device_train_batch_size * torch.cuda.device_count(),
            shuffle=True
        )
        self.val_loader = DataLoader(
            val_data, batch_size=training_args.per_device_eval_batch_size, shuffle=False
        )

        # model
        t5_config = T5Config.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir, local_files_only=False)
        self.tokenizer = T5Tokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir, local_files_only=False)
        self.tokenizer.add_tokens(['<', '<s>'])  # T5 vocab does not seem to contain "<"
        self.model = T5ForConditionalGeneration.from_pretrained(model_args.model_name_or_path,
                                                                config=t5_config,
                                                                cache_dir=model_args.cache_dir,
                                                                local_files_only=False).cuda()
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model = torch.nn.DataParallel(self.model)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": sql2text_args.weight_decay_value,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = AdamW(
            optimizer_grouped_parameters, lr=training_args.learning_rate
        )


    def train(self):
        if self.use_self_play:
            self.train_with_data(self.pretrain_loader, self.sql2text_args.pretrain_epochs, 'pretrain')
        self.train_with_data(self.train_loader, self.training_args.num_train_epochs, 'train')

    def train_with_data(self, data_loader, epochs, phase):
        best_bleu = -1
        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0
            for step, batch in enumerate(tqdm.tqdm(data_loader)):
                encoding = self.tokenizer(batch['context'],
                                     padding='longest',
                                     max_length=self.sql2text_args.max_source_seq_length,
                                     truncation=True,
                                     return_tensors="pt")
                input_ids, attention_mask = encoding.input_ids.cuda(), encoding.attention_mask.cuda()
                target_encoding = self.tokenizer(batch['target'],
                                            padding='longest',
                                            max_length=self.sql2text_args.max_target_seq_length,
                                            truncation=True)
                labels = target_encoding.input_ids
                labels = [
                    [(label if label != self.tokenizer.pad_token_id else -100) for label in labels_example] for
                    labels_example in labels
                ]
                labels = torch.tensor(labels).cuda()
                loss = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
                loss = loss.mean()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.detach().item()

            bleu = self.evaluate()
            print("Phase {} Epoch {}: BLEU score {}".format(phase, epoch, bleu))
            if bleu > best_bleu:
                print("Saving model")
                best_bleu = bleu
                self.save()

    def evaluate(self):
        tokenizer = T5Tokenizer.from_pretrained(self.model_args.model_name_or_path,
                                                cache_dir=self.model_args.cache_dir,
                                                local_files_only=False)
        tokenizer.add_tokens(['<', '<s>'])  # T5 vocab does not seem to contain "<"
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token  # to avoid an error
        self.model.eval()
        references = []
        sys = []
        for step, batch in enumerate(self.val_loader):
            inputs = tokenizer(batch['context'],
                               return_tensors="pt",
                               max_length=self.sql2text_args.max_source_seq_length,
                               truncation=True,
                               padding=True)
            output_sequences = self.model.module.generate(
                input_ids=inputs['input_ids'].cuda(),
                attention_mask=inputs['attention_mask'].cuda(),
                num_beams=self.sql2text_args.beam_size,
                do_sample=self.sql2text_args.do_sample,
                length_penalty=self.sql2text_args.length_penalty,
                max_length=self.sql2text_args.max_target_seq_length
            )
            output_sequences = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
            sys.extend(output_sequences)
            references.extend(batch['target'])
            print(output_sequences, batch['target'])
        bleu_score = sacrebleu.corpus_bleu(sys, [references])
        return bleu_score.score

    def save(self):
        path = os.path.join(self.training_args.output_dir)
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.model.module, os.path.join(path, "model_checkpoint"))

def main(use_self_play):
    _, model_args, data_args, _, training_args, sql2text_args = parse_args()
    preprocess_dataset()
    trainer = Trainer(model_args, data_args, training_args, sql2text_args, use_self_play)
    trainer.train()

if __name__ == "__main__":
    main(False)