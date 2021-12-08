from transformers import AdamW, T5Tokenizer, T5ForConditionalGeneration
import torch
from torch.utils.data import Dataset
import json
from torch.utils.data import DataLoader
import sacrebleu
import os
from transformers.models.t5.configuration_t5 import T5Config, T5_PRETRAINED_CONFIG_ARCHIVE_MAP
import tqdm

model_name = 'mrm8488/t5-base-finetuned-wikiSQL-sql-to-en'

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
            context = ' | '.join([item['goal'], item['question'], item['query'], item['schema']])
            target = item['target']
            self.processed_data.append({'context': context.lower(),
                                   'target': target.lower()})

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        return self.processed_data[idx]

class Trainer:

    def __init__(self, args, use_self_play):
        self.args = args
        self.use_self_play = use_self_play
        spider_data = SQL2TextDataset(args['spider_file'])
        self.spider_loader = DataLoader(
            spider_data, batch_size=args['batch_size'], shuffle=True
        )
        if use_self_play:
            pretrain_data = SQL2TextDataset(args['pretrain_file'])
            self.pretrain_loader = DataLoader(
                pretrain_data, batch_size=args['batch_size'], shuffle=True
            )
        train_data = SQL2TextDataset(args['train_file'])
        val_data = SQL2TextDataset(args['val_file'])
        self.train_loader = DataLoader(
            train_data, batch_size=args['batch_size'], shuffle=True
        )
        self.val_loader = DataLoader(
            val_data, batch_size=args['eval_batch_size'], shuffle=False
        )
        t5_config = T5Config.from_pretrained(model_name, cache_dir='./pretrained_models', local_files_only=True)
        t5_config.dropout_rate = self.args['dropout_rate']
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, cache_dir='./pretrained_models', local_files_only=True)  # mrm8488/t5-base-finetuned-wikiSQL-sql-to-en
        self.tokenizer.add_tokens(['<', '<s>'])  # T5 vocab does not seem to contain "<"
        self.model = T5ForConditionalGeneration.from_pretrained(model_name, config=t5_config, cache_dir='./pretrained_models', local_files_only=True).cuda()
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model = torch.nn.DataParallel(self.model)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args['weight_decay'],
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = AdamW(
            optimizer_grouped_parameters, lr=args['lr']
        )


    def train(self):
        self.train_with_data(self.spider_loader, self.args['spider_epochs'], 'spider')
        if self.use_self_play:
            self.train_with_data(self.pretrain_loader, self.args['pretrain_epochs'], 'pretrain')
        self.train_with_data(self.train_loader, self.args['epochs'], 'train')

    def train_with_data(self, data_loader, epochs, phase):
        best_bleu = -1
        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0
            for step, batch in enumerate(tqdm.tqdm(data_loader)):
                encoding = self.tokenizer(batch['context'],
                                     padding='longest',
                                     max_length=self.args['max_source_length'],
                                     truncation=True,
                                     return_tensors="pt")
                input_ids, attention_mask = encoding.input_ids.cuda(), encoding.attention_mask.cuda()
                target_encoding = self.tokenizer(batch['target'],
                                            padding='longest',
                                            max_length=self.args['max_target_length'],
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
            # print("Phase {} Epoch {} Step {} training loss {}".format(phase, epoch, step, total_loss / step))
            print("Phase {} Epoch {}: BLEU score {}".format(phase, epoch, bleu))
            if bleu > best_bleu:
                print("Saving model")
                best_bleu = bleu
                self.save()

    def evaluate(self):
        tokenizer = T5Tokenizer.from_pretrained(model_name, cache_dir='./pretrained_models', local_files_only=True)
        tokenizer.add_tokens(['<', '<s>'])  # T5 vocab does not seem to contain "<"
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token  # to avoid an error
        self.model.eval()
        references = []
        sys = []
        for step, batch in enumerate(self.val_loader):
            inputs = tokenizer(batch['context'],
                               return_tensors="pt",
                               max_length=self.args['max_source_length'],
                               truncation=True,
                               padding=True)
            output_sequences = self.model.module.generate(
                input_ids=inputs['input_ids'].cuda(),
                attention_mask=inputs['attention_mask'].cuda(),
                num_beams=self.args['num_beams'],
                do_sample=self.args['do_sample'],
                length_penalty=self.args['length_penalty'],
                max_length=self.args['max_target_length']
            )
            output_sequences = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
            sys.extend(output_sequences)
            references.extend(batch['target'])
            print(output_sequences, batch['target'])
        bleu_score = sacrebleu.corpus_bleu(sys, [references])
        return bleu_score.score

    def save(self):
        if not os.path.exists(self.args['logdir']):
            os.makedirs(self.args['logdir'])
        torch.save(self.model.module, self.args['logdir'] + "/model_checkpoint")

def main(args, use_self_play):
    trainer = Trainer(args, use_self_play)
    trainer.train()