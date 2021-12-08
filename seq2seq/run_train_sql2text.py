from seq2seq.utils.args import parse_args
import datasets
from seq2seq.utils.cosql import cosql_add_serialized_schema

def main():
    picard_args, model_args, data_args, data_training_args, training_args = parse_args()
    cosql_dataset_dict = datasets.load.load_dataset(
        path=data_args.dataset_paths["cosql"], cache_dir=model_args.cache_dir
    )
    _cosql_add_serialized_schema = lambda ex: cosql_add_serialized_schema(
        ex=ex,
        data_training_args=data_training_args,
    )
    dataset = cosql_dataset_dict['train'].map(
        _cosql_add_serialized_schema,
        batched=False,
        num_proc=data_training_args.preprocessing_num_workers,
        load_from_cache_file=not data_training_args.overwrite_cache,
    )
    print(dataset[0])

if __name__ == "__main__":
    main()