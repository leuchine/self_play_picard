# coding=utf-8
# Copyright 2021 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Loading the self-play data for CoSQL"""


import json
from third_party.spider.preprocess.get_tables import dump_db_json_schema

import datasets
import glob

logger = datasets.logging.get_logger(__name__)


_DESCRIPTION = """\
Loading the self-play data for training.
"""


class SelfPlayData(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="self_play_data",
            version=VERSION,
            description="Self-play data for training",
        ),
    ]

    def __init__(self, *args, writer_batch_size=None, **kwargs):
        dataset_in_use = kwargs['dataset_in_use']
        self.save_self_play_path = kwargs['save_self_play_path']
        del kwargs['dataset_in_use']
        del kwargs['save_self_play_path']
        super().__init__(*args, writer_batch_size=writer_batch_size, **kwargs)
        self.schema_cache = dict()
        if 'cosql' in dataset_in_use:
            self.dataset = 'cosql_dataset'
        elif 'sparc' in dataset_in_use:
            self.dataset = 'sparc_dataset'
        else:
            raise Exception("Unknown dataset")


    def _info(self):
        features = datasets.Features(
            {
                "goal": datasets.Value("string"),
                "query": datasets.Value("string"),
                "utterances": datasets.features.Sequence(datasets.Value("string")),
                "turn_idx": datasets.Value("int32"),
                "question": datasets.Value("string"),
                "db_id": datasets.Value("string"),
                "db_path": datasets.Value("string"),
                "db_table_names": datasets.features.Sequence(datasets.Value("string")),
                "db_column_names": datasets.features.Sequence(
                    {
                        "table_id": datasets.Value("int32"),
                        "column_name": datasets.Value("string"),
                    }
                ),
                "db_column_types": datasets.features.Sequence(datasets.Value("string")),
                "db_primary_keys": datasets.features.Sequence({"column_id": datasets.Value("int32")}),
                "db_foreign_keys": datasets.features.Sequence(
                    {
                        "column_id": datasets.Value("int32"),
                        "other_column_id": datasets.Value("int32"),
                    }
                ),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=None,
            license=None,
            citation=None,
        )

    def _split_generators(self, dl_manager):
        if self.dataset == 'cosql_dataset':
            _URL = "https://drive.google.com/uc?export=download&id=14x6lsWqlu6gR-aYxa6cemslDN3qT3zxP"
        elif self.dataset == 'sparc_dataset':
            _URL = "https://drive.google.com/uc?export=download&id=13Abvu5SUMSP3SJM-ZIj66mOkeyAquR73"

        downloaded_filepath = dl_manager.download_and_extract(_URL)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_filepath_repxh": self.save_self_play_path + "/self_play_*.jsonl",
                    "db_path": downloaded_filepath + "/{}/database".format(self.dataset),
                },
            ),
        ]

    def _generate_examples(self, data_filepath_repx, db_path):
        """This function returns the examples in the raw (text) form."""
        idx = 0  # indexing each training instance
        for data_filepath in glob.glob(data_filepath_repx):
            logger.info("generating examples from = %s", data_filepath)
            with open(data_filepath) as f:
                for line in f:
                    turn_data = json.loads(line)
                    db_id = turn_data["db_id"]
                    if db_id not in self.schema_cache:
                        self.schema_cache[db_id] = dump_db_json_schema(
                            db_path + "/" + db_id + "/" + db_id + ".sqlite", db_id
                        )
                    schema = self.schema_cache[db_id]

                    db_info = {
                        "db_id": db_id,
                        "db_path": db_path,
                        "db_table_names": schema["table_names_original"],
                        "db_column_names": [
                            {"table_id": table_id, "column_name": column_name}
                            for table_id, column_name in schema["column_names_original"]
                        ],
                        "db_column_types": schema["column_types"],
                        "db_primary_keys": [{"column_id": column_id} for column_id in schema["primary_keys"]],
                        "db_foreign_keys": [
                            {"column_id": column_id, "other_column_id": other_column_id}
                            for column_id, other_column_id in schema["foreign_keys"]
                        ],
                    }

                    yield idx, {
                        "goal": turn_data["goal"],
                        "utterances": turn_data["utterances"],
                        "question": turn_data["question"],
                        "query": turn_data["query"],
                        "turn_idx": turn_data["turn_idx"],
                        **db_info,
                    }
                    idx += 1
