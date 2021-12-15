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
"""SParC: Cross-Domain Semantic Parsing in Context"""

import json
from third_party.spider.preprocess.get_tables import dump_db_json_schema

import datasets


logger = datasets.logging.get_logger(__name__)

_CITATION = """\
@misc{yu2019sparc,
      title={SParC: Cross-Domain Semantic Parsing in Context}, 
      author={Tao Yu and Rui Zhang and Michihiro Yasunaga and Yi Chern Tan and Xi Victoria Lin and Suyi Li and Heyang Er and Irene Li and Bo Pang and Tao Chen and Emily Ji and Shreya Dixit and David Proctor and Sungrok Shim and Jonathan Kraft and Vincent Zhang and Caiming Xiong and Richard Socher and Dragomir Radev},
      year={2019},
      eprint={1906.02285},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""
_DESCRIPTION = """\
CoSQL is a large-scale dataset for cross-domain SemanticParsing in Context
"""
_HOMEPAGE = "https://github.com/taoyds/sparc"
_LICENSE = "CC BY-SA 4.0"

_URL = "https://drive.google.com/uc?export=download&id=13Abvu5SUMSP3SJM-ZIj66mOkeyAquR73"

class Sparc(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="sparc",
            version=VERSION,
            description="a large-scale dataset for cross-domain SemanticParsing in Context",
        ),
    ]

    def __init__(self, *args, writer_batch_size=None, **kwargs):
        super().__init__(*args, writer_batch_size=writer_batch_size, **kwargs)
        self.schema_cache = dict()

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
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        downloaded_filepath = dl_manager.download_and_extract(_URL)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_filepath": downloaded_filepath + "/sparc/train.json",
                    "db_path": downloaded_filepath + "/sparc/database",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "data_filepath": downloaded_filepath + "/sparc/dev.json",
                    "db_path": downloaded_filepath + "/sparc/database",
                },
            ),
        ]

    # tutorial: https://huggingface.co/docs/datasets/add_dataset.html
    # The input arguments of datasets.DatasetBuilder._generate_examples() are defined by t
    # he gen_kwargs dictionary returned by the datasets.DatasetBuilder._split_generator()
    def _generate_examples(self, data_filepath, db_path):
        logger.info("generating examples from = %s", data_filepath)
        idx = 0 # indexing each training instance
        with open(data_filepath, encoding="utf-8") as f:
            sparc = json.load(f)
            for sample in sparc:
                db_id = sample["database_id"]
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
                    "goal": sample["final"]["query"],
                    "utterances": [sample["final"]["utterance"]], # a list
                    "question": sample["final"]["utterance"], 
                    "query": sample["final"]["query"],
                    "turn_idx": -1,
                    **db_info,
                }
                idx += 1
                utterances = []
                for turn_idx, turn in enumerate(sample["interaction"]):
                    utterances.extend([turn["utterance"]]) # for each turn, its utterance is all the history utterances in that turn
                    yield idx, {
                        "goal": sample["final"]["query"],
                        "utterances": list(utterances),
                        "question": turn["utterance"], # question is the current utternace (no history)
                        "query": turn["query"],
                        "turn_idx": turn_idx,
                        **db_info,
                    }
                    idx += 1





