This is the official implementation of the following paper:

Qi Liu, Zihuiwen Ye, Tao Yu, Phil Blunsom and Linfeng Song. [Augmenting Multi-Turn Text-to-SQL Datasets with Self-Play]


## About Self-Play for Text-to-SQL
The task of context-dependent text-to-SQL aims to convert multi-turn user utterances to formal SQL queries. This is a challenging task due to both the scarcity of training data from which to learn complex contextual dependencies and to generalize to unseen databases. In this paper we explore augmenting the training datasets using self-play, which leverages contextual information to synthesize new interactions to adapt the model to new databases. We first design a SQL-to-text model conditioned on a sampled goal query, which represents a user’s intent, that then converses with a text-to-SQL semantic parser to generate new interactions. We then filter the synthesized interactions and retrain the models with the augmented data. We find that self-play improves the accuracy of a strong baseline on SParC and CoSQL, two widely used cross-domain text-to-SQL datasets. Our analysis shows that self-play simulates various conversational thematic relations,  enhances cross-domain generalization and improves beam-search.


### Training

The scripts for training text-to-SQL models on the datasets CoSQL and SParC are `launch_cosql.sh`, `launch_sparc.sh`, respectively. We take `launch_sparc.sh` as an example. 
You can run it with:
```
$ bash launch_sparc.sh
```

<br /> In `launch_sparc.sh`, there are four commands. 
```
nohup make train_sparc
```
This trains a text-to-SQL model on the specified dataset (SParC), and saves the checkpoints in the output directory `train_sparc` under the `seq2seq` directory.  <br /><br />  


```
nohup make train_sql2text_sparc 
```
This trains a SQL-to-text model on the specified dataset (SParC), saves the checkpoints in `train_sql2text_sparc` under the `seq2seq` directory.  <br /><br />  


```
nohup  make self_play_sparc
```
This generates synthetic self-play examples by first sampling goal guery templates proposed in [GAZP](https://github.com/vzhong/gazp), then using the trained text-to-SQL and SQL-to-text models to converse with each other.  <br /><br />      

```
nohup  make train_sparc_self_play
```
This retrains the text-to-SQL and SQL-to-text models with the generated self-play interactions, in addition to the training data (SParC). It is saved in `train_sparc_self_play` under the `seq2seq` directory.  <br /><br />  


### Evaluation

To evaluate the models trained on SParC, please run:
```
$ make eval_sparc
```

To evaluate the models trained on CoSQL, please run:
```
$ make eval_cosql
```
