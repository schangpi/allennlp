import sys
from IPython import embed
from typing import Dict, Any, Iterable
import argparse
import logging

from allennlp.common.util import prepare_environment
from allennlp.models.archival import load_archive
from allennlp.models.model import Model

# Disable some of the more verbose logging statements
logging.getLogger('allennlp.common.params').disabled = True
logging.getLogger('allennlp.nn.initializers').disabled = True
logging.getLogger('allennlp.modules.token_embedders.embedding').setLevel(logging.INFO)

archive_file = sys.argv[1]
output_name = sys.argv[2]
# Load from archive
archive = load_archive(archive_file)
config = archive.config
prepare_environment(config)
model = archive.model
# embed()

all_tasks = ['upos', 'xpos', 'chunk', 'ner', 'mwe', 'sem', 'semtr', 'supsense', 'com', 'frame', 'hyp']

with open(output_name + '_task_vectors.tsv', 'w') as f:
    for tsk in all_tasks:
        task_id = model.vocab.get_token_index(tsk)
        task_vec = model.task_field_embedder.token_embedder_tokens.weight[task_id].data.cpu().numpy().tolist()
        print('\t'.join([str(x) for x in task_vec]))
        f.write('\t'.join([str(x) for x in task_vec]))
        f.write('\n')

with open(output_name + '_task_metadata.tsv', 'w') as f:
    for tsk in all_tasks:
        print(tsk)
        f.write(tsk)
        f.write('\n')

with open(output_name + '_task_word_vectors.txt', 'w') as f:
    for tsk in all_tasks:
        task_id = model.vocab.get_token_index(tsk)
        task_vec = model.task_field_embedder.token_embedder_tokens.weight[task_id].data.cpu().numpy().tolist()
        print(tsk, ' '.join([str(x) for x in task_vec]))
        f.write(tsk + ' ' + ' '.join([str(x) for x in task_vec]))
        f.write('\n')