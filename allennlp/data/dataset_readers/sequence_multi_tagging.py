from typing import Dict, List
import logging
import os

from overrides import overrides

from allennlp.common import Params
from allennlp.common.file_utils import cached_path
from allennlp.common.tqdm import Tqdm
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField, LabelField, ListField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

DEFAULT_WORD_TAG_DELIMITER = "###"
# ALL_TASKS = ['upos', 'xpos', 'chunk', 'ner', 'mwe', 'sem', 'semtr', 'supsense', 'ccg', 'com']

@DatasetReader.register("sequence_multi_tagging")
class SequenceMultiTaggingDatasetReader(DatasetReader):
    """
    Reads instances from a pretokenised file where each line is in the following format:

    WORD###TAG [TAB] WORD###TAG [TAB] ..... \n

    and converts it into a ``Dataset`` suitable for sequence tagging. You can also specify
    alternative delimiters in the constructor.

    Parameters
    ----------
    word_tag_delimiter: ``str``, optional (default=``"###"``)
        The text that separates each WORD from its TAG.
    token_delimiter: ``str``, optional (default=``None``)
        The text that separates each WORD-TAG pair from the next pair. If ``None``
        then the line will just be split on whitespace.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
        Note that the `output` tags will always correspond to single token IDs based on how they
        are pre-tokenised in the data file.
    """
    def __init__(self,
                 word_tag_delimiter: str = DEFAULT_WORD_TAG_DELIMITER,
                 token_delimiter: str = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 tasks: str = None,
                 domains: str = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._word_tag_delimiter = word_tag_delimiter
        self._token_delimiter = token_delimiter
        self._tasks = tasks
        self._domains = domains

    @overrides
    def _read(self, file_path):
        # if `file_path` is a URL, redirect to the cache
        # file_path = cached_path(file_path)
        for filename in os.listdir(file_path):
            filename_splitted = filename.split('_')
            task_name = filename_splitted[-3]
            domain_name = filename_splitted[-2]
            if task_name not in self._tasks or domain_name not in self._domains:
                continue
            with open(os.path.join(file_path, filename), "r") as data_file:
                logger.info("Reading instances from lines in file at: %s", filename)
                for line in Tqdm.tqdm(data_file):
                    line = line.strip("\n")
                    # skip blank lines
                    if not line:
                        continue
                    tokens_and_tags = [pair.rsplit(self._word_tag_delimiter, 1)
                                       for pair in line.split(self._token_delimiter)]
                    tokens = [Token(token) for token, tag in tokens_and_tags]
                    tags = [tag for token, tag in tokens_and_tags]
                    sequence = TextField(tokens, self._token_indexers)
                    sequence_tags = SequenceLabelField(tags, sequence, label_namespace=task_name + '_labels')
                    task_field = LabelField(task_name, label_namespace="task_labels")
                    domain_field = LabelField(domain_name, label_namespace="domain_labels")
                    input_dict = {'task_token': task_field,
                                  'domain_token': domain_field,
                                  'tokens': sequence}
                    all_tags = []
                    empty_tags = ['O'] * len(tags)
                    for tsk in self._tasks:
                        if tsk != task_name:
                            empty_sequence_tags = SequenceLabelField(empty_tags, sequence,
                                                                     label_namespace=tsk + '_labels')
                            all_tags.append(empty_sequence_tags)
                        else:
                            all_tags.append(sequence_tags)
                    input_dict['all_tags'] = ListField(all_tags)
                    yield Instance(input_dict)

    def text_to_instance(self, tokens: List[Token]) -> Instance:  # type: ignore
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.
        """
        # pylint: disable=arguments-differ
        return Instance({'tokens': TextField(tokens, token_indexers=self._token_indexers)})

    @classmethod
    def from_params(cls, params: Params) -> 'SequenceMultiTaggingDatasetReader':
        token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))
        word_tag_delimiter = params.pop("word_tag_delimiter", DEFAULT_WORD_TAG_DELIMITER)
        token_delimiter = params.pop("token_delimiter", None)
        tasks = params.pop('tasks', None)
        domains = params.pop('domains', None)
        lazy = params.pop('lazy', False)
        params.assert_empty(cls.__name__)
        return SequenceMultiTaggingDatasetReader(token_indexers=token_indexers,
                                                 word_tag_delimiter=word_tag_delimiter,
                                                 token_delimiter=token_delimiter,
                                                 tasks=tasks,
                                                 domains=domains,
                                                 lazy=lazy)