"""
The various :class:`~allennlp.data.iterators.data_iterator.DataIterator` subclasses
can be used to iterate over datasets with different batching and padding schemes.
"""

from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.data.iterators.basic_iterator import BasicIterator
from allennlp.data.iterators.multi_iterator import MultiIterator
from allennlp.data.iterators.bucket_iterator import BucketIterator
from allennlp.data.iterators.adaptive_iterator import AdaptiveIterator
