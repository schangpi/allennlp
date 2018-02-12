from typing import Dict, Optional

from IPython import embed
import numpy
from overrides import overrides

import torch
from torch.autograd import Variable
# from torch.nn.modules.rnn import GRUCell
from torch.nn.modules.rnn import LSTMCell
from torch.nn.modules.linear import Linear
import torch.nn.functional as F

from allennlp.common import Params
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.dataset_readers.seq2multiseq import START_SYMBOL, END_SYMBOL
from allennlp.modules import Attention, TextFieldEmbedder, Seq2SeqEncoder
from allennlp.modules.similarity_functions import SimilarityFunction
from allennlp.modules.token_embedders import Embedding
from allennlp.models.model import Model
from allennlp.models.encoder_decoders.simple_seq2seq import SimpleSeq2Seq
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits, weighted_sum
from allennlp.training.metrics import CategoricalAccuracy, SpanBasedF1Measure


@Model.register("simple_seq2multiseq")
class SimpleSeq2MultiSeq(Model):
    """
    This ``SimpleSeq2MultiSeq`` class is a :class:`Model` which takes a sequence, encodes it, and then
    uses the encoded representations to decode another sequence.  You can use this as the basis for
    a neural machine translation system, an abstractive summarization system, or any other common
    seq2seq problem.  The model here is simple, but should be a decent starting place for
    implementing recent models for these tasks.

    This ``SimpleSeq2MultiSeq`` model takes an encoder (:class:`Seq2SeqEncoder`) as an input, and
    implements the functionality of the decoder.  In this implementation, the decoder uses the
    encoder's outputs in two ways. The hidden state of the decoder is initialized with the output
    from the final time-step of the encoder, and when using attention, a weighted average of the
    outputs from the encoder is concatenated to the inputs of the decoder at every timestep.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        Vocabulary containing source and target vocabularies. They may be under the same namespace
        (``tokens``) or the target tokens can have a different namespace, in which case it needs to
        be specified as ``target_namespace``.
    source_embedder : ``TextFieldEmbedder``, required
        Embedder for source side sequences
    encoder : ``Seq2SeqEncoder``, required
        The encoder of the "encoder/decoder" model
    max_decoding_steps : int, required
        Length of decoded sequences
    target_namespace : str, optional (default = 'tokens')
        If the target side vocabulary is different from the source side's, you need to specify the
        target's namespace here. If not, we'll assume it is "tokens", which is also the default
        choice for the source side, and this might cause them to share vocabularies.
    target_embedding_dim : int, optional (default = source_embedding_dim)
        You can specify an embedding dimensionality for the target side. If not, we'll use the same
        value as the source embedder's.
    attention_function: ``SimilarityFunction``, optional (default = None)
        If you want to use attention to get a dynamic summary of the encoder outputs at each step
        of decoding, this is the function used to compute similarity between the decoder hidden
        state and encoder outputs.
    scheduled_sampling_ratio: float, optional (default = 0.0)
        At each timestep during training, we sample a random number between 0 and 1, and if it is
        not less than this value, we use the ground truth labels for the whole batch. Else, we use
        the predictions from the previous time step for the whole batch. If this value is 0.0
        (default), this corresponds to teacher forcing, and if it is 1.0, it corresponds to not
        using target side ground truth labels.  See the following paper for more information:
        Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks. Bengio et al.,
        2015.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 tasks: str,
                 domains: str,
                 source_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 max_decoding_steps: int,
                 upos_namespace: str = "upos_tokens",
                 ner_namespace: str = "ner_tokens",
                 chunk_namespace: str = "chunk_tokens",
                 target_embedding_dim: int = None,
                 attention_function: SimilarityFunction = None,
                 scheduled_sampling_ratio: float = 0.0,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(SimpleSeq2MultiSeq, self).__init__(vocab, regularizer)
        # print(len(tasks), len(domains))
        self._num_tasks = len(tasks)
        self._tasks = tasks
        self._domains = domains
        self._source_embedder = source_embedder
        self._encoder = encoder
        self._max_decoding_steps = max_decoding_steps
        self._upos_namespace = upos_namespace
        self._ner_namespace = ner_namespace
        self._chunk_namespace = chunk_namespace
        self._attention_function = attention_function
        self._scheduled_sampling_ratio = scheduled_sampling_ratio
        self._upos_seq2seq = SimpleSeq2Seq(vocab=vocab, source_embedder=source_embedder, encoder=encoder,
                                           max_decoding_steps=max_decoding_steps, target_namespace=upos_namespace,
                                           target_embedding_dim=target_embedding_dim,
                                           attention_function=attention_function,
                                           scheduled_sampling_ratio=scheduled_sampling_ratio,
                                           initializer=initializer, regularizer=regularizer)
        self._ner_seq2seq = SimpleSeq2Seq(vocab=vocab, source_embedder=source_embedder, encoder=encoder,
                                          max_decoding_steps=max_decoding_steps, target_namespace=ner_namespace,
                                          target_embedding_dim=target_embedding_dim,
                                          attention_function=attention_function,
                                          scheduled_sampling_ratio=scheduled_sampling_ratio,
                                          initializer=initializer, regularizer=regularizer)
        self._chunk_seq2seq = SimpleSeq2Seq(vocab=vocab, source_embedder=source_embedder, encoder=encoder,
                                            max_decoding_steps=max_decoding_steps, target_namespace=chunk_namespace,
                                            target_embedding_dim=target_embedding_dim,
                                            attention_function=attention_function,
                                            scheduled_sampling_ratio=scheduled_sampling_ratio,
                                            initializer=initializer, regularizer=regularizer)
        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                task_token: torch.LongTensor,
                domain_token: torch.LongTensor,
                source_tokens: Dict[str, torch.LongTensor],
                upos_tokens: Dict[str, torch.LongTensor] = None,
                ner_tokens: Dict[str, torch.LongTensor] = None,
                chunk_tokens: Dict[str, torch.LongTensor] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Decoder logic for producing the entire target sequence.

        Parameters
        ----------
        source_tokens : Dict[str, torch.LongTensor]
           The output of ``TextField.as_array()`` applied on the source ``TextField``. This will be
           passed through a ``TextFieldEmbedder`` and then through an encoder.
        target_tokens : Dict[str, torch.LongTensor], optional (default = None)
           Output of ``Textfield.as_array()`` applied on target ``TextField``. We assume that the
           target tokens are also represented as a ``TextField``.
        """
        # (batch_size, input_sequence_length, encoder_output_dim)
        batch_size = len(source_tokens)
        upos_output_dict = self._upos_seq2seq.forward(source_tokens, upos_tokens)
        ner_output_dict = self._ner_seq2seq.forward(source_tokens, ner_tokens)
        chunk_output_dict = self._chunk_seq2seq.forward(source_tokens, chunk_tokens)
        loss = 0.0
        predictions = []
        label_namespaces = []
        task_token_ids = task_token.data.cpu().numpy()
        for b in range(batch_size):
            task = self.vocab.get_token_from_index(task_token_ids[b][0], namespace="task_labels")
            if task == 'upos':
                loss += upos_output_dict['loss']
                predictions.append(upos_output_dict['predictions'])
            elif task == 'ner':
                loss += ner_output_dict['loss']
                predictions.append(ner_output_dict['predictions'])
            elif task == 'chunk':
                loss += chunk_output_dict['loss']
                predictions.append(chunk_output_dict['predictions'])
            label_namespaces.append(task)
        output_dict = {'loss': loss,
                       'predictions': predictions,
                       'label_namespaces': label_namespaces}
        # embed()
        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        This method overrides ``Model.decode``, which gets called after ``Model.forward``, at test
        time, to finalize predictions. The logic for the decoder part of the encoder-decoder lives
        within the ``forward`` method.

        This method trims the output predictions to the first end symbol, replaces indices with
        corresponding tokens, and adds a field called ``predicted_tokens`` to the ``output_dict``.
        """
        upos_output_dict = self._upos_seq2seq.decode(output_dict)
        ner_output_dict = self._ner_seq2seq.forward(output_dict)
        chunk_output_dict = self._chunk_seq2seq.forward(output_dict)
        all_predicted_tokens = []
        for b, task in enumerate(output_dict['label_namespaces']):
            if task == 'upos':
                all_predicted_tokens.append(upos_output_dict['predictions'][b])
            elif task == 'ner':
                all_predicted_tokens.append(ner_output_dict['predictions'][b])
            elif task == 'chunk':
                all_predicted_tokens.append(chunk_output_dict['predictions'][b])
        output_dict["predicted_tokens"] = all_predicted_tokens
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        upos_accs = {'upos-' + metric_name: metric.get_metric(reset)
                    for metric_name, metric in self._upos_seq2seq.metrics.items()}
        ner_accs = {'ner-' + metric_name: metric.get_metric(reset)
                    for metric_name, metric in self._ner_seq2seq.metrics.items()}
        chunk_accs = {'chunk-' + metric_name: metric.get_metric(reset)
                      for metric_name, metric in self._chunk_seq2seq.metrics.items()}
        upos_metric_dict = self._upos_seq2seq.span_metric.get_metric(reset=reset)
        upos_f1 = {'upos-' + x: y for x, y in upos_metric_dict.items() if "overall" in x}
        ner_metric_dict = self._ner_seq2seq.span_metric.get_metric(reset=reset)
        ner_f1 = {'ner-' + x: y for x, y in ner_metric_dict.items() if "overall" in x}
        chunk_metric_dict = self._chunk_seq2seq.span_metric.get_metric(reset=reset)
        chunk_f1 = {'chunk-' + x: y for x, y in chunk_metric_dict.items() if "overall" in x}
        accs = {metric_name:
                  (upos_accs['upos-' + metric_name] + ner_accs['ner-' + metric_name] +
                   chunk_accs['chunk-' + metric_name])/3 for metric_name, _ in self._upos_seq2seq.metrics.items()}
        f1 = {x: (upos_f1['upos-' + x] + ner_f1['ner-' + x] + chunk_f1['chunk-' + x])/3
              for x, _ in upos_metric_dict.items() if "overall" in x}
        return {**f1, **accs, **upos_f1, **ner_f1, **chunk_f1, **upos_accs, **ner_accs, **chunk_accs}

    @classmethod
    def from_params(cls, vocab, params: Params) -> 'SimpleSeq2MultiSeq':
        tasks = params.pop("tasks")
        domains = params.pop("domains")
        source_embedder_params = params.pop("source_embedder")
        source_embedder = TextFieldEmbedder.from_params(vocab, source_embedder_params)
        encoder = Seq2SeqEncoder.from_params(params.pop("encoder"))
        max_decoding_steps = params.pop("max_decoding_steps")
        upos_namespace = params.pop("upos_namespace", "upos_tokens")
        ner_namespace = params.pop("ner_namespace", "ner_tokens")
        chunk_namespace = params.pop("chunk_namespace", "chunk_tokens")
        # upos_namespace = params.pop("upos_namespace")
        # ner_namespace = params.pop("ner_namespace")
        # chunk_namespace = params.pop("chunk_namespace")
        # If no attention function is specified, we should not use attention, not attention with
        # default similarity function.
        attention_function_type = params.pop("attention_function", None)
        if attention_function_type is not None:
            attention_function = SimilarityFunction.from_params(attention_function_type)
        else:
            attention_function = None
        scheduled_sampling_ratio = params.pop_float("scheduled_sampling_ratio", 0.0)
        initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))
        return cls(vocab,
                   tasks=tasks,
                   domains=domains,
                   source_embedder=source_embedder,
                   encoder=encoder,
                   max_decoding_steps=max_decoding_steps,
                   upos_namespace=upos_namespace,
                   ner_namespace=ner_namespace,
                   chunk_namespace=chunk_namespace,
                   attention_function=attention_function,
                   scheduled_sampling_ratio=scheduled_sampling_ratio,
                   initializer=initializer,
                   regularizer=regularizer
                   )
