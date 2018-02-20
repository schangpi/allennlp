from typing import Dict, Optional

from IPython import embed
import numpy
from overrides import overrides
import torch

from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder, ConditionalRandomField
from allennlp.models.model import Model
from allennlp.models.tagger import Tagger
from allennlp.nn import InitializerApplicator, RegularizerApplicator
# from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
# from allennlp.training.metrics import CategoricalAccuracy, SpanBasedF1Measure


@Model.register("multi_tagger")
class MultiTagger(Model):
    """
    This ``MultiTagger`` simply encodes a sequence of text with a stacked ``Seq2SeqEncoder``, then
    predicts a tag for each token in the sequence.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    stacked_encoder : ``Seq2SeqEncoder``
        The encoder (with its own internal stacking) that we will use in between embedding tokens
        and predicting output tags.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(self, vocab: Vocabulary,
                 tasks: str,
                 domains: str,
                 text_field_embedder: TextFieldEmbedder,
                 stacked_encoder: Seq2SeqEncoder,
                 source_namespace: str = "tokens",
                 label_suffix_namespace: str = "labels",
                 is_crf: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(MultiTagger, self).__init__(vocab, regularizer)
        self.tasks = tasks
        self.domains = domains
        self.task_to_id = {}
        for i, tsk in enumerate(tasks):
            self.task_to_id[tsk] = i
        self.domain_to_id = {}
        for i, dmn in enumerate(domains):
            self.domain_to_id[dmn] = i
        self.source_namespace = source_namespace
        self.label_suffix_namespace = label_suffix_namespace
        self.text_field_embedder = text_field_embedder
        self.stacked_encoder = stacked_encoder
        initializer(self)

        self.taggers = {}
        for tsk in self.tasks:
            print(tsk, torch.cuda.current_device())
            self.taggers[tsk] = Tagger(vocab=vocab,
                                       text_field_embedder=text_field_embedder,
                                       stacked_encoder=stacked_encoder,
                                       source_namespace=source_namespace,
                                       label_namespace=tsk + '_' + label_suffix_namespace,
                                       is_crf=is_crf,
                                       initializer=initializer,
                                       regularizer=regularizer)
        # embed()
        # Check that parameters are the same for
        # each self.taggers[tsk].state_dict()['stacked_encoder._module.weight_ih_l0']
        # And different for
        # self.taggers[tsk].state_dict()['tag_projection_layer._module.weight']

    @overrides
    def forward(self,  # type: ignore
                task_token: torch.LongTensor,
                domain_token: torch.LongTensor,
                tokens: Dict[str, torch.LongTensor],
                all_tags: torch.LongTensor) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        tokens : Dict[str, torch.LongTensor], required
            The output of ``TextField.as_array()``, which should typically be passed directly to a
            ``TextFieldEmbedder``. This output is a dictionary mapping keys to ``TokenIndexer``
            tensors.  At its most basic, using a ``SingleIdTokenIndexer`` this is: ``{"tokens":
            Tensor(batch_size, num_tokens)}``. This dictionary will have the same keys as were used
            for the ``TokenIndexers`` when you created the ``TextField`` representing your
            sequence.  The dictionary is designed to be passed directly to a ``TextFieldEmbedder``,
            which knows how to combine different word representations into a single vector per
            token in your input.
        *_tags : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer gold class labels of shape
            ``(batch_size, num_tokens)``.

        Returns
        -------
        An output dictionary consisting of:
        logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
            unnormalised log probabilities of the tag classes.
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
            a distribution of the tag classes per word.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.

        """
        # num_tasks = all_tags.size(1)
        # map task token to strings
        # map strings to ind
        batch_size = task_token.shape[0]
        predictions = [[]] * batch_size
        label_namespaces = [''] * batch_size
        loss = 0.0
        task_ids = []
        for tt in task_token.squeeze().data.cpu().numpy():
            task_ids.append(self.task_to_id[self.vocab.get_token_from_index(tt, 'task_labels')])
        task_ids = numpy.array(task_ids)
        for i, tsk in enumerate(self.tasks):
            # For each element in tokens (tokens, token_characters, etc.),
            # select only the relevant data in the minibatch
            task_indices = (task_ids == i).nonzero()[0]
            print(i, task_indices)
            if task_indices.shape[0] < 1:
                continue
            task_tokens = {}
            for k in tokens:
                if task_indices.shape[0] > 1:
                    task_tokens[k] = tokens[k][task_indices, :].squeeze()
                else:
                    task_tokens[k] = tokens[k][task_indices, :]
            task_all_tags = all_tags[:, i, :].squeeze()
            # embed()
            task_output_dict = self.taggers[tsk].forward(task_tokens, task_all_tags[task_indices, :].contiguous())
            task_tags = task_output_dict['tags']
            for j, ti in enumerate(task_indices):
                predictions[ti] = task_tags[j]
                label_namespaces[ti] = tsk
            loss += task_output_dict['loss']
        output_dict = {'loss': loss,
                       'tags': predictions,
                       'label_namespaces': label_namespaces}
        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Converts the tag ids to the actual tags.
        ``output_dict["tags"]`` is a list of lists of tag_ids,
        so we use an ugly nested list comprehension.
        """
        task_output_dicts = {}
        for tsk in self.tasks:
            task_output_dicts[tsk + '_labels'] = self.taggers[tsk].decode(output_dict)
        all_predicted_tokens = []
        for b, task_namespace in enumerate(output_dict['label_namespaces']):
            all_predicted_tokens.append(task_output_dicts[task_namespace]['tags'][b])
        output_dict['tags'] = all_predicted_tokens
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        task_accs = {}
        task_f1s = {}
        for tsk in self.tasks:
            # for metric_name, metric in self.taggers[tsk].metrics.items():
            #     task_accs[tsk + '-' + metric_name] = metric.get_metric(reset)
            metric_dict = self.taggers[tsk].span_metric.get_metric(reset=reset)
            for x, y in metric_dict.items():
                # if "overall" in x:
                if "f1-measure-overall" in x:
                    task_f1s[tsk + '-' + x] = y
        # avg_accs = {}
        avg_f1s = {}
        # for metric_name, _ in self.taggers[self.tasks[0]].metrics.items():
            # total_acc = []
            # for tsk in self.tasks:
            #     total_acc.append(task_accs[tsk + '-' + metric_name])
            # avg_accs[metric_name] = sum(total_acc) / len(total_acc)
        for x, _ in self.taggers[self.tasks[0]].span_metric.get_metric(reset=reset).items():
            # if "overall" in x:
            if "f1-measure-overall" in x:
                total_f1 = []
                for tsk in self.tasks:
                    total_f1.append(task_f1s[tsk + '-' + x])
                avg_f1s[x] = sum(total_f1) / len(total_f1)
        # return {**avg_f1s, ** avg_accs, **task_f1s, **task_accs, }
        return {**avg_f1s, **task_f1s}

    @overrides
    def set_device(self, device):
        for tsk in self.tasks:
            self.taggers[tsk] = self.taggers[tsk].cuda(device)

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'MultiTagger':
        tasks = params.pop("tasks")
        domains = params.pop("domains")
        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(vocab, embedder_params)
        stacked_encoder = Seq2SeqEncoder.from_params(params.pop("stacked_encoder"))
        source_namespace = params.pop("source_namespace", "tokens")
        label_suffix_namespace = params.pop("label_suffix_namespace", "labels")
        is_crf = params.pop("is_crf", False)
        # device = params.pop("device", -1)
        initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))
        return cls(vocab=vocab,
                   tasks=tasks,
                   domains=domains,
                   text_field_embedder=text_field_embedder,
                   stacked_encoder=stacked_encoder,
                   source_namespace=source_namespace,
                   label_suffix_namespace=label_suffix_namespace,
                   is_crf=is_crf,
                   initializer=initializer,
                   regularizer=regularizer)
