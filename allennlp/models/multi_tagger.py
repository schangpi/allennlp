from typing import Dict, Optional

from IPython import embed
import numpy
from overrides import overrides
import torch
from torch.nn.modules.linear import Linear
import torch.nn.functional as F
from collections import OrderedDict

from allennlp.common import Params
from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder, ConditionalRandomField
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy, SpanBasedF1Measure


@Model.register("multi_tagger")
class MultiTagger(Model):
    """
    This ``MultiTagger`` encodes a sequence of text with a stacked ``Seq2SeqEncoder``, then
    predicts multiple tags for each token in the sequence.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    tasks : ``str``, required
        A list of task names
    domains : ``str``, required
        A list of domain names
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    stacked_encoder : ``Seq2SeqEncoder``
        The encoder (with its own internal stacking) that we will use in between embedding tokens
        and predicting output tags.
    source_namespace : ``str``, optional (default=``tokens``)
        namespace for vocab of source sentences
    label_suffix_namespace : ``str``, optional (default=``labels``)
        task_name + '_' + label_suffix_namespace is the namespace for vocab of tags for each task
    is_crf : bool, optional (default=``False``)
        Use conditional random field loss instead of the standard softmax
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
        # Create task-to-ID and domain-to-ID mappings
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
        self.label_namespaces = OrderedDict()
        self.tag_projection_layer = OrderedDict()
        self.num_classes = OrderedDict()
        self.is_crf = is_crf
        self.crf = OrderedDict()
        self.metrics = OrderedDict()
        self.span_metric = OrderedDict()
        for tsk in self.tasks:
            task_label_namespace = tsk + '_' + label_suffix_namespace
            self.label_namespaces[tsk] = task_label_namespace
            self.num_classes[tsk] = self.vocab.get_vocab_size(task_label_namespace)
            self.tag_projection_layer[tsk] = TimeDistributed(Linear(self.stacked_encoder.get_output_dim(),
                                                                    self.num_classes[tsk]))
            if is_crf:
                self.crf[tsk] = ConditionalRandomField(self.num_classes[tsk])
            self.metrics[tsk] = {
                "accuracy": CategoricalAccuracy(),
                "accuracy3": CategoricalAccuracy(top_k=3)
            }
            self.span_metric[tsk] = SpanBasedF1Measure(vocab, tag_namespace=task_label_namespace)
        self.tag_projection_layer = torch.nn.Sequential(self.tag_projection_layer)
        if is_crf:
            self.crf = torch.nn.Sequential(self.crf)
        check_dimensions_match(text_field_embedder.get_output_dim(), stacked_encoder.get_input_dim(),
                               "text field embedding dim", "encoder input dim")
        initializer(self)

    def _examine_source_indices(self, preindices):
        if not isinstance(preindices, numpy.ndarray):
            preindices = preindices.data.cpu().numpy()
        all_predicted_tokens = []
        for indices in preindices:
            predicted_tokens = [self.vocab.get_token_from_index(
                x, namespace=self.source_namespace) for x in list(indices)]
            all_predicted_tokens.append(predicted_tokens)
        return all_predicted_tokens

    def _examine_target_indices(self, tsk, preindices):
        if not isinstance(preindices, numpy.ndarray):
            preindices = preindices.data.cpu().numpy()
        all_predicted_tokens = []
        for indices in preindices:
            indices = list(indices)
            # Collect indices till the first end_symbol
            # if self._end_index in indices:
            #     indices = indices[:indices.index(self._end_index)]
            predicted_tokens = [self.vocab.get_token_from_index(
                x, namespace=self.label_namespaces[tsk]) for x in indices]
            all_predicted_tokens.append(predicted_tokens)
        return all_predicted_tokens

    def _print_source_target_triplets(self, tsk, src, tgt, true_tgt):
        src = self._examine_source_indices(src)
        true_tgt = self._examine_target_indices(tsk, true_tgt)
        tgt = self._examine_target_indices(tsk, tgt)
        for i in [0, int(len(src)/2), -1]:
            print('Source:      ', ' '.join(src[i]))
            print('Target:      ', ' '.join(tgt[i]))
            print('True target: ', ' '.join(true_tgt[i]))
        print('')

    def task_forward(self,
                     tsk_id,
                     tsk,
                     tokens: Dict[str, torch.LongTensor],
                     tags: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        embedded_text_input = self.text_field_embedder(tokens)
        mask = get_text_field_mask(tokens)
        encoded_text = self.stacked_encoder(embedded_text_input, mask)
        logits = self.tag_projection_layer[tsk_id].forward(encoded_text) # (batch_size, sequence_length, num_classes)
        if self.is_crf:
            predicted_tags = self.crf[tsk_id].viterbi_tags(logits, mask)
        else:
            reshaped_log_probs = logits.view(-1, self.num_classes[tsk])
            batch_size, sequence_length, _ = embedded_text_input.size()
            class_probabilities = F.softmax(reshaped_log_probs, dim=-1).view(
                [batch_size, sequence_length, self.num_classes[tsk]])
            all_predictions = class_probabilities.cpu().data.numpy()
            if all_predictions.ndim == 3:
                predictions_list = [all_predictions[i] for i in range(all_predictions.shape[0])]
            else:
                predictions_list = [all_predictions]
            predicted_tags = []
            for predictions in predictions_list:
                argmax_indices = numpy.argmax(predictions, axis=-1)
                predicted_tags.append(argmax_indices)
        output_dict = {"logits": logits,
                       "tags": predicted_tags,
                       "mask": mask}
        if tags is not None:
            loss = 0.0
            if self.is_crf:
                log_likelihood = self.crf[tsk_id](logits, tags, mask)
                loss = -log_likelihood
            else:
                loss = sequence_cross_entropy_with_logits(logits, tags, mask)
            for metric in self.metrics[tsk].values():
                metric(logits, tags, mask.float())
            output_dict["loss"] = loss

            # Represent viterbi tags as "class probabilities" that we can
            # feed into the `span_metric`
            class_probabilities = logits * 0.
            for i, instance_tags in enumerate(predicted_tags):
                for j, tag_id in enumerate(instance_tags):
                    class_probabilities[i, j, tag_id] = 1
            self.span_metric[tsk](class_probabilities, tags, mask)
            self._print_source_target_triplets(tsk, tokens['tokens'], numpy.array(predicted_tags), tags)
        return output_dict

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
        batch_size = task_token.shape[0]
        predictions = [[]] * batch_size
        label_namespaces = [''] * batch_size
        loss = 0.0
        task_ids = []
        for tt in task_token.squeeze().data.cpu().numpy():
            # map task token to strings
            # map strings to ind
            task_ids.append(self.task_to_id[self.vocab.get_token_from_index(tt, 'task_labels')])
        task_ids = numpy.array(task_ids)
        # batch_one = False

        for i, tsk in enumerate(self.tasks):
            # For each element in tokens (tokens, token_characters, etc.),
            # select only the relevant data in the minibatch
            task_indices = (task_ids == i).nonzero()[0]
            print(i, task_indices)
            if task_indices.shape[0] < 1:
                continue
            # elif task_indices.shape[0] == 1:
            #     task_indices = numpy.concatenate((task_indices, task_indices), axis=0)
            task_tokens = {}
            for k in tokens:
                if task_indices.shape[0] > 1:
                    task_tokens[k] = tokens[k][task_indices, :].squeeze()
                else:
                    task_tokens[k] = tokens[k][task_indices, :]
            task_all_tags = all_tags[:, i, :].squeeze(dim=1)
            if task_indices.shape[0] > 1:
                task_all_tags = task_all_tags[task_indices, :].squeeze().contiguous()
            else:
                task_all_tags = task_all_tags[task_indices, :].contiguous()
                # embed()
            task_output_dict = self.task_forward(i, tsk, task_tokens, task_all_tags)
            task_tags = task_output_dict['tags']
            for j, ti in enumerate(task_indices):
                predictions[ti] = task_tags[j]
                label_namespaces[ti] = tsk
            loss += task_output_dict['loss']
            # if task_indices.shape[0] == 1:
            #     batch_one = True
        # if batch_one:
        #     embed()
        output_dict = {'loss': loss,
                       'tags': predictions,
                       'label_namespaces': label_namespaces}
        return output_dict

    def task_decode(self, tsk, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        output_dict["tags"] = [
            [self.vocab.get_token_from_index(tag, namespace=self.label_namespaces[tsk])
             for tag in instance_tags]
            for instance_tags in output_dict["tags"]]
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
            tmp_output_dict = output_dict.copy()
            task_output_dicts[tsk + '_labels'] = self.task_decode(tsk, tmp_output_dict)
        all_predicted_tokens = []
        for b, task_namespace in enumerate(output_dict['label_namespaces']):
            all_predicted_tokens.append(task_output_dicts[task_namespace]['tags'][b])
        output_dict['tags'] = all_predicted_tokens
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        # task_accs = {}
        task_f1s = {}
        for tsk in self.tasks:
            # for metric_name, metric in self.metrics[tsk].items():
            #     task_accs[tsk + '-' + metric_name] = metric.get_metric(reset)
            metric_dict = self.span_metric[tsk].get_metric(reset=reset)
            for x, y in metric_dict.items():
                # if "overall" in x:
                if "f1-measure-overall" in x:
                    task_f1s[tsk + '-' + x] = y
        # avg_accs = {}
        avg_f1s = {}
        # for metric_name, _ in self.metrics[self.tasks[0]].items():
            # total_acc = []
            # for tsk in self.tasks:
            #     total_acc.append(task_accs[tsk + '-' + metric_name])
            # avg_accs[metric_name] = sum(total_acc) / len(total_acc)
        for x, _ in self.span_metric[self.tasks[0]].get_metric(reset=reset).items():
            # if "overall" in x:
            if "f1-measure-overall" in x:
                total_f1 = []
                for tsk in self.tasks:
                    total_f1.append(task_f1s[tsk + '-' + x])
                avg_f1s[x] = sum(total_f1) / len(total_f1)
        # return {**avg_f1s, ** avg_accs, **task_f1s, **task_accs, }
        return {**avg_f1s, **task_f1s}

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
