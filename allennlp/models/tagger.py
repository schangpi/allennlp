from typing import Dict, Optional

# from IPython import embed
import numpy
from overrides import overrides
import torch
from torch.nn.modules.linear import Linear
import torch.nn.functional as F

from allennlp.common import Params
from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder, ConditionalRandomField
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy, SpanBasedF1Measure


@Model.register("tagger")
class Tagger(Model):
    """
    This ``Tagger`` encodes a sequence of text with a stacked ``Seq2SeqEncoder``, then
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
    source_namespace : ``str``, optional (default=``tokens``)
        namespace for vocab of source sentences
    label_namespace : ``str``, optional (default=``labels``)
        namespace for vocab of tags
    is_crf : bool, optional (default=``False``)
        Use conditional random field loss instead of the standard softmax
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 stacked_encoder: Seq2SeqEncoder,
                 source_namespace: str = "tokens",
                 label_namespace: str = "labels",
                 is_crf: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(Tagger, self).__init__(vocab, regularizer)
        self.source_namespace = source_namespace
        self.label_namespace = label_namespace
        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size(label_namespace)
        self.stacked_encoder = stacked_encoder
        self.tag_projection_layer = TimeDistributed(Linear(self.stacked_encoder.get_output_dim(), self.num_classes))
        self.is_crf = is_crf
        if is_crf:
            self.crf = ConditionalRandomField(self.num_classes)
        check_dimensions_match(text_field_embedder.get_output_dim(), stacked_encoder.get_input_dim(),
                               "text field embedding dim", "encoder input dim")
        self.metrics = {
                "accuracy": CategoricalAccuracy(),
                "accuracy3": CategoricalAccuracy(top_k=3)
        }
        self.span_metric = SpanBasedF1Measure(vocab, tag_namespace=label_namespace)
        initializer(self)
        """
        # Initialize forget gate bias to 1 (applicable to LSTMCell only).
        encoder_parameters = self.stacked_encoder.state_dict()
        for pname in encoder_parameters:
            if 'bias_' in pname:
                print(pname)
                b = encoder_parameters[pname]
                l = len(b)
                b[l // 4:l // 2] = 1.0
        """

    def _examine_source_indices(self, preindices):
        if not isinstance(preindices, numpy.ndarray):
            preindices = preindices.data.cpu().numpy()
        all_predicted_tokens = []
        for indices in preindices:
            predicted_tokens = [self.vocab.get_token_from_index(
                x, namespace=self.source_namespace) for x in list(indices)]
            all_predicted_tokens.append(predicted_tokens)
        return all_predicted_tokens

    def _examine_target_indices(self, preindices):
        if not isinstance(preindices, numpy.ndarray):
            preindices = preindices.data.cpu().numpy()
        all_predicted_tokens = []
        for indices in preindices:
            indices = list(indices)
            # Collect indices till the first end_symbol
            # if self._end_index in indices:
            #     indices = indices[:indices.index(self._end_index)]
            predicted_tokens = [self.vocab.get_token_from_index(
                x, namespace=self.label_namespace) for x in indices]
            all_predicted_tokens.append(predicted_tokens)
        return all_predicted_tokens

    def _print_source_target_triplets(self, src, tgt, true_tgt):
        src = self._examine_source_indices(src)
        true_tgt = self._examine_target_indices(true_tgt)
        tgt = self._examine_target_indices(tgt)
        for i in [0, int(len(src)/2), -1]:
            print('Source:      ', ' '.join(src[i]))
            print('Target:      ', ' '.join(tgt[i]))
            print('True target: ', ' '.join(true_tgt[i]))
        print('')

    @overrides
    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                tags: torch.LongTensor = None,
                task_token: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
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
        tags : torch.LongTensor, optional (default = None)
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
        embedded_text_input = self.text_field_embedder(tokens)
        mask = get_text_field_mask(tokens)
        encoded_text = self.stacked_encoder(embedded_text_input, mask)

        logits = self.tag_projection_layer(encoded_text) # (batch_size, sequence_length, num_classes)
        if self.is_crf:
            predicted_tags = self.crf.viterbi_tags(logits, mask)
        else:
            reshaped_log_probs = logits.view(-1, self.num_classes)
            batch_size, sequence_length, _ = embedded_text_input.size()
            class_probabilities = F.softmax(reshaped_log_probs, dim=-1).view(
                [batch_size, sequence_length, self.num_classes])
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
                log_likelihood = self.crf(logits, tags, mask)
                loss = -log_likelihood
            else:
                loss = sequence_cross_entropy_with_logits(logits, tags, mask)
            for metric in self.metrics.values():
                metric(logits, tags, mask.float())
            output_dict["loss"] = loss

            # Represent viterbi tags as "class probabilities" that we can
            # feed into the `span_metric`
            class_probabilities = logits * 0.
            for i, instance_tags in enumerate(predicted_tags):
                for j, tag_id in enumerate(instance_tags):
                    class_probabilities[i, j, tag_id] = 1
            self.span_metric(class_probabilities, tags, mask)
            self._print_source_target_triplets(tokens['tokens'], numpy.array(predicted_tags), tags)
        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Converts the tag ids to the actual tags.
        ``output_dict["tags"]`` is a list of lists of tag_ids,
        so we use an ugly nested list comprehension.
        """
        output_dict["tags"] = [
            [self.vocab.get_token_from_index(tag, namespace=self.label_namespace)
             for tag in instance_tags]
            for instance_tags in output_dict["tags"]]
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        # accs = {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}
        metric_dict = self.span_metric.get_metric(reset=reset)
        f1 = {x: y for x, y in metric_dict.items() if "overall" in x}
        # return {**f1, **accs}
        return {**f1}

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'Tagger':
        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(vocab, embedder_params)
        stacked_encoder = Seq2SeqEncoder.from_params(params.pop("stacked_encoder"))
        source_namespace = params.pop("source_namespace", "tokens")
        label_namespace = params.pop("label_namespace", "labels")
        is_crf = params.pop("is_crf", False)
        initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))

        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   stacked_encoder=stacked_encoder,
                   source_namespace=source_namespace,
                   label_namespace=label_namespace,
                   is_crf=is_crf,
                   initializer=initializer,
                   regularizer=regularizer)