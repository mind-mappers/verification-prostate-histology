from tensorflow.keras.layers import Dropout
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras import constraints
from tensorflow.keras import activations
from tensorflow.keras import backend
import tensorflow as tf
import numpy as np
import string
import math
import re

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.framework import ops

_CHR_IDX = string.ascii_lowercase


def gelu(features, approximate=False, name=None):
    with ops.name_scope(name, "Gelu", [features]):
        features = ops.convert_to_tensor(features, name="features")
        if approximate:
            coeff = math_ops.cast(0.044715, features.dtype)
            return 0.5 * features * (
                1.0 + math_ops.tanh(0.7978845608028654 *
                                    (features + coeff * math_ops.pow(features, 3))))
        else:
            return 0.5 * features * (1.0 + math_ops.erf(
                features / math_ops.cast(1.4142135623730951, features.dtype)))

class Softmax(tf.keras.layers.Layer):
    def __init__(self, axis=-1, **kwargs):
        super(Softmax, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis

    def call(self, inputs, mask=None):
        if mask is not None:
            # Since mask is 1.0 for positions we want to keep and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -1e.9 for masked positions.
            large_neg = tf.float16.min if inputs.dtype == tf.float16 else -1e9
            adder = (1.0 - tf.cast(mask, inputs.dtype)) * (large_neg)

            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            inputs += adder
        if isinstance(self.axis, (tuple, list)):
            if len(self.axis) > 1:
                return tf.exp(inputs - tf.reduce_logsumexp(
                    inputs, axis=self.axis, keepdims=True))
            else:
                return backend.softmax(inputs, axis=self.axis[0])
        return backend.softmax(inputs, axis=self.axis)

    def get_config(self):
        config = {'axis': self.axis}
        base_config = super(Softmax, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

def _analyze_split_string(split_string,
                          bias_axes,
                          input_shape,
                          output_shape,
                          left_elided=False):
    """Analyze an pre-split einsum string to find the weight shape."""
    input_spec = split_string.group(1)
    weight_spec = split_string.group(2)
    output_spec = split_string.group(3)
    elided = len(input_shape) - len(input_spec)

    if isinstance(output_shape, int):
        output_shape = [output_shape]
    else:
        output_shape = list(output_shape)

    output_shape.insert(0, input_shape[0])

    if elided > 0 and left_elided:
        for i in range(1, elided):
            # We already inserted the 0th input dimension at dim 0, so we need to
            # start at location 1 here.
            output_shape.insert(1, input_shape[i])
    elif elided > 0 and not left_elided:
        for i in range(len(input_shape) - elided, len(input_shape)):
            output_shape.append(input_shape[i])

    if left_elided:
        # If we have beginning dimensions elided, we need to use negative indexing
        # to determine where in the input dimension our values are.
        input_dim_map = {
            dim: (i + elided) - len(input_shape) for i, dim in enumerate(input_spec)
        }
        # Because we've constructed the full output shape already, we don't need
        # to do negative indexing.
        output_dim_map = {dim: (i + elided)
                          for i, dim in enumerate(output_spec)}
    else:
        input_dim_map = {dim: i for i, dim in enumerate(input_spec)}
        output_dim_map = {dim: i for i, dim in enumerate(output_spec)}

    for i, dim in enumerate(input_spec):
        input_shape_at_dim = input_shape[i]
        if dim in output_dim_map:
            output_shape_at_dim = output_shape[output_dim_map[dim]]
            if (output_shape_at_dim is not None and
                    output_shape_at_dim != input_shape_at_dim):
                raise ValueError(
                    "Input shape and output shape do not match at shared "
                    f"dimension '{dim}'. Input shape is {input_shape_at_dim}, "
                    "and output shape "
                    f"is {output_shape[output_dim_map[dim]]}.")

    for dim in output_spec:
        if dim not in input_spec and dim not in weight_spec:
            raise ValueError(
                f"Dimension '{dim}' was specified in the output '{output_spec}' but "
                f"has no corresponding dim in the input spec '{input_spec}' or "
                f"weight spec '{output_spec}'")

    weight_shape = []
    for dim in weight_spec:
        if dim in input_dim_map:
            weight_shape.append(input_shape[input_dim_map[dim]])
        elif dim in output_dim_map:
            weight_shape.append(output_shape[output_dim_map[dim]])
        else:
            raise ValueError(
                f"Weight dimension '{dim}' did not have a match in either "
                f"the input spec '{input_spec}' or the output spec '{output_spec}'. "
                "For this layer, the weight must be fully specified.")

    if bias_axes is not None:
        num_left_elided = elided if left_elided else 0
        idx_map = {
            char: output_shape[i + num_left_elided]
            for i, char in enumerate(output_spec)
        }

        for char in bias_axes:
            if char not in output_spec:
                raise ValueError(
                    f"Bias dimension '{char}' was requested, but is not part "
                    f"of the output spec '{output_spec}'")

        first_bias_location = min([output_spec.find(char)
                                  for char in bias_axes])
        bias_output_spec = output_spec[first_bias_location:]

        bias_shape = [
            idx_map[char] if char in bias_axes else 1 for char in bias_output_spec
        ]

        if not left_elided:
            for _ in range(elided):
                bias_shape.append(1)
    else:
        bias_shape = None

    return weight_shape, bias_shape, output_shape

def _get_output_shape(output_rank, known_last_dims):
  return [None] * (output_rank - len(known_last_dims)) + list(known_last_dims)

def _analyze_einsum_string(equation, bias_axes, input_shape, output_shape):
  """Analyzes an einsum string to determine the required weight shape."""

  dot_replaced_string = re.sub(r"\.\.\.", "0", equation)

  # This is the case where no ellipses are present in the string.
  split_string = re.match("([a-zA-Z]+),([a-zA-Z]+)->([a-zA-Z]+)",
                          dot_replaced_string)
  if split_string:
    return _analyze_split_string(split_string, bias_axes, input_shape,
                                 output_shape)

  # This is the case where ellipses are present on the left.
  split_string = re.match("0([a-zA-Z]+),([a-zA-Z]+)->0([a-zA-Z]+)",
                          dot_replaced_string)
  if split_string:
    return _analyze_split_string(
        split_string, bias_axes, input_shape, output_shape, left_elided=True)

  # This is the case where ellipses are present on the right.
  split_string = re.match("([a-zA-Z]{2,})0,([a-zA-Z]+)->([a-zA-Z]+)0",
                          dot_replaced_string)
  if split_string:
    return _analyze_split_string(split_string, bias_axes, input_shape,
                                 output_shape)

  raise ValueError(
      f"Invalid einsum equation '{equation}'. Equations must be in the form "
      "[X],[Y]->[Z], ...[X],[Y]->...[Z], or [X]...,[Y]->[Z]....")



def _build_proj_equation(free_dims, bound_dims, output_dims):
    """Builds an einsum equation for projections inside multi-head attention."""
    input_str = ""
    kernel_str = ""
    output_str = ""
    bias_axes = ""
    letter_offset = 0
    for i in range(free_dims):
        char = _CHR_IDX[i + letter_offset]
        input_str += char
        output_str += char

    letter_offset += free_dims
    for i in range(bound_dims):
        char = _CHR_IDX[i + letter_offset]
        input_str += char
        kernel_str += char

    letter_offset += bound_dims
    for i in range(output_dims):
        char = _CHR_IDX[i + letter_offset]
        kernel_str += char
        output_str += char
        bias_axes += char
    equation = "%s,%s->%s" % (input_str, kernel_str, output_str)

    return equation, bias_axes, len(output_str)

def _build_attention_equation(rank, attn_axes):
    """Builds einsum equations for the attention computation.
    Query, key, value inputs after projection are expected to have the shape as:
    `(bs, <non-attention dims>, <attention dims>, num_heads, channels)`.
    `bs` and `<non-attention dims>` are treated as `<batch dims>`.
    The attention operations can be generalized:
    (1) Query-key dot product:
    `(<batch dims>, <query attention dims>, num_heads, channels), (<batch dims>,
    <key attention dims>, num_heads, channels) -> (<batch dims>,
    num_heads, <query attention dims>, <key attention dims>)`
    (2) Combination:
    `(<batch dims>, num_heads, <query attention dims>, <key attention dims>),
    (<batch dims>, <value attention dims>, num_heads, channels) -> (<batch dims>,
    <query attention dims>, num_heads, channels)`
    Args:
      rank: Rank of query, key, value tensors.
      attn_axes: List/tuple of axes, `[-1, rank)`,
        that attention will be applied to.
    Returns:
      Einsum equations.
    """
    target_notation = _CHR_IDX[:rank]
    # `batch_dims` includes the head dim.
    batch_dims = tuple(np.delete(range(rank), attn_axes + (rank - 1,)))
    letter_offset = rank
    source_notation = ""
    for i in range(rank):
        if i in batch_dims or i == rank - 1:
            source_notation += target_notation[i]
        else:
            source_notation += _CHR_IDX[letter_offset]
            letter_offset += 1

    product_notation = "".join([target_notation[i] for i in batch_dims] +
                               [target_notation[i] for i in attn_axes] +
                               [source_notation[i] for i in attn_axes])
    dot_product_equation = "%s,%s->%s" % (source_notation, target_notation,
                                          product_notation)
    attn_scores_rank = len(product_notation)
    combine_equation = "%s,%s->%s" % (product_notation, source_notation,
                                      target_notation)
    return dot_product_equation, combine_equation, attn_scores_rank



class EinsumDense(tf.keras.layers.Layer):
  def __init__(self,
               equation,
               output_shape,
               activation=None,
               bias_axes=None,
               kernel_initializer="glorot_uniform",
               bias_initializer="zeros",
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    super(EinsumDense, self).__init__(**kwargs)
    self.equation = equation
    if isinstance(output_shape, int):
      self.partial_output_shape = [output_shape]
    else:
      self.partial_output_shape = list(output_shape)
    self.bias_axes = bias_axes
    self.activation = activations.get(activation)
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.kernel_constraint = constraints.get(kernel_constraint)
    self.bias_constraint = constraints.get(bias_constraint)

  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    shape_data = _analyze_einsum_string(self.equation,
                                        self.bias_axes,
                                        input_shape,
                                        self.partial_output_shape)
    kernel_shape, bias_shape, self.full_output_shape = shape_data
    self.kernel = self.add_weight(
        "kernel",
        shape=kernel_shape,
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        dtype=self.dtype,
        trainable=True)

    if bias_shape is not None:
      self.bias = self.add_weight(
          "bias",
          shape=bias_shape,
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          dtype=self.dtype,
          trainable=True)
    else:
      self.bias = None
    super(EinsumDense, self).build(input_shape)

  def compute_output_shape(self, _):
    return tf.TensorShape(self.full_output_shape)

  def get_config(self):
    config = {
        "output_shape": self.partial_output_shape,
        "equation": self.equation,
        "activation": activations.serialize(self.activation),
        "bias_axes": self.bias_axes,
        "kernel_initializer": initializers.serialize(self.kernel_initializer),
        "bias_initializer": initializers.serialize(self.bias_initializer),
        "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
        "bias_regularizer": regularizers.serialize(self.bias_regularizer),
        "activity_regularizer":
            regularizers.serialize(self.activity_regularizer),
        "kernel_constraint": constraints.serialize(self.kernel_constraint),
        "bias_constraint": constraints.serialize(self.bias_constraint),
    }
    base_config = super(EinsumDense, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self, inputs):
    ret = tf.einsum(self.equation, inputs, self.kernel)
    if self.bias is not None:
      ret += self.bias
    if self.activation is not None:
      ret = self.activation(ret)
    return ret


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self,
                 num_heads,
                 key_dim,
                 value_dim=None,
                 dropout=0.0,
                 use_bias=True,
                 output_shape=None,
                 attention_axes=None,
                 kernel_initializer="glorot_uniform",
                 bias_initializer="zeros",
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self._num_heads = num_heads
        self._key_dim = key_dim
        self._value_dim = value_dim if value_dim else key_dim
        self._dropout = dropout
        self._use_bias = use_bias
        self._output_shape = output_shape
        self._kernel_initializer = initializers.get(kernel_initializer)
        self._bias_initializer = initializers.get(bias_initializer)
        self._kernel_regularizer = regularizers.get(kernel_regularizer)
        self._bias_regularizer = regularizers.get(bias_regularizer)
        self._activity_regularizer = regularizers.get(activity_regularizer)
        self._kernel_constraint = constraints.get(kernel_constraint)
        self._bias_constraint = constraints.get(bias_constraint)
        if attention_axes is not None and not isinstance(attention_axes,
                                                         collections.abc.Sized):
            self._attention_axes = (attention_axes,)
        else:
            self._attention_axes = attention_axes
        self._built_from_signature = False
        self._query_shape, self._key_shape, self._value_shape = None, None, None

        def get_config(self):
            config = {
                "num_heads": self._num_heads,
                "key_dim": self._key_dim,
                "value_dim": self._value_dim,
                "dropout": self._dropout,
                "use_bias": self._use_bias,
                "output_shape": self._output_shape,
                "attention_axes": self._attention_axes,
                "kernel_initializer":
                    initializers.serialize(self._kernel_initializer),
                "bias_initializer":
                    initializers.serialize(self._bias_initializer),
                "kernel_regularizer":
                    regularizers.serialize(self._kernel_regularizer),
                "bias_regularizer":
                    regularizers.serialize(self._bias_regularizer),
                "activity_regularizer":
                    regularizers.serialize(self._activity_regularizer),
                "kernel_constraint":
                    constraints.serialize(self._kernel_constraint),
                "bias_constraint":
                    constraints.serialize(self._bias_constraint),
                "query_shape": self._query_shape,
                "key_shape": self._key_shape,
                "value_shape": self._value_shape,
            }
            base_config = super(MultiHeadAttention, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))

        @classmethod
        def from_config(cls, config):
            # If the layer has a different build() function from the Keras default,
            # we need to trigger the customized build to create weights.
            query_shape = config.pop("query_shape")
            key_shape = config.pop("key_shape")
            value_shape = config.pop("value_shape")
            layer = cls(**config)
            if None in [query_shape, key_shape, value_shape]:
                logging.warning(
                    "One of dimensions of the input shape is missing. It should have been"
                    " memorized when the layer was serialized. "
                    "%s is created without weights.",
                    str(cls))
            else:
                layer._build_from_signature(
                    query_shape, value_shape, key_shape)  # pylint: disable=protected-access
            return layer

    def _build_from_signature(self, query, value, key=None):
        """Builds layers and variables.
        Once the method is called, self._built_from_signature will be set to True.
        Args:
          query: Query tensor or TensorShape.
          value: Value tensor or TensorShape.
          key: Key tensor or TensorShape.
        """
        self._built_from_signature = True
        if hasattr(query, "shape"):
            self._query_shape = tf.TensorShape(query.shape)
        else:
            self._query_shape = tf.TensorShape(query)
        if hasattr(value, "shape"):
            self._value_shape = tf.TensorShape(value.shape)
        else:
            self._value_shape = tf.TensorShape(value)
        if key is None:
            self._key_shape = self._value_shape
        elif hasattr(key, "shape"):
            self._key_shape = tf.TensorShape(key.shape)
        else:
            self._key_shape = tf.TensorShape(key)

        common_kwargs = dict(
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            activity_regularizer=self._activity_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint)
        # Any setup work performed only once should happen in an `init_scope`
        # to avoid creating symbolic Tensors that will later pollute any eager
        # operations.
        with tf.init_scope():
            free_dims = self._query_shape.rank - 1
            einsum_equation, bias_axes, output_rank = _build_proj_equation(
                free_dims, bound_dims=1, output_dims=2)
            self._query_dense = EinsumDense(
                einsum_equation,
                output_shape=_get_output_shape(output_rank - 1,
                                               [self._num_heads, self._key_dim]),
                bias_axes=bias_axes if self._use_bias else None,
                name="query",
                **common_kwargs)
            einsum_equation, bias_axes, output_rank = _build_proj_equation(
                self._key_shape.rank - 1, bound_dims=1, output_dims=2)
            self._key_dense = EinsumDense(
                einsum_equation,
                output_shape=_get_output_shape(output_rank - 1,
                                               [self._num_heads, self._key_dim]),
                bias_axes=bias_axes if self._use_bias else None,
                name="key",
                **common_kwargs)
            einsum_equation, bias_axes, output_rank = _build_proj_equation(
                self._value_shape.rank - 1, bound_dims=1, output_dims=2)
            self._value_dense = EinsumDense(
                einsum_equation,
                output_shape=_get_output_shape(output_rank - 1,
                                               [self._num_heads, self._value_dim]),
                bias_axes=bias_axes if self._use_bias else None,
                name="value",
                **common_kwargs)

            # Builds the attention computations for multi-head dot product attention.
            # These computations could be wrapped into the keras attention layer once
            # it support mult-head einsum computations.
            self._build_attention(output_rank)
            self._output_dense = self._make_output_dense(
                free_dims, common_kwargs, "attention_output")

    def _make_output_dense(self, free_dims, common_kwargs, name=None):
        """Builds the output projection matrix.
        Args:
          free_dims: Number of free dimensions for einsum equation building.
          common_kwargs: Common keyword arguments for einsum layer.
          name: Name for the projection layer.
        Returns:
          Projection layer.
        """
        if self._output_shape:
            if not isinstance(self._output_shape, collections.abc.Sized):
                output_shape = [self._output_shape]
            else:
                output_shape = self._output_shape
        else:
            output_shape = [self._query_shape[-1]]
        einsum_equation, bias_axes, output_rank = _build_proj_equation(
            free_dims, bound_dims=2, output_dims=len(output_shape))
        return EinsumDense(
            einsum_equation,
            output_shape=_get_output_shape(output_rank - 1, output_shape),
            bias_axes=bias_axes if self._use_bias else None,
            name=name,
            **common_kwargs)

    def _build_attention(self, rank):
        """Builds multi-head dot-product attention computations.
        This function builds attributes necessary for `_compute_attention` to
        costomize attention computation to replace the default dot-product
        attention.
        Args:
          rank: the rank of query, key, value tensors.
        """
        if self._attention_axes is None:
            self._attention_axes = tuple(range(1, rank - 2))
        else:
            self._attention_axes = tuple(self._attention_axes)
        self._dot_product_equation, self._combine_equation, attn_scores_rank = (
            _build_attention_equation(rank, attn_axes=self._attention_axes))
        norm_axes = tuple(
            range(attn_scores_rank - len(self._attention_axes), attn_scores_rank))
        self._softmax = Softmax(axis=norm_axes)
        self._dropout_layer = Dropout(rate=self._dropout)

    def _masked_softmax(self, attention_scores, attention_mask=None):
        # Normalize the attention scores to probabilities.
        # `attention_scores` = [B, N, T, S]
        if attention_mask is not None:
            # The expand dim happens starting from the `num_heads` dimension,
            # (<batch_dims>, num_heads, <query_attention_dims, key_attention_dims>)
            mask_expansion_axis = -len(self._attention_axes) * 2 - 1
            for _ in range(len(attention_scores.shape) - len(attention_mask.shape)):
                attention_mask = tf.expand_dims(
                    attention_mask, axis=mask_expansion_axis)
        return self._softmax(attention_scores, attention_mask)

    def _compute_attention(self,
                           query,
                           key,
                           value,
                           attention_mask=None,
                           training=None):
        """Applies Dot-product attention with query, key, value tensors.
        This function defines the computation inside `call` with projected
        multi-head Q, K, V inputs. Users can override this function for customized
        attention implementation.
        Args:
          query: Projected query `Tensor` of shape `(B, T, N, key_dim)`.
          key: Projected key `Tensor` of shape `(B, T, N, key_dim)`.
          value: Projected value `Tensor` of shape `(B, T, N, value_dim)`.
          attention_mask: a boolean mask of shape `(B, T, S)`, that prevents
            attention to certain positions.
          training: Python boolean indicating whether the layer should behave in
            training mode (adding dropout) or in inference mode (doing nothing).
        Returns:
          attention_output: Multi-headed outputs of attention computation.
          attention_scores: Multi-headed attention weights.
        """
        # Note: Applying scalar multiply at the smaller end of einsum improves
        # XLA performance, but may introduce slight numeric differences in
        # the Transformer attention head.
        query = tf.multiply(query, 1.0 / math.sqrt(float(self._key_dim)))

        # Take the dot product between "query" and "key" to get the raw
        # attention scores.
        attention_scores = tf.einsum(self._dot_product_equation, key,
                                     query)

        attention_scores = self._masked_softmax(
            attention_scores, attention_mask)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_scores_dropout = self._dropout_layer(
            attention_scores, training=training)

        # `context_layer` = [B, T, N, H]
        attention_output = tf.einsum(self._combine_equation,
                                     attention_scores_dropout, value)
        return attention_output, attention_scores

    def call(self,
             query,
             value,
             key=None,
             attention_mask=None,
             return_attention_scores=False,
             training=None):
        if not self._built_from_signature:
            self._build_from_signature(query=query, value=value, key=key)
        if key is None:
            key = value
        if attention_mask is not None:
            attention_mask = attention_mask[:, tf.newaxis, tf.newaxis, :]

        #   N = `num_attention_heads`
        #   H = `size_per_head`
        # `query` = [B, T, N ,H]
        query = self._query_dense(query)

        # `key` = [B, S, N, H]
        key = self._key_dense(key)

        # `value` = [B, S, N, H]
        value = self._value_dense(value)

        attention_output, attention_scores = self._compute_attention(
            query, key, value, attention_mask, training)
        attention_output = self._output_dense(attention_output)

        if return_attention_scores:
            return attention_output, attention_scores
        return attention_output
