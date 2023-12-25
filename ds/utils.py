import jax
import tensorflow as tf
from flax import struct


@struct.dataclass
class Batch:
    inputs: jax.Array
    labels: jax.Array


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def make_tf_record_example(example):
    example = tf.train.Example(
        features=tf.train.Features(
            feature=dict(tokens=_bytes_feature(example.tobytes()))
        )
    )

    return example


def decode_tf_record_example(record_bytes):
    example = tf.io.parse_single_example(
        # Data
        record_bytes,
        # Schema
        features={"tokens": tf.io.FixedLenFeature([], tf.string, default_value="")},
    )

    return tf.io.decode_raw(input_bytes=example["tokens"], out_type=tf.uint16)
