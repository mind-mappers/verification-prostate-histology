from custom_keras_layers import Correlation3D, TiledCorrelation3D
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import numpy as np
import warnings
import tensorflow as tf
warnings.filterwarnings("ignore")
tf.enable_eager_execution()

def test_correlation_3d():
    A = tf.convert_to_tensor([
        [
        [[1,0,0], [0,1,0]],
        [[0,0,0], [0,0,1]],
        [[0,0,0], [0,1,0]],
        ]
    ], dtype=tf.float32)
    B = tf.convert_to_tensor([
        [
        [[1,0,0], [0,1,0]],
        [[0,0,0], [0,0,0]],
        [[0,0,0], [0,0,1]],
        ]
    ], dtype=tf.float32) # only difference is last item

    expected_out = np.array([
        [
        [[1,0,0,0,0,0], [0,1,0,0,0,0]],
        [[0,0,0,0,0,0], [0,0,0,0,0,1]],
        [[0,0,0,0,0,0], [0,1,0,0,0,0]],
        ]
    ], dtype=np.float32)

    A_in = Input(shape=(A.shape[1:]))
    B_in = Input(shape=(B.shape[1:]))
    out = Correlation3D()([A_in, B_in])
    model = Model([A_in, B_in], out)
    output = model.predict([A, B], steps=1)

    np.testing.assert_array_equal(output, expected_out)
    print("3D Correlation Passed")


def test_tiled_correlation_3d():
    A = tf.convert_to_tensor([
        [
        [[1, 1], [2, 2], [-3, -4]],
        [[3, 3], [4, 4], [-3, -4]],
        [[5,5], [6, 6], [-3, -4]],
        [[-1,-2], [-3, -4], [-3, -4]]
        ],
        [
        [[7,7], [8,8], [-3, -4]],
        [[9,9], [10,10], [-3, -4]],
        [[11,11], [12,12], [-3, -4]],
        [[-3,-3], [-5, -5], [-3, -4]]
        ]
    ], dtype=tf.float32)

    B = tf.convert_to_tensor([
        [
        [[1, 1], [2, 2], [-3, -4]],
        [[3, 3], [4, 4], [-3, -4]],
        [[5,5], [6, 6], [-3, -4]],
        [[-1,-2], [-3, -4], [-3, -4]]
        ],
        [
        [[7,7], [8,8], [-3, -4]],
        [[9,9], [10,10], [-3, -4]],
        [[11,11], [12,12], [-3, -4]],
        [[-3,-3], [-5, -5], [-3, -4]]
        ]
    ], dtype=tf.float32) * -1

    coords = tf.convert_to_tensor([[1,1], [2,1]], dtype=tf.int32)
    tile_shape = (3, 3)

    expected_tile_output = tf.convert_to_tensor([
        [
        [[1, 1], [2, 2], [-3, -4]],
        [[3, 3], [4, 4], [-3, -4]],
        [[5,5], [6, 6], [-3, -4]]
        ],
        [
        [[9,9], [10,10], [-3, -4]],
        [[11,11], [12,12], [-3, -4]],
        [[-3,-3], [-5, -5], [-3, -4]]
        ]
    ], dtype=tf.float32)


    A_in = Input(shape=(A.shape[1:]), batch_size=2)
    B_in = Input(shape=(B.shape[1:]), batch_size=2)
    coord_in = Input(shape=(2,), batch_size=2, dtype=tf.int32)
    tile_corr = TiledCorrelation3D(tile_shape=tile_shape, testing_tiles=True)
    out = tile_corr([A_in, B_in, coord_in])
    model = Model([A_in, B_in, coord_in], out)
    output = model.predict([A, B, coords], steps=1)

    np.testing.assert_array_equal(output, expected_tile_output)
    print("Passed 3D tile check")

test_correlation_3d()
test_tiled_correlation_3d()
