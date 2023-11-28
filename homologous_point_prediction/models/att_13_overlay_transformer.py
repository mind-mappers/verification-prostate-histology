from homologous_point_prediction.models.custom_layers import AddOnes, ImagePatchEncoder3, Softmax, SquareBased, DotProductCorrelation, L2Norm, ModelToggle, ImagePatchEncoder2, EdgePointAugmentation, ModalityEncoder, HardCodedPositions, RangeOut, GatherPatches, Patches, ProjectPoints, PositionEncoder, ImagePatchEncoder, ExpandToBatch, ActivePointMask, ActivePointMaskMultiplication
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import LayerNormalization, BatchNormalization, DepthwiseConv2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Attention
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Softmax
from tensorflow.keras.layers import Add, Multiply, Lambda
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import Zeros
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, Conv3D
from homologous_point_prediction.models.multihead_attention import MultiHeadAttention
from homologous_point_prediction.models.multihead_attention import gelu
import tensorflow as tf
import tensorflow_addons as tfa
import math


def localized_self(embeddings, num_heads=4, proj_dim=64, expand_ratio=3):
    # https://arxiv.org/pdf/2104.05707.pdf
    reshaped = Reshape((32, 32, proj_dim))(embeddings)
    reshaped = Conv2D(proj_dim*expand_ratio, kernel_size=(1, 1))(reshaped)
    reshaped = tf.keras.layers.DepthwiseConv2D((3,3), strides=(1, 1), padding="same")(reshaped)
    reshaped = Conv2D(proj_dim, kernel_size=(1, 1))(reshaped)
    return Reshape((-1, proj_dim))(reshaped)

def SEP_BLOCK(embeddings_1, num_patches, proj_dim, expand_ratio=4, num_heads=4, cross_attention=False):
    x = LayerNormalization(epsilon=1e-6)(embeddings_1)
    x = Add()([embeddings_1, MultiHeadAttention(num_heads=num_heads, key_dim=proj_dim, dropout=0.1)(x, x, x)])

    x2 = LayerNormalization(epsilon=1e-6)(x)
    x2 = localized_self(x2, proj_dim=proj_dim, expand_ratio=expand_ratio)

    x_final = Add()([x, x2])
    return x_final

def SEP_BLOCKS(x, patch_count, projection_dim, num_layers=6, cross_attention=2):
    for i in range(num_layers):
        x = SEP_BLOCK(x, patch_count, projection_dim, cross_attention=(i>=num_layers-cross_attention))
    return x

def get_path(num_patches, patch_size, projection_dim, name=""):
    patches = Input(shape=(num_patches, patch_size**2))
    position_embeddings = PositionEncoder(num_patches, projection_dim, name=name+"_position_encoder")(patches)
    x = ImagePatchEncoder3(patch_size, projection_dim, name=name+"_patch_encoder")(patches)
    y = Add()([x, position_embeddings])
    z = SEP_BLOCKS(y, num_patches, projection_dim, num_layers=6, cross_attention=0)
    return Model(patches, z, name=name)

def get_model(image_shape=(512,512,1), learning_rate=0.001, points_per_input=1):
    projection_dim = 128
    patch_size = 16
    num_patches = (512 // patch_size)**2
    patch_per_row = (512 // patch_size)

    # Set up inputs
    fixed_image = Input(shape=image_shape)
    moving_image = Input(shape=image_shape)
    points = Input(shape=(2,), dtype=tf.int32) if points_per_input == 1 else Input(shape=(points_per_input, 2), dtype=tf.int32)
    fixed_modality = Input(shape=(1,), dtype=tf.int32)
    moving_modality = Input(shape=(1,), dtype=tf.int32)

    # Cut patches
    fixed_patches = Patches(patch_size)(fixed_image)
    moving_patches = Patches(patch_size)(moving_image)

    histology_path = get_path(num_patches=num_patches, patch_size=patch_size, projection_dim=projection_dim,name="hist_path")
    mr_path = get_path(num_patches=num_patches, patch_size=patch_size, projection_dim=projection_dim, name="mr_path")

    toggle = ModelToggle()
    encoded_fixed = toggle([fixed_modality, histology_path(fixed_patches), mr_path(fixed_patches)])
    encoded_moving = toggle([moving_modality, histology_path(moving_patches), mr_path(moving_patches)])
    #dummies = tf.keras.layers.Embedding(2, projection_dim, embeddings_initializer="zeros", trainable=False)(moving_modality)
    #encoded_fixed = Add()([encoded_fixed, dummies])
    #encoded_moving = Add()([encoded_moving, dummies])


    initial_estimate_embeddings, offsets = GatherPatches(image_shape, projection_dim=projection_dim, name="gather")(encoded_fixed, points, return_offsets=True)
    initial_estimate_embeddings = DotProductCorrelation(name="attention")(initial_estimate_embeddings, encoded_moving)
    out = SquareBased(patch_per_row, patch_size)(initial_estimate_embeddings)
    out = Add()([out, offsets])
    print(offsets)
    out = ActivePointMaskMultiplication(name="active")(out, points) # num points

    model = Model([fixed_image, moving_image, points, fixed_modality, moving_modality], out)
    return model