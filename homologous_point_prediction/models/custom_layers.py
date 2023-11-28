from tensorflow.keras import backend
from tensorflow.keras import layers
from tensorflow.keras.layers import LayerNormalization, Add
from tensorflow.keras import backend as K
import builtins
import tensorflow as tf
import numpy as np



class Softmax(layers.Layer):
    def __init__(self, **kwargs):
        super(Softmax, self).__init__()
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config.update({})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, values):
        return tf.nn.softmax(values, axis=-1)


class DotProductCorrelation(layers.Layer):
    def __init__(self, **kwargs):
        super(DotProductCorrelation, self).__init__()
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config.update({})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, query, values):
        return tf.nn.softmax(tf.matmul(query, values, transpose_b=True))

class L2Norm(layers.Layer):
    def __init__(self, **kwargs):
        super(L2Norm, self).__init__()
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config.update({})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, values):
        return K.l2_normalize(values, axis=1)

class SquareBased(layers.Layer):
    def __init__(self, patch_per_side, patch_size, **kwargs):
        super(SquareBased, self).__init__()
        self.patch_per_side = patch_per_side
        self.patch_size = patch_size
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config.update({'patch_per_side': self.patch_per_side, 'patch_size': self.patch_size})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, softmaxes):
        ranges = (tf.range(start=0, limit=self.patch_per_side, delta=1, dtype=tf.float32) * self.patch_size) + self.patch_size//2
        X, Y = tf.meshgrid(ranges, ranges)
        x = tf.reduce_sum(softmaxes * tf.reshape(X, [-1]), axis=-1)
        y = tf.reduce_sum(softmaxes * tf.reshape(Y, [-1]), axis=-1)
        result = tf.stack((y, x), axis=-1)
        return result

class ModelToggle(layers.Layer):
    '''
    Histology images = 0
    MRI images = 1
    '''
    def __init__(self, **kwargs):
        super(ModelToggle, self).__init__()
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config.update({})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, ins):
        flags, hist_called, mri_called = ins
        return tf.where(tf.equal(tf.reshape(flags, [-1, 1, 1]), 1), hist_called, mri_called)

class AddOnes(layers.Layer):
    def __init__(self, **kwargs):
        super(AddOnes, self).__init__()
        super().__init__(**kwargs)
        self.add = Add()

    def get_config(self):
        config = super().get_config()
        config.update({})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, ins):
        return self.add([tf.ones(tf.shape(ins)), ins])

def loss(a=0.5, b=0.5):
    def custom_loss(gt, pred):
        # [batch_size, points_per, 2]
        # Padded with zeros at end of each points per
        gt = tf.cast(gt, "double")
        pred = tf.cast(pred, "double")

        # Deal with the zero padding points
        valid_points = tf.cast(tf.reshape(tf.reduce_sum(gt, axis=-1), [tf.shape(gt)[0], tf.shape(gt)[1]]) > 0, "double")
        valid_point_counts = tf.cast(tf.reduce_sum(valid_points, axis=-1), "double")

        # Calculate centroids ignoring zero points
        centroids = tf.reduce_sum(gt, axis=-2) / valid_point_counts
        centroids = tf.reshape(centroids, [tf.shape(centroids)[0], 1, tf.shape(centroids)[1]])

        # Calculate distances
        differnce_from_centroids = tf.cast(gt, "double") - centroids
        distance_from_centroids = tf.math.sqrt(tf.reduce_sum(differnce_from_centroids**2, axis=-1))

        # Weight the center points
        x_distances = tf.math.abs(differnce_from_centroids[:, :, 1] * valid_points)
        x_weights = x_distances / tf.reshape(tf.reduce_max(x_distances, axis=-1), [-1, 1])
        x_weights = (x_weights * -1) + 1


        # Calc scaling factor 
        max_from_center = tf.reshape(tf.reduce_max(distance_from_centroids * valid_points, axis=-1), [-1, 1])
        scaling_factors =  (valid_points * distance_from_centroids) / max_from_center
        #scaling_factors = (scaling_factors + x_weights) / 2 # add in the x scale
        squared_distances = tf.reduce_sum((tf.cast(gt, "double") - tf.cast(pred, "double"))**2, axis=-1) / tf.reshape(valid_point_counts, [-1, 1])
        return tf.reduce_sum((a * squared_distances) + (b * squared_distances * scaling_factors), axis=-1)

    return custom_loss

class GatherPatches(layers.Layer):
    """
    Takes in encoded patches and coordinates sets. Gathers the embedded patches correpsonding
    to the points
    """
    def __init__(self, image_size=(512, 512, 1), projection_dim=64,  **kwargs):
        super(GatherPatches, self).__init__()
        self.image_size = image_size
        self.projection_dim = projection_dim
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config.update({
            "image_size": self.image_size,
            "projection_dim": self.projection_dim,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


    def call(self, embeddings, point_coordinates, return_offsets=False):
        # embeddings [batch_size, patches_per_image, embedding_dim]
        # point_coordinates [batch_size, points_per_image, 2]
        embedding_shape = tf.shape(embeddings)
        batch_size = embedding_shape[0]
        patches_per_image = embedding_shape[1]
        embedding_dim = embedding_shape[2]
        patches_per_row = tf.cast(tf.math.sqrt(tf.cast(patches_per_image, tf.float32)), tf.int32)
        patch_size = tf.math.floordiv(tf.cast(self.image_size[0], tf.float32), tf.cast(patches_per_row, tf.float32))

        # Calculate offsets in each square
        offsets = point_coordinates % tf.cast(patch_size, tf.int32)
        patch_center = tf.cast(patch_size, tf.int32)//2
        offsets_from_center = offsets - patch_center
        offsets_from_center = tf.cast(offsets_from_center, tf.float32)
        
        # Calcualte where points fall in grid
        simple_points = tf.reshape(point_coordinates, [-1, 2])
        simple_points = tf.math.floordiv(simple_points, tf.cast(patch_size, tf.int32))
        simple_points = (simple_points[:, 0] * patches_per_row) + simple_points[:, 1]
        
        # Compensate for batch size
        image_starts = tf.range(start=0, limit=batch_size, delta=1, dtype=tf.int32) * patches_per_image
        image_starts = tf.reshape(image_starts, [-1, 1])
        simple_points = tf.reshape(simple_points, [batch_size, -1])
        simple_points = simple_points + image_starts
        simple_points = tf.reshape(simple_points, [-1])

        # Take the relevant rows
        embeddings = tf.reshape(embeddings, [-1, self.projection_dim])
        taken_embeddings = tf.gather(embeddings, indices=simple_points, axis=0)
        
        # Final reshape
        taken_embeddings = tf.reshape(taken_embeddings, [batch_size, -1, self.projection_dim])
        # If we have one point per image, zqueeze to 2d
        if return_offsets:
            return taken_embeddings, offsets_from_center
        return taken_embeddings

def transform(images, transforms, fill_mode='reflect', fill_value=0.0, interpolation='bilinear', output_shape=None, name=None):
    
    with backend.name_scope(name or 'transform'):
        if output_shape is None:
            output_shape = tf.shape(images)[1:3]
            if not tf.executing_eagerly():
                output_shape_value = tf.get_static_value(output_shape)
                if output_shape_value is not None:
                    output_shape = output_shape_value

        output_shape = tf.convert_to_tensor(
            output_shape, tf.int32, name='output_shape')

        if not output_shape.get_shape().is_compatible_with([2]):
            raise ValueError('output_shape must be a 1-D Tensor of 2 elements: '
                             'new_height, new_width, instead got '
                             '{}'.format(output_shape))

        fill_value = tf.convert_to_tensor(fill_value, tf.float32, name='fill_value')

        return tf.raw_ops.ImageProjectiveTransformV2(
            images=images,
            output_shape=output_shape,
            transforms=transforms,
            fill_mode=fill_mode.upper(),
            interpolation=interpolation.upper())

def get_translation_matrix(translations, name=None):
    with backend.name_scope(name or 'translation_matrix'):
        num_translations = tf.shape(translations)[0]
        # The translation matrix looks like:
        #     [[1 0 -dx]
        #      [0 1 -dy]
        #      [0 0 1]]
        # where the last entry is implicit.
        # Translation matrices are always float32.
        return tf.concat(
            values=[
                tf.ones((num_translations, 1), tf.float32),
                tf.zeros((num_translations, 1), tf.float32),
                -translations[:, 0, None],
                tf.zeros((num_translations, 1), tf.float32),
                tf.ones((num_translations, 1), tf.float32),
                -translations[:, 1, None],
                tf.zeros((num_translations, 2), tf.float32),
            ],
            axis=1)

class ShiftImage(layers.Layer):
    '''
    Randomly shift an image and coordinates during training only
    '''

    def __init__(self, point_window_radius, **kwargs):
        super(ShiftImage, self).__init__()
        self.point_window_radius = point_window_radius
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config.update({
            'point_window_radius': self.point_window_radius
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, img, pts):

        training = False
        try:
            training = builtins.in_training_session
        except:
            print("In training session is not present in globals")

        print("In training session", training)
        def random_shift_image():
            pts_type = pts.dtype
            sh = tf.shape(img)
            batch_size, height, width = sh[0], sh[1], sh[2]
            orig_point_shape = tf.shape(pts)
            points = tf.cast(tf.reshape(pts, [-1, 2]), dtype=tf.int32)
            
            space_top = tf.cast(points[:, 0] - self.point_window_radius, tf.float32)
            space_bottom = tf.cast(height - points[:, 0] - self.point_window_radius, tf.float32)
            space_left = tf.cast(points[:, 1] - self.point_window_radius, tf.float32)
            space_right = tf.cast(width - points[:, 1] - self.point_window_radius, tf.float32)
            
            height_translate = tf.random.uniform(
                shape=[batch_size],                                  
                minval=(space_top * -1),
                maxval=(space_bottom),
                dtype=tf.float32)

            width_translate = tf.random.uniform(
                shape=[batch_size],
                minval=(space_left * -1),
                maxval=space_right,
                dtype=tf.float32)

            width_translate = tf.reshape(width_translate, [-1, 1])
            height_translate = tf.reshape(height_translate, [-1, 1])
            
            translations = tf.cast(tf.concat([width_translate, height_translate], axis=1), dtype=tf.float32)

            shifted_images = transform(
                img,
                get_translation_matrix(translations),
                interpolation='bilinear',
                fill_mode='constant')
            shifted_points = points + tf.cast(tf.concat([height_translate, width_translate], axis=1), tf.int32)
            return shifted_images, tf.cast(tf.reshape(shifted_points, orig_point_shape), pts_type)
        return tf.cond(tf.constant(training, dtype=tf.bool), random_shift_image, lambda: (img, pts))



class ProjectPoints(layers.Layer):
    '''
    Take a list of coordiantes and create a tensor
    of image shape. Values == 1 mark where points exist.

    Note: This only works for single-point batches at the moment
    '''
    def __init__(self, image_size, **kwargs):
        super(ProjectPoints, self).__init__()
        self.image_size = image_size
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config.update({
            'image_size': self.image_size
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, points):
        batch_size = tf.shape(points)[0]
        # Calcualte 1D array of 1d coordinates of start of each image in batch
        image_starts = tf.range(start=0, limit=tf.shape(points)[0], delta=1, dtype=tf.int32) * np.prod(self.image_size).astype(np.int32)
        # Ensure the points batches are flattened
        points = tf.reshape(points, [-1, 2])
        # Convert to 1d coord part 1.0
        flattened_points = (points[:, 0] * self.image_size[1] + points[:, 1]) * self.image_size[2]
        # Add the image starts, works for any number of points per image
        col_image_starts = tf.reshape(image_starts, [batch_size, 1])
        flattened_points = tf.reshape(tf.reshape(flattened_points, [batch_size, -1]) + col_image_starts, [-1])
        # Apply ones to results
        num_points = tf.size(flattened_points)
        result = tf.scatter_nd(tf.reshape(flattened_points, [-1, 1]), tf.ones(num_points), [np.prod(self.image_size) * batch_size])
        return tf.reshape(result, (batch_size, ) + self.image_size)



class Patches(layers.Layer):
    '''
    Cuts an image into patches
    '''
    def __init__(self, patch_size, image_size=(512, 512, 1), **kwargs):
        super(Patches, self).__init__()
        self.patch_size = patch_size
        self.image_size = image_size
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config.update({
            'patch_size': self.patch_size,
            'image_size': self.image_size
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, (self.image_size[0]//self.patch_size)**2, patch_dims])
        return patches

class ActivePointMask(layers.Layer):
    def __init__(self, **kwargs):
        super(ActivePointMask, self).__init__()
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config.update({
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


    def call(self, points):
        # Take in [batch_size, num_points_per_image, num_pixels_in_patch]
        col_masks = tf.reduce_sum(points, axis=-1)
        return tf.clip_by_value(col_masks, clip_value_min=0, clip_value_max=1)

class ActivePointMaskMultiplication(layers.Layer):
    def __init__(self, **kwargs):
        super(ActivePointMaskMultiplication, self).__init__()
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config.update({
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


    def call(self, features, points):
        # Take in [batch_size, num_patches_per_image, num_pixels_in_patch]
        col_masks = tf.reshape(tf.reduce_sum(points, axis=-1), [tf.shape(features)[0], tf.shape(features)[1], 1])
        return features * tf.cast(tf.clip_by_value(col_masks, clip_value_min=0, clip_value_max=1), tf.float32)

class EdgePointAugmentation(layers.Layer):
    def __init__(self, **kwargs):
        super(EdgePointAugmentation, self).__init__()
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config.update({})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


    def call(self, features, points):
        # Take in [batch_size, num_patches_per_image, num_pixels_in_patch]

        training = False
        try:
            training = builtins.in_training_session
        except:
            print("In training session is not present in globals")
        y_inputs = points[:, :, :1]
        thresh = tf.reduce_max(y_inputs) - 30
        col_masks = tf.cast(y_inputs > thresh, tf.float32)
        return tf.cond(tf.constant(training, dtype=tf.bool), lambda: features * col_masks, lambda: features)


class PointAttentionMask(layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super(PointAttentionMask, self).__init__()
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_patches': self.num_patches,
            'projection_dim': self.projection_dim,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


    def call(self, points):
        #Get col mask for each image (batch_size, num_patches, 1)
        col_masks = tf.reshape(tf.reduce_sum(points, axis=-1), [tf.shape(points)[0], self.num_patches, 1])
        # Repeat for every key/value
        #repeated_masks = tf.repeat(col_masks, repeats=[self.num_patches], axis=2)
        #repeated_masks = tf.reshape(repeated_masks, [tf.shape(points)[0], self.num_patches, 64])
        return col_masks


class PointPatchEncoder(layers.Layer):
    '''
    Takes in patches with  points encoded as "1" and non
    points encoded as 0
    '''
    def __init__(self, num_patches, projection_dim, **kwargs):
        super(PointPatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.point_embedding = layers.Embedding(input_dim=2, output_dim=projection_dim)
        self.norm = LayerNormalization()
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_patches': self.num_patches,
            'projection_dim': self.projection_dim,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


    def call(self, points):
        point_mask = tf.reshape(tf.reduce_sum(points, axis=-1), [tf.shape(points)[0], -1])
        return self.point_embedding(point_mask)

class ModalityEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, num_start, **kwargs):
        super(ModalityEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.num_start = num_start
        self.modality_embedding = layers.Embedding(input_dim=2, output_dim=projection_dim)
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_patches': self.num_patches,
            'projection_dim': self.projection_dim,
            'num_start': self.num_start
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, patches):
        positions = tf.cast(tf.range(start=0, limit=self.num_patches, delta=1) < self.num_start, 'int32')
        embeddings = self.modality_embedding(positions)
        return embeddings

class PositionEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super(PositionEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_patches': self.num_patches,
            'projection_dim': self.projection_dim,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, x):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        embeddings = self.position_embedding(positions)
        return tf.repeat(tf.expand_dims(embeddings, axis=0), repeats=[tf.shape(x)[0]], axis=0)

class HardCodedPositions(layers.Layer):
    def __init__(self, **kwargs):
        super(HardCodedPositions, self).__init__()
        self.embeddings = tf.constant(
            [[0,0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 8], [0, 9], [0, 10], [0, 11], [0, 12], [0, 13], [0, 14], [0, 15], [0, 17], [0, 18], [00, 19], [00, 20], [0, 21], [0, 22], [0, 23], [0, 24], [0, 25], [0, 26], [0, 27], [0, 28], [0, 29], [0, 30], [0, 31],
            [1,0], [1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8], [1, 8], [1, 9], [1, 10], [1, 11], [1, 12], [1, 13], [1, 14], [1, 15], [1, 17], [1, 18], [1, 19], [1, 20], [1, 21], [1, 22], [1, 23], [1, 24], [1, 25], [1, 26], [1, 27], [1, 28], [1, 29], [1, 30], [1, 31],
            [2,0], [2, 1], [2, 2], [2, 3], [2, 4], [2, 5], [2, 6], [2, 7], [2, 8], [2, 8], [2, 9], [2, 10], [2, 11], [2, 12], [2, 13], [2, 14], [2, 15], [2, 17], [2, 18], [2, 19], [2, 20], [2, 21], [2, 22], [2, 23], [2, 24], [2, 25], [2, 26], [2, 27], [2, 28], [2, 29], [2, 30], [2, 31],
            [3,0], [3, 1], [3, 2], [3, 3], [3, 4], [3, 5], [3, 6], [3, 7], [3, 8], [3, 8], [3, 9], [3, 10], [3, 11], [3, 12], [3, 13], [3, 14], [3, 15], [3, 17], [3, 18], [3, 19], [3, 20], [3, 21], [3, 22], [3, 23], [3, 24], [3, 25], [3, 26], [3, 27], [3, 28], [3, 29], [3, 30], [3, 31],
            [4,0], [4, 1], [4, 2], [4, 3], [4, 4], [4, 5], [4, 6], [4, 7], [4, 8], [4, 8], [4, 9], [4, 10], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 17], [4, 18], [4, 19], [4, 20], [4, 21], [4, 22], [4, 23], [4, 24], [4, 25], [4, 26], [4, 27], [4, 28], [4, 29], [4, 30], [4, 31],
            [5,0], [5, 1], [5, 2], [5, 3], [5, 4], [5, 5], [5, 6], [5, 7], [5, 8], [5, 8], [5, 9], [5, 10], [5, 11], [5, 12], [5, 13], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [5, 22], [5, 23], [5, 24], [5, 25], [5, 26], [5, 27], [5, 28], [5, 29], [5, 30], [5, 31],
            [6,0], [6, 1], [6, 2], [6, 3], [6, 4], [6, 5], [6, 6], [6, 7], [6, 8], [6, 8], [6, 9], [6, 10], [6, 11], [6, 12], [6, 13], [6, 14], [6, 15], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 22], [6, 23], [6, 24], [6, 25], [6, 26], [6, 27], [6, 28], [6, 29], [6, 30], [6, 31],
            [7,0], [7, 1], [7, 2], [7, 3], [7, 4], [7, 5], [7, 6], [7, 7], [7, 8], [7, 8], [7, 9], [7, 10], [7, 11], [7, 12], [7, 13], [7, 14], [7, 15], [7, 17], [7, 18], [7, 19], [7, 20], [7, 21], [7, 22], [7, 23], [7, 24], [7, 25], [7, 26], [7, 27], [7, 28], [7, 29], [7, 30], [7, 31],
            [8,0], [8, 1], [8, 2], [8, 3], [8, 4], [8, 5], [8, 6], [8, 7], [8, 8], [8, 8], [8, 9], [8, 10], [8, 11], [8, 12], [8, 13], [8, 14], [8, 15], [8, 17], [8, 18], [8, 19], [8, 20], [8, 21], [8, 22], [8, 23], [8, 24], [8, 25], [8, 26], [8, 27], [8, 28], [8, 29], [8, 30], [8, 31],
            [9,0], [9, 1], [9, 2], [9, 3], [9, 4], [9, 5], [9, 6], [9, 7], [9, 8], [9, 8], [9, 9], [9, 10], [9, 11], [9, 12], [9, 13], [9, 14], [9, 15], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 22], [9, 23], [9, 24], [9, 25], [9, 26], [9, 27], [9, 28], [9, 29], [9, 30], [9, 31],
            [10,0], [10, 1], [10, 2], [10, 3], [10, 4], [10, 5], [10, 6], [10, 7], [10, 8], [10, 8], [10, 9], [10, 10], [10, 11], [10, 12], [10, 13], [10, 14], [10, 15], [10, 17], [10, 18], [10, 19], [10, 20], [10, 21], [10, 22], [10, 23], [10, 24], [10, 25], [10, 26], [10, 27], [10, 28], [10, 29], [10, 30], [10, 31],
            [11,0], [11, 1], [11, 2], [11, 3], [11, 4], [11, 5], [11, 6], [11, 7], [11, 8], [11, 8], [11, 9], [11, 10], [11, 11], [11, 12], [11, 13], [11, 14], [11, 15], [11, 17], [11, 18], [11, 19], [11, 20], [11, 21], [11, 22], [11, 23], [11, 24], [11, 25], [11, 26], [11, 27], [11, 28], [11, 29], [11, 30], [11, 31],
            [12,0], [12, 1], [12, 2], [12, 3], [12, 4], [12, 5], [12, 6], [12, 7], [12, 8], [12, 8], [12, 9], [12, 10], [12, 11], [12, 12], [12, 13], [12, 14], [12, 15], [12, 17], [12, 18], [12, 19], [12, 20], [12, 21], [12, 22], [12, 23], [12, 24], [12, 25], [12, 26], [12, 27], [12, 28], [12, 29], [12, 30], [12, 31],
            [13,0], [13, 1], [13, 2], [13, 3], [13, 4], [13, 5], [13, 6], [13, 7], [13, 8], [13, 8], [13, 9], [13, 10], [13, 11], [13, 12], [13, 13], [13, 14], [13, 15], [13, 17], [13, 18], [13, 19], [13, 20], [13, 21], [13, 22], [13, 23], [13, 24], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [13, 30], [13, 31],
            [14,0], [14, 1], [14, 2], [14, 3], [14, 4], [14, 5], [14, 6], [14, 7], [14, 8], [14, 8], [14, 9], [14, 10], [14, 11], [14, 12], [14, 13], [14, 14], [14, 15], [14, 17], [14, 18], [14, 19], [14, 20], [14, 21], [14, 22], [14, 23], [14, 24], [14, 25], [14, 26], [14, 27], [14, 28], [14, 29], [14, 30], [14, 31],
            [15,0], [15, 1], [15, 2], [15, 3], [15, 4], [15, 5], [15, 6], [15, 7], [15, 8], [15, 8], [15, 9], [15, 10], [15, 11], [15, 12], [15, 13], [15, 14], [15, 15], [15, 17], [15, 18], [15, 19], [15, 20], [15, 21], [15, 22], [15, 23], [15, 24], [15, 25], [15, 26], [15, 27], [15, 28], [15, 29], [15, 30], [15, 31],
            [16,0], [16, 1], [16, 2], [16, 3], [16, 4], [16, 5], [16, 6], [16, 7], [16, 8], [16, 8], [16, 9], [16, 10], [16, 11], [16, 12], [16, 13], [16, 14], [16, 15], [16, 17], [16, 18], [16, 19], [16, 20], [16, 21], [16, 22], [16, 23], [16, 24], [16, 25], [16, 26], [16, 27], [16, 28], [16, 29], [16, 30], [16, 31],
            [17,0], [17, 1], [17, 2], [17, 3], [17, 4], [17, 5], [17, 6], [17, 7], [17, 8], [17, 8], [17, 9], [17, 10], [17, 11], [17, 12], [17, 13], [17, 14], [17, 15], [17, 17], [17, 18], [17, 19], [17, 20], [17, 21], [17, 22], [17, 23], [17, 24], [17, 25], [17, 26], [17, 27], [17, 28], [17, 29], [17, 30], [17, 31],
            [18,0], [18, 1], [18, 2], [18, 3], [18, 4], [18, 5], [18, 6], [18, 7], [18, 8], [18, 8], [18, 9], [18, 10], [18, 11], [18, 12], [18, 13], [18, 14], [18, 15], [18, 17], [18, 18], [18, 19], [18, 20], [18, 21], [18, 22], [18, 23], [18, 24], [18, 25], [18, 26], [18, 27], [18, 28], [18, 29], [18, 30], [18, 31],
            [19,0], [19, 1], [19, 2], [19, 3], [19, 4], [19, 5], [19, 6], [19, 7], [19, 8], [19, 8], [19, 9], [19, 10], [19, 11], [19, 12], [19, 13], [19, 14], [19, 15], [19, 17], [19, 18], [19, 19], [19, 20], [19, 21], [19, 22], [19, 23], [19, 24], [19, 25], [19, 26], [19, 27], [19, 28], [19, 29], [19, 30], [19, 31],
            [20,0], [20, 1], [20, 2], [20, 3], [20, 4], [20, 5], [20, 6], [20, 7], [20, 8], [20, 8], [20, 9], [20, 10], [20, 11], [20, 12], [20, 13], [20, 14], [20, 15], [20, 17], [20, 18], [20, 19], [20, 20], [20, 21], [20, 22], [20, 23], [20, 24], [20, 25], [20, 26], [20, 27], [20, 28], [20, 29], [20, 30], [20, 31],
            [21,0], [21, 1], [21, 2], [21, 3], [21, 4], [21, 5], [21, 6], [21, 7], [21, 8], [21, 8], [21, 9], [21, 10], [21, 11], [21, 12], [21, 13], [21, 14], [21, 15], [21, 17], [21, 18], [21, 19], [21, 20], [21, 21], [21, 22], [21, 23], [21, 24], [21, 25], [21, 26], [21, 27], [21, 28], [21, 29], [21, 30], [21, 31],
            [22,0], [22, 1], [22, 2], [22, 3], [22, 4], [22, 5], [22, 6], [22, 7], [22, 8], [22, 8], [22, 9], [22, 10], [22, 11], [22, 12], [22, 13], [22, 14], [22, 15], [22, 17], [22, 18], [22, 19], [22, 20], [22, 21], [22, 22], [22, 23], [22, 24], [22, 25], [22, 26], [22, 27], [22, 28], [22, 29], [22, 30], [22, 31],
            [23,0], [23, 1], [23, 2], [23, 3], [23, 4], [23, 5], [23, 6], [23, 7], [23, 8], [23, 8], [23, 9], [23, 10], [23, 11], [23, 12], [23, 13], [23, 14], [23, 15], [23, 17], [23, 18], [23, 19], [23, 20], [23, 21], [23, 22], [23, 23], [23, 24], [23, 25], [23, 26], [23, 27], [23, 28], [23, 29], [23, 30], [23, 31],
            [24,0], [24, 1], [24, 2], [24, 3], [24, 4], [24, 5], [24, 6], [24, 7], [24, 8], [24, 8], [24, 9], [24, 10], [24, 11], [24, 12], [24, 13], [24, 14], [24, 15], [24, 17], [24, 18], [24, 19], [24, 20], [24, 21], [24, 22], [24, 23], [24, 24], [24, 25], [24, 26], [24, 27], [24, 28], [24, 29], [24, 30], [24, 31],
            [25,0], [25, 1], [25, 2], [25, 3], [25, 4], [25, 5], [25, 6], [25, 7], [25, 8], [25, 8], [25, 9], [25, 10], [25, 11], [25, 12], [25, 13], [25, 14], [25, 15], [25, 17], [25, 18], [25, 19], [25, 20], [25, 21], [25, 22], [25, 23], [25, 24], [25, 25], [25, 26], [25, 27], [25, 28], [25, 29], [25, 30], [25, 31],
            [26,0], [26, 1], [26, 2], [26, 3], [26, 4], [26, 5], [26, 6], [26, 7], [26, 8], [26, 8], [26, 9], [26, 10], [26, 11], [26, 12], [26, 13], [26, 14], [26, 15], [26, 17], [26, 18], [26, 19], [26, 20], [26, 21], [26, 22], [26, 23], [26, 24], [26, 25], [26, 26], [26, 27], [26, 28], [26, 29], [26, 30], [26, 31],
            [27,0], [27, 1], [27, 2], [27, 3], [27, 4], [27, 5], [27, 6], [27, 7], [27, 8], [27, 8], [27, 9], [27, 10], [27, 11], [27, 12], [27, 13], [27, 14], [27, 15], [27, 17], [27, 18], [27, 19], [27, 20], [27, 21], [27, 22], [27, 23], [27, 24], [27, 25], [27, 26], [27, 27], [27, 28], [27, 29], [27, 30], [27, 31],
            [28,0], [28, 1], [28, 2], [28, 3], [28, 4], [28, 5], [28, 6], [28, 7], [28, 8], [28, 8], [28, 9], [28, 10], [28, 11], [28, 12], [28, 13], [28, 14], [28, 15], [28, 17], [28, 18], [28, 19], [28, 20], [28, 21], [28, 22], [28, 23], [28, 24], [28, 25], [28, 26], [28, 27], [28, 28], [28, 29], [28, 30], [28, 31],
            [29,0], [29, 1], [29, 2], [29, 3], [29, 4], [29, 5], [29, 6], [29, 7], [29, 8], [29, 8], [29, 9], [29, 10], [29, 11], [29, 12], [29, 13], [29, 14], [29, 15], [29, 17], [29, 18], [29, 19], [29, 20], [29, 21], [29, 22], [29, 23], [29, 24], [29, 25], [29, 26], [29, 27], [29, 28], [29, 29], [29, 30], [29, 31],
            [30,0], [30, 1], [30, 2], [30, 3], [30, 4], [30, 5], [30, 6], [30, 7], [30, 8], [30, 8], [30, 9], [30, 10], [30, 11], [30, 12], [30, 13], [30, 14], [30, 15], [30, 17], [30, 18], [30, 19], [30, 20], [30, 21], [30, 22], [30, 23], [30, 24], [30, 25], [30, 26], [30, 27], [30, 28], [30, 29], [30, 30], [30, 31],
            [31,0], [31, 1], [31, 2], [31, 3], [31, 4], [31, 5], [31, 6], [31, 7], [31, 8], [31, 8], [31, 9], [31, 10], [31, 11], [31, 12], [31, 13], [31, 14], [31, 15], [31, 17], [31, 18], [31, 19], [31, 20], [31, 21], [31, 22], [31, 23], [31, 24], [31, 25], [31, 26], [31, 27], [31, 28], [31, 29], [31, 30], [31, 31]])
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config.update({})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self):
        return self.embeddings


class ExpandToBatch(layers.Layer):
    def __init__(self, **kwargs):
        super(ExpandToBatch, self).__init__()
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config.update({})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, x, reference_tensor):
        # We get the batch size from any tensor with a batch size
        return tf.repeat(tf.expand_dims(x, axis=0), repeats=[tf.shape(reference_tensor)[0]], axis=0)


class ReduceAttentionScores(layers.Layer):
    def __init__(self, **kwargs):
        super(ReduceAttentionScores, self).__init__()
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config.update({})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, x):
        return tf.math.reduce_max(x, axis=-2)


class ImagePatchEncoder(layers.Layer):
    '''
    Transforms patch and adds learnable position vector.
    If points are provided, applies a learnable embedding for
    the presence of a point.
    '''
    def __init__(self, num_patches, projection_dim, **kwargs):
        super(ImagePatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = layers.Dense(units=projection_dim)
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_patches': self.num_patches,
            'projection_dim': self.projection_dim,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


    def call(self, patchs):
        return self.projection(patchs)



class ImagePatchEncoder2(layers.Layer):
    def __init__(self, patch_size, projection_dim, kernel_size=7, **kwargs):
        super(ImagePatchEncoder2, self).__init__()
        self.patch_size = patch_size
        self.projection_dim = projection_dim
        self.kernel_size = kernel_size
        self.proj = layers.Conv2D(projection_dim, kernel_size=(kernel_size, kernel_size))
        self.re = tf.keras.layers.experimental.preprocessing.Resizing(kernel_size, kernel_size)
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config.update({
            'patch_size': self.patch_size,
            'projection_dim': self.projection_dim,
            'kernel_size': self.kernel_size
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, patches):
        reshaped = tf.reshape(patches, [-1, self.patch_size, self.patch_size, 1])
        conved = self.proj(self.re(reshaped))
        return tf.reshape(conved, [tf.shape(patches)[0], -1, self.projection_dim])

class ImagePatchEncoder3(layers.Layer):
    def __init__(self, patch_size, projection_dim, **kwargs):
        super(ImagePatchEncoder3, self).__init__()
        self.patch_size = patch_size
        self.projection_dim = projection_dim
        self.proj = layers.Conv2D(projection_dim, kernel_size=(4, 4))
        self.pre = layers.Conv2D(64, kernel_size=(2, 2), strides=(2, 2))
        self.pre_2 = layers.Conv2D(64, kernel_size=(2, 2), strides=(2, 2))

        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config.update({
            'patch_size': self.patch_size,
            'projection_dim': self.projection_dim
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, patches):
        reshaped = tf.reshape(patches, [-1, self.patch_size, self.patch_size, 1])
        conved = self.proj(self.pre_2(self.pre(reshaped)))
        return tf.reshape(conved, [tf.shape(patches)[0], -1, self.projection_dim])


class RangeOut(layers.Layer):
    '''
    Range Restricted Output
    '''
    def __init__(self, range_values=[0, 512], **kwargs):
        super(RangeOut, self).__init__()
        self.range_values = range_values
        self.o = layers.Dense(units=2, activation="sigmoid")
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config.update({
            'range_values': self.range_values
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


    def call(self, inputs):
        return (self.o(inputs) * (self.range_values[1] - self.range_values[0])) - self.range_values[0]



class GrayscaleToRGB(layers.Layer):
    '''
        Copies grayscale images to RGB 3-layer
    '''

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        return config

    def call(self, grayscale_images):
        return tf.image.grayscale_to_rgb(grayscale_images, name=None)

class Correlation3D(layers.Layer):
    '''
        Given two input feature maps A, B each of shape (height, width, num features),
        calculate a correlation matrix of shape (height, width, height*width)
        where each sclar value, <i, j, k> corresponds to the correlation score of A[i, j] with B.flatten()[k]
    '''

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        return config

    def call(self, inputs):
        feature_map_a, feature_map_b = inputs
        batch_size, height, width, num_features = feature_map_a.shape

        # Reshape so the pixels are flattened and each column is one feature in A
        # Each row is a feature in B and the pixels are flattened
        # A shape is (batch_size, num_pixels, num features)
        # B shape is (batch_size, num_features, num_pixels)\
        feature_map_a = tf.reshape(feature_map_a, [-1, height*width, num_features])
        feature_map_b = tf.transpose(tf.reshape(feature_map_b, [-1, height*width, num_features]), [0, 2, 1])

        # Multiply the matrices (this operates on each pair in the batch size)
        # resulting in (batch_size, num_pixels, num_pixels)
        # each row corresponds to the flattened index in matrix A
        # each column corresponds to the flattened index in matrix B
        correlation_matrix = feature_map_a @ feature_map_b

        # Restore the shape of A and keep b indices flat
        correlation_matrix = tf.reshape(correlation_matrix, [-1, height, width, height*width])
        return correlation_matrix


class TiledCorrelation3D(layers.Layer):
    '''
        Performs the same task as Correlation3D, but only on one patch of a fixed image.
        Greatly increases the memory efficiency of Correlation3D when only one point
        need be analyzed.
    '''

    def __init__(self, tile_shape=[5,5], testing_tiles=False, **kwargs):
        assert tile_shape[0] == tile_shape[1]
        assert (tile_shape[0] % 2) != 0
        super().__init__()
        self.tile_shape = tile_shape
        self.tile_radius = self.tile_shape[0] // 2
        self.testing_tiles = testing_tiles
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config.update({
            'tile_shape': self.tile_shape,
            'testing_tiles': self.testing_tiles,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def build(self, input_shape):
        _, self.height, self.width, self.depth = input_shape[0]
        self.single_image_size = self.width*self.height*self.depth


        # Build flat representation of tiles taking width, height, and depth into account
        flat_tile_indices = tf.reshape(tf.range(0, self.single_image_size, dtype=tf.int32), [self.height, self.width, self.depth])
        self.flat_tile_indices = tf.reshape(flat_tile_indices[:self.tile_shape[0], :self.tile_shape[1], :], [1, -1])

    @tf.function
    def _get_tiles(self, tensor, coords):
        '''
            @param tensor: A series of feature maps of shape (batch_size, height, width, num_features)
            @param coords : A tensor containing the centers of the points on the feature maps (batch size, 2)
            return: Returns a 3D volume of size (tile_size, tile_size, num_features) for each feature map in the batch
                Each tile will be centered on the coordinates contained in the coords row of corresponding location in batch
                total return tensor shape is (batch_size, tile_size, tile_size, num_features) 
        '''
        batch_size = tf.shape(tensor)[0]
        # Calculate the starting coordinates of each tile in batch
        coords = coords - self.tile_radius # Upper left row and cols
        starting_image_indices = tf.range(0, batch_size, dtype=tf.int32) * self.single_image_size
        coords = tf.reshape(((coords[:, 0] * self.width + coords[:, 1]) * self.depth) + starting_image_indices, [-1, 1])
        
        # coords (batchsize, 1) tensor with the index of the tile for each cube
        # self.flat_tile_indices (batch_size, tile_size) tensor with the indices of the take values
        # This is a broadcast with flat_tile_indices
        flat_take_coords = tf.reshape(coords + self.flat_tile_indices, [-1, 1])
        
        output_shape = [batch_size, self.tile_shape[0], self.tile_shape[1], self.depth]
        tiles = tf.reshape(tf.gather_nd(tf.reshape(tensor, [-1]), flat_take_coords, batch_dims=0), output_shape)
        return tiles


    def call(self, inputs):
        feature_map_a, feature_map_b, points = inputs
        feature_map_a = self._get_tiles(feature_map_a, points)
        if self.testing_tiles:
            return feature_map_a

        # A shape is (batch_size, num_pixels_in_tile, num features)
        # B shape is (batch_size, depth, num_pixels)
        feature_map_a = tf.reshape(feature_map_a, [-1, self.tile_shape[0]**2, self.depth])
        feature_map_b = tf.transpose(tf.reshape(feature_map_b, [-1, self.height*self.width, self.depth]), [0, 2, 1])

        # Multiply the matrices (this operates on each pair in the batch size)
        # resulting in (batch_size, num_pixels, num_pixels)
        # each row corresponds to the flattened index in matrix A
        # each column corresponds to the flattened index in matrix B
        correlation_matrix = feature_map_a @ feature_map_b

        # Restore the shape of A and keep b indices flat
        correlation_matrix = tf.reshape(correlation_matrix, [-1, self.tile_shape[0], self.tile_shape[1], self.height*self.width])
        return correlation_matrix


class PointTranslation(layers.Layer):
    '''
        Translates point coordinates from one image to a resized image/
        Pass in [points, original_layer, new_layer]
        Works with single points per input (batch_size, 2)
        Works with multiple points per input (batch_size, numpoints, 2)
    '''
    def __init__(self, name=None, **kwargs):
        super(PointTranslation, self).__init__(name=name, **kwargs)

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def build(self, input_shape):
        orig_shape = input_shape[1].as_list()[1:]
        new_shape = input_shape[2].as_list()[1:]
        self.original_points_shape = input_shape[0].as_list()[1:]
        self.scaling_factors = tf.convert_to_tensor([orig_shape[0]/new_shape[0], orig_shape[1]/new_shape[1]])

    def call(self, inputs):
        points, _, _ = inputs
        points = tf.reshape(tf.cast(points, tf.float32), [-1, 2])
        return tf.reshape(tf.cast(tf.math.floordiv(points, self.scaling_factors), tf.int32), [-1] + self.original_points_shape)