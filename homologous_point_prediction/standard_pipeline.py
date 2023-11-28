from homologous_point_prediction.pipeline_components.point_placement import compute_points, filter_points
from homologous_point_prediction.pipeline_components.warp import warp_image, warp_points
from homologous_point_prediction.data_processing.helpers import reverse_center_prostate, center_prostate, pad_points
from mri_histology_toolkit.register import RegisterPipeline
import matplotlib.pyplot as plt
from homologous_point_prediction.models.helpers import load_model
import numpy as np

def make_grid():
    x = np.linspace(0, 512, 512//45 + 1)
    y = np.linspace(0, 512,  512//45 + 1)
    xv, yv = np.meshgrid(x, y)
    return np.concatenate((yv.reshape((-1, 1)), xv.reshape(-1, 1)), axis=-1)

POINT_PLACEMENT_DIST_THRESH = 40

class HomologousPointPipeline(RegisterPipeline):

    def __init__(self, model_path=None, model=None, point_method="sift", requires_scaling=False, model_name="homologous_point_pipeline"):
        assert (model_path is None) ^ (model is None)
        super().__init__()
        self.model_name = model_name
        self.model_path = model_path
        self.model = model
        # sift, human, grid
        self.point_method = point_method
        self.requires_scaling = requires_scaling
    
    def batch_predict(self, model, fixed, moving, points, num_points=75):
        print(fixed.shape)
        print(moving.shape)
        print(points.shape)
        return model.predict([fixed.reshape((1,512,512,1)), moving.reshape((1,512,512,1)), pad_points(points, num_points).reshape(1, num_points, 2), np.zeros((1,1), dtype=np.int32), np.ones((1,1), dtype=np.int32)]).reshape((-1, 2))[:len(points)]

    def fit(self, data_dict):
        super().fit(data_dict)
        scale_pad = 50
        # Load Model
        model = self.model if self.model is not None else load_model(self.model_path)

        # Input point placement
        if self.point_method == 'sift':
            input_points = np.array(compute_points(data_dict["grayscale_hist"] * 255, dist_thresh=POINT_PLACEMENT_DIST_THRESH))
        elif self.point_method == 'human':
            input_points = data_dict["hist_points"]
        elif self.point_method == 'grid':
            input_points = make_grid()

        input_points = filter_points(input_points, data_dict["grayscale_hist"])
        input_points = input_points[:75, :]
        self.start_points = input_points

        fixed_image = data_dict["grayscale_hist"]
        moving_image = data_dict["unmasked_mri"]

        if self.requires_scaling:
            fixed_image, input_points, _ = center_prostate(data_dict["grayscale_hist"], input_points, other=None, padding=scale_pad, mask_points=data_dict["hist_points"])
            _, _, moving_image = center_prostate(data_dict["unmasked_mri"], data_dict["mri_points"], other=moving_image, padding=scale_pad)

        output_points = self.batch_predict(model, fixed_image, moving_image, input_points)

        # If necessary scale points back to the 512 by 512 shape
        if self.requires_scaling:
            output_points = reverse_center_prostate(data_dict["unmasked_mri"], output_points, padding=scale_pad, mask_points=data_dict["mri_points"])

        # Save the values required for replicating process
        self.end_points = output_points

    def warp_points(self, points):
        super().warp_points(points)
        return warp_points(points, self.start_points, self.end_points)

    def warp_surface(self, surface):
        super().warp_surface(surface)
        return warp_image(surface, self.start_points, self.end_points)


