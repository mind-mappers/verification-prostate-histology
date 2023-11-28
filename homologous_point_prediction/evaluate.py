from mri_histology_toolkit.register import analyze_pipeline_qualitative, analyze_pipeline_quantitative
from homologous_point_prediction.standard_pipeline import HomologousPointPipeline
from mri_histology_toolkit.data_loader import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

def evaluate(model, output_dir, requires_scaling=False):
    '''
    Evaluate a model and place results in output_dir

    @param model: A loaded keras model or a string path to a saved model
    @param: Output dir, the dir in which to place results
    '''
    if type(model) == str:
        pipeline = HomologousPointPipeline(model_path=model, requires_scaling=requires_scaling)
    else:
        pipeline = HomologousPointPipeline(model=model, requires_scaling=requires_scaling)

    data_loader = DataLoader(config_path="/home/ruchtia/git/method_analysis/configs/test_config.json")

    analyze_pipeline_qualitative(data_loader[0], pipeline, save_path=output_dir)
    analyze_pipeline_quantitative(data_loader, pipeline, save_path=output_dir)


def show_in_out(model, output_dir, id="", requires_scaling=True):
    if type(model) == str:
        pipeline = HomologousPointPipeline(model_path=model, requires_scaling=requires_scaling)
    else:
        pipeline = HomologousPointPipeline(model=model, requires_scaling=requires_scaling)

    data_loader = DataLoader(config_path="/home/ruchtia/git/method_analysis/configs/test_config.json")
    data_dict = data_loader[11]


    def save_or_show(name, fig=None):
        path = os.path.join(output_dir, name)
        if fig is None:
            plt.savefig(path)
        else:
            fig.savefig(path)
        plt.clf()


    pipeline.fit(data_dict)
    sp, ep = pipeline.start_points, pipeline.end_points
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    title = 'Input_vs_Predicted_Output_Points_{0}'.format(id)
    fig.suptitle(title)
    axs[0].imshow(data_dict["rgb_hist"])
    axs[0].scatter(sp[:, 1], sp[:, 0])
    axs[1].imshow(data_dict["unmasked_mri"], cmap="gray")
    axs[1].scatter(ep[:, 1], ep[:, 0])
    save_or_show(title)