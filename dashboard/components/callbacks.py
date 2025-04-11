import numpy as np
import pandas as pd
import plotly.express as px
from dash import Input, Output
from dash.exceptions import PreventUpdate
from loguru import logger
from utils.mechanics import generate_planar_values
from utils.plotting import build_polar_figure, build_scatter_figure
from utils.utils import encode_image

from data.data_processing import prepare_scatter_data
from data.experiment_loader import (get_cached_experiments,
                                    get_image_from_experiment,
                                    get_matrix_from_experiment)


def get_material_properties(data):
    run_id = get_run_id(data)
    mat_type, mat = get_matrix_from_experiment(run_id)
    thetas, Es, Gs, nus = generate_planar_values(
        mat, input_style='standard' if mat_type == 'C' else 'mandel')
    df = pd.DataFrame({
        'theta': thetas*180/np.pi,
        'E': Es,
        'G': Gs,
        'nu': nus
    })

    return df


def get_run_id(data):
    return int(data['points'][0]['hovertext'].split(': ')[1])


def register_callbacks(app):
    logger.info('CALLBACKS: Registering')

    @app.callback(
        Output('scatter-plot',    'figure'),
        [
            Input('x-axis-dropdown', 'value'),
            Input('y-axis-dropdown', 'value'),
            Input('nu-filter', 'value'),
            Input('E-filter', 'value'),
            Input('basis-filter', 'value'),
            Input('mode-filter', 'value'),
            Input('dist-filter', 'value'),
            # Input('rerun-filter', 'value'),
        ],
    )
    def update_scatter_plot_cb(x_metric, y_metric, nu_filter, E_filter, basis_filter, mode_filter, dist_filter):

        filters = {
            'nu': nu_filter,
            'E_min': E_filter,
            'basis_v': basis_filter,
            'extremal_mode': mode_filter,
            # 'dist_type': bool(rerun_filter)
        }
        # print(bool(rerun_filter))
        logger.debug(f"Applied filters: {filters}")
        experiments = get_cached_experiments()
        logger.info(f"{len(experiments)} experiments in update_scatter_plot()")

        scatter_df = prepare_scatter_data(x_metric,
                                          y_metric,
                                          experiments,
                                          filters)

        fig = build_scatter_figure(scatter_df, x_metric, y_metric)

        return fig

    @app.callback(
        [Output('EG-polar-plot', 'figure'),
         Output('Nu-polar-plot', 'figure')],
        Input('scatter-plot', 'hoverData'),
        Input('scatter-plot', 'selectedData')
    )
    def update_polar_plot_cb(hover_data, selected_data):
        logger.debug(f"hoverData: {hover_data}")
        logger.debug(f"selectedData: {selected_data}")

        data = selected_data if selected_data else hover_data

        if data is None:
            raise PreventUpdate

        title = data['points'][0]['hovertext']
        colors = px.colors.qualitative.D3
        df = get_material_properties(data)
        eg_fig = build_polar_figure(
            df, 'theta', ['E', 'G'], r_limits=[0, 0.55], colors=colors, title=title)
        nu_fig = build_polar_figure(
            df, 'theta', 'nu', [colors[2]], r_limits=[-1, 1.05], title=title)

        return eg_fig, nu_fig

    @app.callback(
        Output('hover-image', 'src'),
        #  Output('hover-text', 'children')],
        Input('scatter-plot', 'hoverData'),
        Input('scatter-plot', 'selectedData')
    )
    def update_array_img_cb(hover_data, selected_data):
        data = selected_data if selected_data else hover_data
        if data:
            # run_id = int(data['points'][0]['hovertext'].split(': ')[1])
            run_id = get_run_id(data)
            try:
                exp_img = get_image_from_experiment(run_id)
                img_src = f"data:image/png;base64,{encode_image(exp_img)}"
                return img_src  # , ''
            except Exception as e:
                logger.error(
                    f"Exception occured while loading image for run {run_id}: {e}")
        logger.debug("No data for datapoint")
        return ''

    @app.callback(
        [
            Output('basis-filter', 'value'),
            Output('mode-filter', 'value'),
            Output('nu-filter', 'value'),
            Output('E-filter', 'value'),
        ],
        [Input('clear-button', 'n_clicks')]
    )
    def clear_dropdowns_cb(n_clicks):
        if n_clicks is None:
            raise PreventUpdate
        return [], [], [], []

    logger.info("CALLBACKS: Finished")
