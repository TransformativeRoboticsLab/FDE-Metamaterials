import numpy as np
import pandas as pd
import plotly.express as px
from dash import Input, Output
from dash.exceptions import PreventUpdate
from data.data_processing import prepare_scatter_data
from data.experiment_loader import (get_cached_experiments,
                                    get_image_from_experiment,
                                    get_matrix_from_experiment)
from loguru import logger
from utils.mechanics import generate_planar_values
from utils.plotting import build_scatter_figure
from utils.utils import encode_image


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
            Input('dist-filter', 'value')
        ],
    )
    def update_scatter_plot_cb(x_metric, y_metric, nu_filter, E_filter, basis_filter, mode_filter, dist_filter):

        filters = {
            'nu': nu_filter,
            'E_min': E_filter,
            'basis_v': basis_filter,
            'extremal_mode': mode_filter,
            'dist_type': dist_filter
        }
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
        Output('hover-image', 'src'),
        #  Output('hover-text', 'children')],
        Input('scatter-plot', 'hoverData')
    )
    def update_array_img_cb(hover_data):
        if hover_data:
            run_id = int(hover_data['points'][0]['hovertext'].split(': ')[1])
            try:
                exp_img = get_image_from_experiment(run_id)
                img_src = f"data:image/png;base64,{encode_image(exp_img)}"
                return img_src  # , ''
            except Exception as e:
                logger.error(
                    f"Exception occured while loading image for run {run_id}: {e}")
        logger.debug("No hover_data for datapoint")
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
    def clear_dropdowns(n_clicks):
        if n_clicks is None:
            raise PreventUpdate
        return [], [], [], []

    @app.callback(
        [Output('EG-polar-plot', 'figure'),
         Output('Nu-polar-plot', 'figure')],
        Input('scatter-plot', 'hoverData')
    )
    def update_polar_plot(hover_data):
        if hover_data is None:
            raise PreventUpdate

        run_id = int(hover_data['points'][0]['hovertext'].split(': ')[1])
        logger.debug
        mat_type, mat = get_matrix_from_experiment(run_id)
        thetas, Es, Gs, nus = generate_planar_values(
            mat, input_style='standard' if mat_type == 'C' else 'mandel')
        df = pd.DataFrame({
            'theta': thetas*180/np.pi,
            'E': Es,
            'G': Gs,
            'nu': nus
        })

        # Create the EG polar plot
        eg_fig = px.line_polar(
            df,
            r='E',
            theta='theta',
            color_discrete_sequence=['red'],
        )
        eg_fig.data[0].name = 'E'
        eg_fig.data[0].showlegend = True
        eg_fig.add_scatterpolar(
            r=df['G'],
            theta=df['theta'],
            mode='lines',
            name='G',
            line=dict(color='blue')
        )

        eg_fig.update_layout(
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            transition=dict(
                duration=500,
                easing='cubic-in-out',
                ordering='layout first'
            ),
            polar=dict(
                angularaxis=dict(
                    direction="counterclockwise",
                    # Puts 0 degrees at the right (3 o'clock position)
                    rotation=90
                )
            )
        )

        # Create the Nu polar plot
        nu_fig = px.line_polar(
            df,
            r='nu',
            theta='theta',
            color_discrete_sequence=['green'],
        )
        nu_fig.data[0].name = 'ν'
        nu_fig.data[0].showlegend = True

        nu_fig.update_layout(
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            transition=dict(
                duration=500,
                easing='cubic-in-out',
                ordering='layout first'
            ),
            polar=dict(
                radialaxis=dict(
                    range=[-1, 1]  # Set a fixed range for Poisson's ratio
                ),
                angularaxis=dict(
                    direction="counterclockwise",
                    # Puts 0 degrees at the right (3 o'clock position)
                    rotation=90
                )
            )
        )

        return eg_fig, nu_fig

    logger.info("CALLBACKS: Finished")
