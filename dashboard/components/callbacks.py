from dash import Input, Output
from data.data_processing import prepare_scatter_data
from data.experiment_loader import (get_cached_experiments,
                                    get_image_from_experiment)
from loguru import logger
from utils.plotting import build_scatter_figure
from utils.utils import encode_image


def register_callbacks(app):
    @app.callback(
        Output('scatter-plot',    'figure'),
        [
            Input('x-axis-dropdown', 'value'),
            Input('y-axis-dropdown', 'value'),
            Input('nu-filter', 'value'),
            Input('E-filter', 'value'),
            Input('basis-filter', 'value'),
            Input('mode-filter', 'value'),
            Input('yx-line-toggle',  'value')
        ],
    )
    def update_scatter_plot_cb(x_metric, y_metric, nu_filter, E_filter, basis_filter, mode_filter, plot_yx_line):

        filters = {
            'nu': nu_filter,
            'E_min': E_filter,
            'basis_v': basis_filter,
            'extremal_mode': mode_filter,
        }
        experiments = get_cached_experiments()
        logger.info(f"{len(experiments)} experiments in update_scatter_plot()")
        
        scatter_df = prepare_scatter_data(x_metric, y_metric, experiments, filters)
        
        fig = build_scatter_figure(scatter_df, x_metric, y_metric, False, )
        
        return fig

    @app.callback(
        Output('hover-image', 'src'),
        #  Output('hover-text', 'children')],
        Input('scatter-plot', 'hoverData')
    )
    def update_on_hover_cb(hover_data):
        if hover_data:
            run_id = int(hover_data['points'][0]['hovertext'].split(': ')[1])
            exp_img = get_image_from_experiment(run_id)
            img_src = f"data:image/png;base64,{encode_image(exp_img)}"
            # exp_txt = get_text_from_experiment(loader, run_id)
            return img_src#, ''
        logger.debug("No hover_data for datapoint")
        return ''#, 'Hover over a point to see details here.'

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
        if n_clicks > 0:
            return [], [], [], []
        return [], [], [] , []