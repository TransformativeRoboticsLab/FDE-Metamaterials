from dash import Input, Output
from data.data_processing import prepare_scatter_data
from data.experiment_loader import (get_cached_experiments,
                                    get_image_from_experiment)
from utils.plotting import build_scatter_figure, update_dropdown_options
from utils.utils import encode_image


def register_callbacks(app):
    @app.callback(
        [Output('scatter-plot',    'figure'),
        Output('x-axis-dropdown', 'options'),
        Output('y-axis-dropdown', 'options'),
        Output('nu-filter', 'options'),
        Output('E-filter', 'options')
        ],
        Input('x-axis-dropdown', 'value'),
        Input('y-axis-dropdown', 'value'),
        Input('nu-filter', 'value'),
        Input('E-filter', 'value'),
        Input('yx-line-toggle',  'value'),
    )
    def update_scatter_plot(x_metric, y_metric, nu_filter, E_filter, plot_yx_line):

        experiments = get_cached_experiments()
        print(f"{len(experiments)} experiments in update_scatter_plot()")
        
        scatter_df = prepare_scatter_data(x_metric, y_metric, experiments, nu_filter, E_filter)
        
        fig = build_scatter_figure(scatter_df, x_metric, y_metric, False, )
        
        metric_dropdowns, nu_dropdowns, E_dropdowns = update_dropdown_options(experiments)
        
        return fig, metric_dropdowns, metric_dropdowns, nu_dropdowns, E_dropdowns

    @app.callback(
        Output('hover-image', 'src'),
        #  Output('hover-text', 'children')],
        Input('scatter-plot', 'hoverData')
    )
    def update_on_hover(hover_data):
        if hover_data:
            run_id = int(hover_data['points'][0]['hovertext'].split(': ')[1])
            exp_img = get_image_from_experiment(run_id)
            img_src = f"data:image/png;base64,{encode_image(exp_img)}"
            # exp_txt = get_text_from_experiment(loader, run_id)
            return img_src#, ''
        return ''#, 'Hover over a point to see details here.'