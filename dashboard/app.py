import dash
from components.callbacks import register_callbacks
from components.layout import create_layout
from data.experiment_loader import (init_experiments_load,
                                    start_experiment_loader_thread)
from loguru import logger


def main():
    logger.success("Starting main app")
    try:
        app = dash.Dash(__name__, external_stylesheets=["https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css"])
        logger.success("Dashboard started")
    except Exception as e:
        logger.exception(f"Exception creating dash app: {e}")


    # start with synchronous load
    init_experiments_load()
    start_experiment_loader_thread()

    app.layout = create_layout()
    register_callbacks(app)

    app.run_server(debug=True, host='localhost', port=8050)

if __name__ == "__main__":
    main()
