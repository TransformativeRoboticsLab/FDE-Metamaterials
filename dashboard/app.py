import dash
from components.callbacks import register_callbacks
from components.layout import create_layout
from data.experiment_loader import start_experiment_loader_thread

# start async loader
start_experiment_loader_thread()

app = dash.Dash(__name__, external_stylesheets=["https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css"])
app.layout = create_layout()

# Register callbacks
register_callbacks(app)

if __name__ == "__main__":
    app.run_server(debug=True, host='localhost', port=8050)
