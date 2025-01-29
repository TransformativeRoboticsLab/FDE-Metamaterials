import threading
import time

import dash
import plotly.express as px
import plotly.graph_objects as go
from components.layout import create_layout
from dash.dependencies import Input, Output
from incense import ExperimentLoader
from utils import *

# reference later
# https://medium.com/plotly/how-to-create-a-beautiful-interactive-dashboard-layout-in-python-with-plotly-dash-a45c57bb2f3c#:~:text=We%E2%80%99ll%20look%20at%20how%20to%20develop%20a%20dashboard%20grid%20and





app = dash.Dash(__name__)
app.layout = create_layout()




def main():
    load_thread = threading.Thread(target=load_experiments_async, args=(loader, 'extremal', filter_tags))
    load_thread.daemon = True
    load_thread.start()
    app.run_server(debug=True, host='localhost', port=8050)

if __name__ == '__main__':
    main()