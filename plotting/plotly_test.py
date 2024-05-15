# import plotly.express as px
# fig = px.bar(x=["a", "b", "c"], y=[1, 3, 2])
# fig.write_html('first_figure.html', auto_open=True)

import plotly.express as px
import plotly.graph_objects as go

import pandas as pd

# df = px.data.gapminder().query("country=='Brazil'")
df = px.data.gapminder().query("continent=='Oceania'")
fig = px.line_3d(df, x="gdpPercap", y="pop", z="year", color="country")
fig.show()

# Read data from a csv
z_data = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/api_docs/mt_bruno_elevation.csv')

fig2 = go.Figure(data=[go.Surface(z=z_data.values)])

fig2.update_layout(title='Mt Bruno Elevation', autosize=False,
                  width=500, height=500,
                  margin=dict(l=65, r=50, b=65, t=90))

fig2.show()