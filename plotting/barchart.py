from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def generate_mesh(x_min, x_max, y_min, y_max, z_min, z_max, color_value, flat_shading, hover_info, opacity: float = 1):
    return go.Mesh3d(
        x=[
            x_min, x_min, x_max, x_max,
            x_min, x_min, x_max, x_max,
        ],
        y=[
            y_min, y_max, y_max, y_min,
            y_min, y_max, y_max, y_min,
        ],
        z=[
            z_min, z_min, z_min, z_min,
            z_max, z_max, z_max, z_max,
        ],
        color=color_value,
        i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
        j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
        k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
        opacity=opacity,
        flatshading=flat_shading,
        hovertext='text',
        hoverinfo=hover_info,
    )


def create_z_grid(len_x_df_uniq, len_y_df_uniq, z_df):
    z_temp_df = []

    for x, y in itertools.product(range(len_x_df_uniq), range(len_y_df_uniq)):
        if x == y:
            z_temp_df.append(z_df[x])
        else:
            z_temp_df.append(None)
    return z_temp_df


def figure_layout(
        fig: go.Figure,
        x_legend: str, y_legend, x_min, len_x_df_uniq, x_title, y_title, len_y_df_uniq,
        z_legend, z_title, title,
):
    y_min = 0

    fig.update_layout(
        scene=dict(
            xaxis=dict(
                tickmode='array',
                ticktext=x_legend,
                tickvals=np.arange(x_min, len_x_df_uniq * 2, step=2),
                title=x_title,
            ),
            yaxis=dict(
                tickmode='array',
                ticktext=y_legend,
                tickvals=np.arange(y_min, len_y_df_uniq * 2, step=2),
                title=y_title,
            ),
            zaxis=dict(title=z_title),
        ),
    )
    if z_legend is None:
        fig.update_layout(
            scene=dict(
                zaxis=dict(
                    tickmode='array',
                    ticktext=z_legend,
                    title=z_title,
                ),
            ),
            template='plotly_white',
        )
    fig.update_layout(title=title)

    return fig


def bar_charts_from_sparse_array(
        x_df, y_df, z_df, x_min=0, y_min=0, z_min='auto', step=1, color='x',
        x_legend='auto', y_legend='auto', z_legend='auto', flat_shading=True,
        x_title='', y_title='', z_title='', hover_info='z', title='',
) -> go.Figure:
    """
    Convert a dataframe in 3D barchart similar to matplotlib ones
    Example :
        xdf = pd.Series([1, 10])
        ydf = pd.Series([2, 4])
        zdf = pd.Series([10, 30, 20, 45])
        fig = plotly_bar_charts_3d(xdf, ydf, zdf, color='x+y')
        fig.show()
    :param x_df: Serie or list of data corresponding to x-axis
    :param y_df: Serie or list  of data corresponding to y-axis
    :param z_df: Serie or list  of data corresponding to height of the bar chart
    :param x_min: Starting position for x-axis
    :param y_min: Starting position for y-axis
    :param z_min: Minimum value of the barchart, if set to auto minimum value is 0.8 * minimum of z_df to obtain more
    packed charts
    :param step: Distance between two bar charts
    :param color: Axis to create color, possible parameters are
    x for a different color for each change of x
    y for a different color for each change of y
    or x+y to get a different color for each bar
    :param x_legend: Legend of x-axis, if set to auto the legend is based on x_df
    :param y_legend: Legend of y-axis, if set to auto the legend is based on y_df
    :param z_legend: Legend of z axis, if set to auto the legend is not shown
    :param flat_shading:
    :param x_title: Title of x-axis
    :param y_title: Title of y-axis
    :param z_title: Title of z axis
    :param hover_info: Hover info, z by default
    :param title: Title of the graph, not functional for the moment
    :return: 3D mesh figure acting as 3D bar charts
    """
    z_df = list(pd.Series(z_df))

    if z_min == 'auto':
        z_min = 0.8 * min(z_df)

    mesh_list = []
    colors = px.colors.qualitative.Plotly
    color_value = 0

    len_x_df_uniq = len(x_df)
    len_y_df_uniq = len(y_df)

    z_temp_df = create_z_grid(len_x_df_uniq, len_y_df_uniq, z_df)

    for idx, x_data in enumerate(x_df):
        if color == 'x':
            color_value = colors[idx % 9]

        for idx2, y_data in enumerate(y_df):
            if color == 'x+y':
                color_value = colors[(idx + idx2 * len_y_df_uniq) % 9]

            elif color == 'y':
                color_value = colors[idx2 % 9]

            x_max = x_min + step
            y_max = y_min + step

            z_max = z_temp_df[idx * len_x_df_uniq + idx2]

            if z_max is not None:
                mesh_list.append(
                    generate_mesh(
                        x_min, x_max, y_min, y_max, z_min, z_max, color_value,
                        flat_shading, hover_info,
                    ),
                )
            else:
                mesh_list.append(
                    generate_mesh(
                        x_min, x_max, y_min, y_max, z_min, z_max, color_value,
                        flat_shading, hover_info, opacity=0.01,
                    ),
                )
            x_min += 2 * step
        y_min += 2 * step
        x_min = 0
    fig = go.Figure(mesh_list)

    if x_legend == 'auto':
        x_legend = x_df
        x_legend = [str(x_ax) for x_ax in x_legend]
    if y_legend == 'auto':
        y_legend = y_df
        y_legend = [str(y_ax) for y_ax in y_legend]
    if z_legend == 'auto':
        z_legend = None

    fig = figure_layout(
        fig, x_legend, y_legend, x_min, len_x_df_uniq, x_title, y_title, len_y_df_uniq,
        z_legend, z_title, title,
    )

    return fig


def bar_charts3d_from_array(
        x_df, y_df, z_df, x_min=0, y_min=0, z_min='auto', step=1, color='x',
        x_legend='auto', y_legend='auto', z_legend='auto', flat_shading=True,
        x_title='', y_title='', z_title='', hover_info='z', title='',
) -> go.Figure:
    """
    Convert a dataframe in 3D bar charts similar to matplotlib ones
        Example :
        x_df = pd.Series([1, 1, 10, 10])
        y_df = pd.Series([2, 4, 2 ,4])
        z_df = pd.Series([10, 30, 20, 45])
    :param x_df: Serie of data corresponding to x-axis
    :param y_df: Serie of data corresponding to y-axis
    :param z_df: Serie of data corresponding to height of the bar chart
    :param x_min: Starting position for x-axis
    :param y_min: Starting position for y-axis
    :param z_min: Minimum value of the barchart, if set to auto minimum value is 0.8 * minimum of z_df to obtain more
    packed charts
    :param step: Distance between two bar charts
    :param color: Axis to create color, possible parameters are
    x for a different color for each change of x
    y for a different color for each change of y
    or x+y to get a different color for each bar
    :param x_legend: Legend of x-axis, if set to auto the legend is based on x_df
    :param y_legend: Legend of y-axis, if set to auto the legend is based on y_df
    :param z_legend: Legend of z axis, if set to auto the legend is not shown
    :param flat_shading:
    :param x_title: Title of x-axis
    :param y_title: Title of y-axis
    :param z_title: Title of z axis
    :param hover_info: Hover info, z by default
    :param title: Title of the graph, not functional for the moment
    :return: 3D mesh figure acting as 3D bar charts
    """

    if z_min == 'auto':
        z_min = 0.8 * min(z_df)
    mesh_list = []
    colors = px.colors.qualitative.Plotly
    color_value = 0

    x_df = pd.Series(x_df)
    y_df = pd.Series(y_df)
    z_df = pd.Series(z_df)

    x_df_uniq = x_df.unique()
    y_df_uniq = y_df.unique()
    len_x_df_uniq = len(x_df_uniq)
    len_y_df_uniq = len(y_df_uniq)

    for idx, x_data in enumerate(x_df_uniq):
        if color == 'x':
            color_value = colors[idx % 9]
        for idx2, y_data in enumerate(y_df_uniq):
            if color == 'x+y':
                color_value = colors[(idx + idx2 * len(y_df.unique())) % 9]
            elif color == 'y':
                color_value = colors[idx2 % 9]
            x_max = x_min + step
            y_max = y_min + step
            z_max = z_df[idx + idx2 * len_x_df_uniq]
            mesh_list.append(
                generate_mesh(
                    x_min, x_max, y_min, y_max, z_min, z_max, color_value, flat_shading,
                    hover_info,
                ),
            )
            x_min += 2 * step
        y_min += 2 * step
        x_min = 0

    if x_legend == 'auto':
        x_legend = x_df_uniq
        x_legend = [str(x_ax) for x_ax in x_legend]
    if y_legend == 'auto':
        y_legend = y_df_uniq
        y_legend = [str(y_ax) for y_ax in y_legend]
    if z_legend == 'auto':
        z_legend = None

    fig = go.Figure(mesh_list)

    fig = figure_layout(
        fig, x_legend, y_legend, x_min, len_x_df_uniq, x_title, y_title, len_y_df_uniq,
        z_legend, z_title, title,
    )

    return fig


def verify_input(x, y, z) -> bool:
    """"
    Verify that input is valid
    """
    if len(x) * len(y) == len(z):
        return True
    if len(x) == len(y) == len(z):
        return True
    raise (
        ValueError(
            f'Input arguments are not matching, received x:{len(x)}, y:{len(y)}, z:{len(z)}, expected x*y=z '
            f'or x=y=z',
        )
    )


def convert_to_str(x: list[str]):
    """"
    Convert a list to a string
    """
    return [str(x) for x in x]


def plotly_bar_charts_3d(
        x_df, y_df, z_df, x_min=0, y_min=0, z_min='auto', step=1, color='x',
        x_legend='auto', y_legend='auto', z_legend='auto', flat_shading=True,
        x_title='', y_title='', z_title='', hover_info='z', title='',
):
    """
    Generate a barchart in 3D or a sparse barchart in 3D
    Examples :
        xdf = pd.Series([1, 10])
        ydf = pd.Series([2, 4])
        zdf = pd.Series([10, 30, 20, 45])
        fig = plotly_bar_charts_3d(xdf, ydf, zdf, color='x+y')
        fig.show()
        features = [2, 3, 5, 10, 20]
        neighbours = [31, 24, 10, 28, 48]
        accuracies = [0.9727, 0.9994, 0.9994, 0.9995, 0.9995]
        plotly_bar_charts_3d(
            features, neighbours, accuracies,
            x_title='Features', y_title='Neighbours', z_title='Accuracy',
        ).show()
    """
    verify_input(x_df, y_df, z_df)
    return (
        bar_charts3d_from_array(
            x_df,
            y_df,
            z_df,
            x_min=x_min,
            y_min=y_min,
            z_min=z_min,
            step=step,
            color=color,
            x_legend=x_legend,
            y_legend=y_legend,
            z_legend=z_legend,
            flat_shading=flat_shading,
            x_title=x_title,
            y_title=y_title,
            z_title=z_title,
            hover_info=hover_info,
            title=title,
        )
        if len(z_df) == (len(x_df) * len(y_df)) or (len(set(x_df)) + len(set(y_df))) == len(set(z_df))
        else bar_charts_from_sparse_array(
            x_df,
            y_df,
            z_df,
            x_min=x_min,
            y_min=y_min,
            z_min=z_min,
            step=step,
            color=color,
            x_legend=x_legend,
            y_legend=y_legend,
            z_legend=z_legend,
            flat_shading=flat_shading,
            x_title=x_title,
            y_title=y_title,
            z_title=z_title,
            hover_info=hover_info,
            title=title,
        )
    )


if __name__ == '__main__':
    xdf = pd.Series([1, 10])
    ydf = pd.Series([2, 4])
    zdf = pd.Series([10, 30, 20, 45])
    fig = plotly_bar_charts_3d(xdf, ydf, zdf, color='x+y')
    fig.show()
    features = [2, 3, 5, 10, 20]
    neighbours = [31, 24, 10, 28, 48]
    accuracies = [0.9727, 0.9994, 0.9994, 0.9995, 0.9995]
    plotly_bar_charts_3d(
        features, neighbours, accuracies,
        x_title='Features', y_title='Neighbours', z_title='Accuracy',
    ).show()
    df = pd.read_csv('examples/dataExample.csv')
    fig = plotly_bar_charts_3d(
        df['Gamma'], df['C'], df['score 1'], x_title='Gamma', y_title='C',
        color='y',
    )
    fig.show()
