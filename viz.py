import time
import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from utils import *

COLORS = px.colors.qualitative.Plotly + ['#DDDDDD']
PERIOD = 360
  

def plot_trace(df_trace, H):
    _id = int(df_trace['id'].values[0])
    if _id == 0:
        color = 'white'
    elif _id < 0:
        color = COLORS[-1]
    else:
        color = COLORS[_id]

    trace = df_trace.start

    theta_per_tick = PERIOD / H
    width = theta_per_tick * df_trace['dt']
    r = 100 - 20 * (trace // H)
    theta = (trace + 0.5 * df_trace['dt']) % H * theta_per_tick

    bar = go.Barpolar(r=r, base=0, theta=theta, width=width,
                    marker=dict(color=color, line_width=2, line_color='black'), 
                    showlegend=False, opacity=1.0)

    return bar


import copy
def plot_polar(tasks, df_traces, size=800, output_dir=None):
    H = compute_H(tasks)
    ticks = np.arange(H)
    theta_per_tick = PERIOD / H

    fig_polar = go.Figure()

    def plot_events_deadlines(i, task):
        ticks = get_ticks(task, H, for_deadlines=True)
        r = [100] * ticks.size
        theta = ticks * theta_per_tick
        scatter = go.Scatterpolar(r=r, theta=theta, mode='markers', showlegend=True, name=f'Task {i + 1}',
                                marker=dict(color=COLORS[i], symbol='circle', size=(size / 50)) )
        return scatter

    def plot_events_releases(i, task):
        arrow_len = size / 2.25
        ticks = get_ticks(task, H)

        theta = ticks * theta_per_tick * (2 * np.pi / 360.0)

        print(ticks, H, theta_per_tick)

        axs = arrow_len * np.cos(theta)
        ays = arrow_len * np.sin(theta)

        print(axs, ays)

        for ax, ay in zip(axs, ays):
            fig_polar.add_annotation(x=0.5, y=0.5, ax=ax, ay=-ay,
                                    arrowside='start', arrowcolor=COLORS[i],
                                    arrowhead=2, arrowwidth=2)

    for i, task in enumerate(tasks):
        plot_events_releases(i, task)

        scatter_deadlines = plot_events_deadlines(i, task)
        fig_polar.add_trace(scatter_deadlines)

    fig_polar.update_layout(template=None, autosize=False, width=size, height=size,
                            polar=dict(angularaxis=dict(ticks='', ticktext=ticks,
                                                        # type='category', period=H,
                                                        tickmode='array', tickvals=ticks * (360 / H)),
                                        radialaxis = dict(range=[0, 100], showticklabels=False, ticks='') ) )

    # for t in range(df_traces.shape[0]):
    #     df_trace = df_traces.iloc[t:t+1]
        
    #     bar = plot_trace(df_trace, H)
    #     fig_polar.add_trace(bar)

    #     if output_dir is not None:
    #         pio.write_image(fig_polar, f'{output_dir}/{t:03d}.png', 
    #                         width=size, height=size, scale=1)

    # fig_polar.add_shape(type="path",
    #                     path="M 0.8,1 Q 1.0,1.0 0.9,0.9",
    #                     line_color="RoyalBlue",)

    fig_polar.show()


def process_traces(traces):
    df_traces = pd.DataFrame(traces, columns=['id'])

    df_traces['dt'] = df_traces['id'].ne( df_traces['id'].shift() ).cumsum()
    df_traces = df_traces.groupby('dt', as_index=False).aggregate({'id': 'first', 'dt': 'count'})

    df_traces['end'] = df_traces['dt'].cumsum()
    df_traces['start'] = df_traces['end'] - df_traces['dt']

    df_traces['task'] = df_traces['id'].abs()
    return df_traces


def plot_gantt(tasks, df_traces, fig_size=None):
    df_traces = df_traces.loc[df_traces['id'] != 0].copy()
    df_traces.loc[ df_traces['id'] < 0, 'id'] = 0
    df_traces.loc[:, ('id')] = df_traces['id'].astype(str)

    fig_gantt = px.timeline(df_traces, x_start='start', x_end='end', y='task', 
                        color='id', color_discrete_sequence=COLORS, opacity=0.8)
    fig_gantt.update_traces(marker=dict(line_color='black', line_width=1.0))
        
    y_offset = -0.25
    n = len(tasks)
    def plot_events_gantt(i, ticks, for_deadlines=False):
        y = n - (i + 1) + y_offset
        for t in ticks:
            if for_deadlines:
                fig_gantt.add_shape(type='circle', x0=t, y0=(y - 0.05), x1=(t + 0.25), y1=(y + 0.05), 
                                    xref='x', yref='y', 
                                    line_color=COLORS[i], fillcolor=COLORS[i])
            else:
                fig_gantt.add_annotation(x=t, y=(y + 0.75), ax=t, ay=(y - 0.1), 
                                        axref='x', ayref='y',
                                        showarrow=True, arrowhead=2, arrowwidth=2, arrowcolor=COLORS[i])

    def plot_task_events_gantt(i, task, H, for_deadlines=False, size=None):
        if size is None:
            size = H
        n_repeats = (size // H) + 1

        ticks = get_ticks(task, H, for_deadlines=for_deadlines, n_repeats=n_repeats)
        ticks = ticks[ ticks <= size]
        plot_events_gantt(i, ticks, for_deadlines)

    for data in fig_gantt.data:
        data.width = 0.5
        filtered = df_traces['id'] == data.name

        data.base = df_traces[filtered]['start'].tolist()
        data.x = df_traces[filtered]['dt'].tolist()
        data.y = df_traces[filtered]['task'].tolist()

        i = int(data.name) - 1
        data.marker.color = COLORS[i]

        if i == -1:
            data.showlegend = False
            continue

        # Add datum lines
        fig_gantt.add_hline(y=(i + y_offset) )
        
        # Add events
        H = compute_H(tasks)
        size = df_traces['dt'].sum()
        plot_task_events_gantt(i, tasks[i], H, size=size)
        plot_task_events_gantt(i, tasks[i], H, for_deadlines=True, size=size)

    fig_gantt.update_layout(width=fig_size, height=fig_size)
    fig_gantt.update_xaxes(type='linear')
    fig_gantt.update_yaxes(type='category', categoryorder='category descending')
    fig_gantt.show()


