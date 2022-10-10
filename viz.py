from turtle import width
import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

from utils import *

COLORS = px.colors.qualitative.Alphabet
PERIOD = 360


def plot_events(i, ticks, theta_per_tick, r=60, symbol=None, size=20):
    r = [r] * ticks.size
    theta = ticks * theta_per_tick
    scatter = go.Scatterpolar(r=r, theta=theta, mode='markers', showlegend=False,
                            marker=dict(color=COLORS[i], symbol=symbol, size=20) )

    return scatter


def plot_task_events(i, task, H, for_deadlines=False):
    theta_per_tick = PERIOD / H
    symbol = 'star-diamond'
    r = 60
    
    if for_deadlines:
        symbol = 'circle'
        r = 50

    ticks = get_ticks(task, H, for_deadlines=for_deadlines)
    scatter = plot_events(i, ticks, theta_per_tick, r, symbol)
    return scatter


def plot_trace(_id, df_traces, H):
    df_trace = df_traces[ df_traces['id'] == str(_id + 1) ]
    trace = df_trace.start

    theta_per_tick = PERIOD / H
    width = theta_per_tick * df_trace['dt']
    r = 50 - 5 * (trace // H)
    theta = (trace + 0.5 * df_trace['dt']) * theta_per_tick

    bar = go.Barpolar(r=r, base=0, theta=theta, width=width, name=f'Task {_id + 1}',
                    marker=dict(color=COLORS[_id], line_width=1, line_color='black'), 
                    opacity=0.8)

    return bar


def plot_polar(tasks, df_traces, size=800):
    fig_polar = go.Figure()

    H = compute_H(tasks)
    ticks = np.arange(H)
    for i, task in enumerate(tasks):
        scatter_releases = plot_task_events(i, task, H)
        fig_polar.add_trace(scatter_releases)

        scatter_deadlines = plot_task_events(i, task, H, for_deadlines=True)
        fig_polar.add_trace(scatter_deadlines)

        bar = plot_trace(i, df_traces, H)
        fig_polar.add_trace(bar)    

    # Add cost trace
    bar = plot_trace(-1, df_traces, H)
    bar.showlegend = False
    fig_polar.add_trace(bar)

    fig_polar.update_layout(template=None, autosize=False, width=size, height=size,
                            polar=dict(angularaxis=dict(ticks='outside',
                                                # type='category', period=H,
                                                tickmode='array', tickvals=ticks * (360 / H), ticktext=ticks),
                                        radialaxis = dict(showticklabels=False, ticks='') ) )

    fig_polar.show()


def process_traces(traces):
    df_traces = pd.DataFrame(traces, columns=['id'])

    df_traces['dt'] = df_traces['id'].ne( df_traces['id'].shift() ).cumsum()
    df_traces = df_traces.groupby('dt', as_index=False).aggregate({'id': 'first', 'dt': 'count'})

    df_traces['end'] = df_traces['dt'].cumsum()
    df_traces['start'] = df_traces['end'] - df_traces['dt']

    df_traces['task'] = df_traces['id'].abs()
    df_traces = df_traces[df_traces['id'] != 0]
    df_traces.loc[ df_traces['id'] < 0, 'id'] = 0
    df_traces[['id']] = df_traces[['id']].astype(str)

    return df_traces


def plot_gantt(tasks, df_traces, fig_size=None):
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


