import numpy as np

import plotly.graph_objs as go
import plotly.io as pio


def donut_chart(data, pred):
    labels = ['Gender', 'Education Level', 'Socio-Economic stats', 'Age', 'Num. Prev attempts', 'Credits',
              'Disability', 'Productivity']
    colors = ['gold', 'mediumturquoise', 'darkorange', 'green',
              'gold', 'mediumturquoise', 'darkorange', 'red']
    og_values = data[0]
    values = np.abs(data[0])

    # Use `hole` to create a donut-like pie chart
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.5)])
    fig.update_traces(hoverinfo='label+percent', textinfo='label', textfont_size=15,
                      marker=dict(colors=list(map(setcolor, og_values)), line=dict(color='#000000', width=2)),)
    fig.update_layout(title=f"Model's (reason) for predicting student outcome as: '{pred}'\n",
                      template='plotly_dark', width=800, height=1000,
                      margin_b=200,
                      annotations=[dict(xref='paper',
                                        yref='paper',
                                        x=0.5, y=-0.25,
                                        showarrow=False,
                                        text="Green represents feature(s) that contributed positively (as a %) <br> "
                                             "Red represents feature(s) that contributed negatively (as a %)",
                                        font=dict(color='white', size=18)),
                                   ],
                      )
    return fig


def waterfall_chart(data):
    fig = go.Figure(go.Waterfall(
        name="20", orientation="v",
        measure=["relative", "relative", "total", "relative", "relative", "total"],
        x=['Gender', 'Education Level', 'Socio-Economic stats', 'Age', 'Num. Prev attempts', 'Credits', 'Disability',
           'Productivity'],
        textposition="outside",
        text=[str(v) for v in data] + ["Total"],
        y=data,
        decreasing={"marker": {"color": "#fa0000"}},
        increasing={"marker": {"color": "#fa00f6"}},
        connector={"line": {"color": "#ffa8fe"}},
    ))
    
    fig.update_traces(textfont_size=30,)

    fig.update_layout(
        title="Feature Importance (absolute value of probability distribution for the classification)",
        showlegend=True,
        margin_b=200,
        annotations=[dict(xref='paper',
                          yref='paper',
                          x=0.5, y=-0.25,
                          showarrow=False,
                          text='This is my caption for the Plotly figure',
                          font=dict(color='white', size=25)),
                     ],
        template='plotly_dark', width=800, height=1000,)
    return fig


def setcolor(value):
    if value <= 0:
        return "red"
    elif value > 0:
        return "lightgreen"
