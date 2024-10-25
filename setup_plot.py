""" Plotting utilities.
    author: Daniel Nichols
    date: May 2023
"""
# std imports
from typing import Iterable, Optional, Tuple, Union

# tpl imports
from cycler import cycler
import matplotlib as mpl
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import font_manager as fm 
from matplotlib.pyplot import gca
import matplotlib




def set_aspect_ratio(ratio=3/5, logx=None, logy=None, axis=None):
    if axis is None:
        axis = gca() 
    xleft, xright = axis.get_xlim()
    if logx is not None:
        xleft = math.log(xleft, logx)
        xright = math.log(xright, logx)
    ybottom, ytop = axis.get_ylim()
    if logy is not None:
        ytop = math.log(ytop, logy)
        print(ytop, ybottom)
        ybottom = math.log(ybottom, logy)
    axis.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)



def render(ax, output: Optional[str] = None):
    """ Render a plot """
    if output:
        plt.savefig(output)
    else:
        plt.show()


def plot_line_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: Optional[str] = None,
    title_fontsize: Optional[int] = None,
    xlabel: Optional[str] = None,
    xlabel_fontsize: Optional[int] = None,
    ylabel: Optional[str] = None,
    ylabel_fontsize: Optional[int] = None,
    legend_title: Optional[str] = None,
    logx: Optional[int] = None,
    figsize: tuple = (12, 8),
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    tight_layout: bool = False,
    **kwargs
):
    """ Plot a line chart """
    plt.figure(figsize=figsize)
    ax = sns.lineplot(data=df, x=x, y=y, **kwargs)
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel(xlabel, fontsize=xlabel_fontsize)
    ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)

    if logx:
        ax.set_xscale('log', base=logx)

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    ax.yaxis.grid(linestyle='dashed')
    ax.spines['left'].set_color('#606060')
    ax.spines['bottom'].set_color('#606060')
    global_cycler = cycler(color=get_colors()) + cycler(linestyle=get_linestyles())
    ax.set_prop_cycle(global_cycler)

    if legend_title:
        ax.get_legend().set_title(legend_title)

    if tight_layout:
        plt.tight_layout()

    return ax


def plot_bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: Optional[str] = None,
    title_fontsize: Optional[int] = None,
    xlabel: Optional[str] = None,
    xlabel_fontsize: Optional[int] = None,
    ylabel: Optional[str] = None,
    ylabel_fontsize: Optional[int] = None,
    legend_title: Optional[str] = None,
    xtick_rotation: int = 0,
    figsize: tuple = (5, 3),
    colors: Optional[Iterable[str]] = None,
    linestyles: Optional[Iterable[str]] = None,
    hatch: bool = True,
    hatches: Optional[Iterable[str]] = None,
    tight_layout: bool = True,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    logx: Optional[int] = None,
    logy: Optional[int] = None,
    labels: bool = False,
    label_fontsize: Optional[int] = None,
    label_fmt: str = '{:.2f}',
    label_xytext: Tuple[int, int] = (0, 10),
    error: Optional[str] = None,
    **kwargs
):
    """ Plot a bar chart """
    plt.figure(figsize=figsize)
    colors = colors or get_colors()
    ax = sns.barplot(data=df, x=x, y=y, palette=colors, **kwargs)
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel(xlabel, fontsize=xlabel_fontsize)
    ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=xtick_rotation)

    if legend_title is not None:
        if legend_title == '':
            legend_handles, legend_labels = ax.get_legend_handles_labels()
            ax.legend(handles=legend_handles[:], labels=legend_labels[:])
        else:    
            ax.get_legend().set_title(legend_title)

    if logx:
        ax.set_xscale('log', base=logx)
    if logy:
        ax.set_yscale('log', base=logy)

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    if error:
        ax.errorbar(
            x=df[x],
            y=df[y],
            yerr=df[error],
            fmt='none',
            color='#606060',
            capsize=5,
            elinewidth=2,
            capthick=2
        )

    if labels:
        for p in ax.patches:
            ax.annotate(
                label_fmt.format(p.get_height()),
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center',
                va='center',
                xytext=label_xytext,
                textcoords='offset points',
                fontsize=label_fontsize
            )

    ax.yaxis.grid(linestyle='dashed')
    ax.spines['left'].set_color('#606060')
    ax.spines['bottom'].set_color('#606060')
    linestyles = linestyles or get_linestyles()
    if len(colors) > len(linestyles):
        linestyles = linestyles * (len(colors) // len(linestyles) + 1)
        linestyles = linestyles[:len(colors)]
    global_cycler = cycler(color=colors) + cycler(linestyle=linestyles)
    ax.set_prop_cycle(global_cycler)

    if hatch:
        hatches = hatches or get_hatches()
        
        if 'hue' in kwargs:
            n_groups = len(df[kwargs['hue']].unique())
            group_size = len(df[x].unique())
        else:
            n_groups = len(df[x].unique())
            group_size = 1

      
        print (len(ax.patches))
        for i, bar in enumerate(ax.patches):
            bar.set_hatch(hatches[(i // group_size) % len(hatches)])
            bar.set_edgecolor('k')

        if 'hue' in kwargs:
            for i, p in enumerate(ax.get_legend().get_patches()):
                p.set_hatch(hatches[(i % n_groups) % len(hatches)])
                p.set_edgecolor('k')

    if tight_layout:
        plt.tight_layout()

    return ax



def plot_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: Optional[str] = None,
    label: bool = False,
    label_col: Optional[str] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    legend_title: Optional[str] = None,
    figsize: tuple = (12, 8),
    output: Optional[str] = None,
    **kwargs
):
    """ Plot a scatter plot """
    plt.figure(figsize=figsize)
    colors = get_darker_colors()
    markers = get_markers()

    if hue:
        # Get the number of unique values in the hue column
        n_hue = len(df[hue].unique())
        colors = colors[:n_hue]

    if kwargs.get('style'):
        style = kwargs['style']
        n_markers = len(df[style].unique())
        markers = markers[:n_markers]


    ax = sns.scatterplot(data=df, x=x, y=y, hue=hue, palette=colors, 
    markers = markers, **kwargs)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if legend_title:
        ax.get_legend().set_title(legend_title)

    if label and label_col:
        for line in range(0, df.shape[0]):
            ax.text(
                df[x][line] + 0.0001,
                df[y][line],
                df[label_col][line],
                horizontalalignment="left",
                size="small",
                color="black"
            )

    ax.yaxis.grid(linestyle='dashed')
    ax.spines['left'].set_color('#606060')
    ax.spines['bottom'].set_color('#606060')
    linestyles = get_linestyles()[:n_hue]
    global_cycler = cycler(color=colors) + cycler(linestyle=linestyles)
    ax.set_prop_cycle(global_cycler)

    plt.tight_layout()

    return ax


def plot_heatmap(
    df: pd.DataFrame,
    title: Optional[str] = None,
    title_fontsize: Optional[int] = None,
    xlabel: Optional[str] = None,
    xlabel_fontsize: Optional[int] = None,
    ylabel: Optional[str] = None,
    ylabel_fontsize: Optional[int] = None,
    figsize: tuple = (12, 8),
    tight_layout: bool = False,
    xtick_rotation: int = 0,
    ytick_rotation: int = 0,
    **kwargs
) -> mpl.axes.Axes:
    """ Plot a heatmap """
    plt.figure(figsize=figsize)
    ax = sns.heatmap(data=df, **kwargs)

    if title:
        ax.set_title(title, fontsize=title_fontsize)

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=xlabel_fontsize)
    
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=xtick_rotation)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=ytick_rotation)

    if tight_layout:
        plt.tight_layout()

    return ax


def plot_boxplot(
    df: pd.DataFrame,
    x: str,
    y: str,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    title_fontsize: Optional[int] = None,
    xlabel: Optional[str] = None,
    xlabel_fontsize: Optional[int] = None,
    ylabel: Optional[str] = None,
    ylabel_fontsize: Optional[int] = None,
    colors: Optional[Iterable[str]] = None,
    figsize: tuple = (12, 8),
    tight_layout: bool = False,
    xtick_rotation: int = 0,
    **kwargs
) -> mpl.axes.Axes:
    """ Plot a boxplot """
    plt.figure(figsize=figsize)
    colors = colors or get_colors()
    PROPS = {
        'boxprops':{'edgecolor':'black'},
        'medianprops':{'color':'black'},
        'whiskerprops':{'color':'black'},
        'capprops':{'color':'black'}
    }
    ax = sns.boxplot(data=df, x=x, y=y, palette=colors, **PROPS, **kwargs)
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel(xlabel, fontsize=xlabel_fontsize)
    ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=xtick_rotation)

    if xlim:
        ax.set_xlim(xlim)

    if ylim:
        ax.set_ylim(ylim)

    if tight_layout:
        plt.tight_layout()

    ax.yaxis.grid(linestyle='dashed')
    ax.spines['left'].set_color('#606060')
    ax.spines['bottom'].set_color('#606060')

    return ax


def plot_violin(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: Optional[str] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    title_fontsize: Optional[int] = None,
    xlabel: Optional[str] = None,
    xlabel_fontsize: Optional[int] = None,
    ylabel: Optional[str] = None,
    ylabel_fontsize: Optional[int] = None,
    colors: Optional[Iterable[str]] = None,
    figsize: tuple = (12, 8),
    tight_layout: bool = False,
    xtick_rotation: int = 0,
    **kwargs
) -> mpl.axes.Axes:
    """ Plot a boxplot """
    plt.figure(figsize=figsize)
    colors = colors or get_colors()
    ax = sns.violinplot(data=df, x=x, y=y, hue=hue, palette=colors, **kwargs)
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel(xlabel, fontsize=xlabel_fontsize)
    ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=xtick_rotation)

    if xlim:
        ax.set_xlim(xlim)

    if ylim:
        ax.set_ylim(ylim)

    if tight_layout:
        plt.tight_layout()

    ax.yaxis.grid(linestyle='dashed')
    ax.spines['left'].set_color('#606060')
    ax.spines['bottom'].set_color('#606060')

    return ax


def set_font(ttf_fpath, font_scale=1.0):
    from matplotlib.font_manager import FontProperties, fontManager
    fontManager.addfont(ttf_fpath)
    prop = FontProperties(fname=ttf_fpath)
    sns.set(font=prop.get_name(), font_scale=font_scale)

def setup():
    mpl.use('Agg')
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['lines.linewidth'] = 2
    mpl.rcParams['lines.markersize'] = 8
    mpl.rcParams['font.size'] = 18

    sns.set_style(rc={'axes.facecolor': "w", "grid.color": "#a9a9a9"})
    #sns.set_style(rc={'axes.facecolor': "w", "grid.color": "w"})

def setup_global():
    font_entry = fm.FontEntry(
        fname = './gillsans.ttf',
        name='gill-sans')

    # set font
    fm.fontManager.ttflist.insert(0, font_entry) 
    mpl.rcParams['font.family'] = font_entry.name 

    mpl.use('Agg')
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['lines.linewidth'] = 2
    mpl.rcParams['lines.markersize'] = 5
    mpl.rcParams['hatch.linewidth'] = 0.5




linestyle_tuple = [
     ('solid', 'solid'),      # Same as (0, ()) or '-'
     ('dotted', 'dotted'),    # Same as (0, (1, 1)) or '.'
     ('dashed', 'dashed'),    # Same as '--'
     ('dashdot', 'dashdot'),
     
     ('loosely dotted',        (0, (1, 10))),
     ('dotted',                (0, (1, 1))),
     ('densely dotted',        (0, (1, 1))),

     ('loosely dashed',        (0, (5, 10))),
     ('dashed',                (0, (5, 5))),
     ('densely dashed',        (0, (5, 1))),

     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))
    ]

linestyle_dict = {k : v for k,v in linestyle_tuple}

def get_colors():
    #return ['#D55E00', '#009E73','#0072B2', '#CC79A7', '#000000', '#E03A3D']
    #return ['#D55E00', '#009E73','#0072B2', '#CC79A7', '#643B9F', '#E03A3D']
    return ['#FD8A8A', '#A8D1D1', '#9EA1D4', '#FFCBCB', '#DFEBEB', '#F1F7B5']

def get_darker_colors():
    # Return darker tones of the colors in get_colors()
    return ['#E03A3D', '#0072B2', '#643B9F', '#D55E00', '#009E73', '#CC79A7']





def get_hatches():
    return ['x', 'xxx', '\\\\', '||','///', '+', 'o', '.', '*', '-', 'ooo', '+++']

def get_linestyles():
    return [
        linestyle_dict['solid'], 
        linestyle_dict['dotted'], 
        linestyle_dict['dashed'],
        linestyle_dict['dashdotted'],
        linestyle_dict['dashdotdotted'],
        linestyle_dict['solid'],
    ]

def get_markers():
    return ['o', '^', 's', 'o', 'd', 'x']
