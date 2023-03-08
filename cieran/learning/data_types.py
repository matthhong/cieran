"""
Modules for queries and user responses.

:TODO: OrdinalQuery classes will be implemented so that the library will include ordinal data, which was used for reward learning in:
    K. Li, M. Tucker, E. Biyik, E. Novoseller, J. W. Burdick, Y. Sui, D. Sadigh, Y. Yue, A. D. Ames;
    "ROIAL: Region of Interest Active Learning for Characterizing Exoskeleton Gait Preference Landscapes", ICRA'21.
"""
from typing import List, Union
from copy import deepcopy, copy
import itertools
import numpy as np
import time

from cieran.basics import Trajectory, TrajectorySet
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from geomdl import fitting

from coloraide import Color
from IPython.display import display, clear_output
import ipywidgets as widgets
    
t = np.linspace(0, 2 * np.pi, 1024)
data2d = np.sin(t)[:, np.newaxis] * np.cos(t)[np.newaxis, :]

def default_chart(cmap):
   
    fig, ax = plt.subplots()
    ax.imshow(data2d, cmap=cmap)

    plt.show()

def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


class Query:
    """
    An abstract parent class that is useful for typing.
    
    A query is a question to the user.
    """
    def __init__(self):
        pass
        
    def copy(self):
        """Returns a deep copy of the query."""
        # return deepcopy(self)
        return copy(self)
        
    def visualize(self, delay: float = 0.):
        """Visualizes the query, i.e., asks it to the user.
        
        Args:
            delay (float): The waiting time between each trajectory visualization in seconds.
        """
        raise NotImplementedError


class QueryWithResponse:
    """
    An abstract parent class that is useful for typing.
    
    An instance of this class holds both the query and the user's response to that query.
    
    Parameters:
        query (Query): The query.
        
    Attributes:
        query (Query): The query.
    """
    def __init__(self, query: Query):
        self.query = query



class WeakComparisonQuery(Query):
    """
    A weak comparison query is one where the user is presented with two trajectories and asked for their favorite among
    them, but also given an option to say 'they are about equal'.
    
    Parameters:
        slate (TrajectorySet or List[Trajectory]): The set of trajectories that will be presented to the user.

    Attributes:
        K (int): The number of trajectories in the query. It is always equal to 2 and kept for consistency with 
            :class:`.PreferenceQuery` and :class:`.FullRankingQuery`.
        response_set (numpy.array): The set of possible responses to the query, which is always equal to [-1, 0, 1]
            where -1 represents the `About Equal` option.
        
    Raises:
        AssertionError: if slate does not have exactly 2 trajectories.
    """
    def __init__(self, slate: Union[TrajectorySet, List[Trajectory]], chart=None):
        super().__init__()
        assert isinstance(slate, TrajectorySet) or isinstance(slate, list), 'Query constructor requires a TrajectorySet object for the slate.'
        self.slate = slate
        self.chart = chart
        assert(self.K == 2), 'Weak comparison queries can only be pairwise comparisons, but ' + str(self.K) + ' trajectories were given.'
    
    @property
    def slate(self) -> TrajectorySet:
        """Returns a :class:`.TrajectorySet` of the trajectories in the query."""
        return self._slate
    
    @slate.setter
    def slate(self, new_slate: Union[TrajectorySet, List[Trajectory]]):
        """Sets the slate of trajectories in the query."""
        self._slate = new_slate if isinstance(new_slate, TrajectorySet) else TrajectorySet(new_slate)
        self.K = self._slate.size
        self.response_set = np.array([-1,0,1])

    def visualize(self, delay: float = 0.) -> int:
        """Visualizes the query and interactively asks for a response.
        
        Args:
            delay (float): The waiting time between each trajectory visualization in seconds.
            
        Returns:
            int: The response of the user.
        """
        ramp1 = self.slate[0].ramp
        ramp2 = self.slate[1].ramp

        # if is_notebook():
        if self.chart is None:
            self.chart = default_chart

        out = widgets.Output()
        # out = widgets.Output(layout={'border': '1px solid black'})

        plot_widget1 = widgets.Output()
        plot_widget2 = widgets.Output()

        with plot_widget1:
            # self.slate[0].plot_all()
            self.chart(ramp1)

        with plot_widget2:
            # self.slate[1].plot_all()
            self.chart(ramp2)

        box_layout = widgets.Layout(display='flex',
                        # flex_flow='column',

                        flex_flow='row',
                        justify_content='space-around')
        plots_hbox = widgets.VBox([plot_widget1, plot_widget2], layout=box_layout)

        display(out)
        with out:
            display(plots_hbox)
        
        # else:
        #     fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        #     im1=ax[0].imshow(data2d, cmap=ramp1)
        #     im2=ax[1].imshow(data2d, cmap=ramp2)

        #     plt.show()

        selection = None
        while selection is None:
            selection = input('Which is better for a paper figure? Enter a number (1 left, 2 right, 0 about equal): ')
            selection = str(int(selection) - 1)
            if not isinteger(selection) or int(selection) not in self.response_set:
                selection = None

        clear_output()

        return int(selection)



class WeakComparison(QueryWithResponse):
    """
    A Weak Comparison feedback.
    
    Contains the :class:`.WeakComparisonQuery` the user responded to and the response.
    
    Parameters:
        query (WeakComparisonQuery): The query for which the feedback was given.
        response (int): The response of the user to the query.
        
    Attributes:
        response (int): The response of the user to the query.
        
    Raises:
        AssertionError: if the response is not in the response set of the query.
    """
    def __init__(self, query: WeakComparisonQuery, response: int):
        super().__init__(query)
        assert(response in self.query.response_set), 'Invalid response ' + str(response) +  ' for the weak comparison query.'
        self.response = response


def isinteger(input: str) -> bool:
    """Returns whether input is an integer.
    
    :Note: This function returns False if input is a string of a float, e.g., '3.0'.
    :TODO: Should this go to utils?
    
    Args:
        input (str): The string to be checked for being an integer.
    
    Returns:
        bool: True if the :py:attr:`input` is an integer, False otherwise.
    
    Raises:
        AssertionError: if the input is not a string.
    """
    assert(isinstance(input, str)), 'Invalid input to the isinteger method. The input must be a string.'
    try:
        a = int(input)
        return True
    except:
        return False