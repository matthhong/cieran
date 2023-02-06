"""
Modules for queries and user responses.

:TODO: OrdinalQuery classes will be implemented so that the library will include ordinal data, which was used for reward learning in:
    K. Li, M. Tucker, E. Biyik, E. Novoseller, J. W. Burdick, Y. Sui, D. Sadigh, Y. Yue, A. D. Ames;
    "ROIAL: Region of Interest Active Learning for Characterizing Exoskeleton Gait Preference Landscapes", ICRA'21.
"""
from typing import List, Union
from copy import deepcopy
import itertools
import numpy as np
import time

from aprel.basics import Trajectory, TrajectorySet
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from geomdl import fitting

from coloraide import Color
    

t = np.linspace(0, 2 * np.pi, 1024)
data2d = np.sin(t)[:, np.newaxis] * np.cos(t)[np.newaxis, :]


class Ramping:

    def __init__(self, control_points, truncate_front=0, truncate_back=0):
        self.truncate_front = truncate_front
        self.truncate_back = truncate_back

        self.control_points = control_points
        self.ramp = None
        self.path = []
        self.cmap = None

    def generate_control_points(self):
        pass

    def interpolate(self, mode='cubic'):
        # interpolate between points using cubic splines with centripetal parameterization
        # save the ramp
        self.generate_control_points()

        if mode == 'cubic':
            # Do global curve interpolation
            self.ramp = fitting.interpolate_curve(self.control_points, 3, centripetal=True)
            
    # def truncate(self):
    #     # truncate the ramp at the start and end given truncate_front and truncate_back (in percent)
        
    #     # truncate the front
    #     # breakpoint()
    #     self.ramp.knotvector = self.ramp.knotvector[round(self.truncate_front * len(self.ramp.knotvector)):len(self.ramp.knotvector) - round(self.truncate_back * len(self.ramp.knotvector))]

    def lab_to_rgb(self, lab):
        # Convert a CIELAB value to an RGB value
        return Color("lab({}% {} {} / 1)".format(*lab)).convert("srgb")

    def execute(self):
        # execute the ramp
        self.interpolate()
        # self.truncate()
    
        # Get points from the interpolated geomdl ramp
        # However, the distance between points is not constant,
        # so we need to normalize the distance between points using arc length
        # We want to parameterize by t', which measures normalized arclength.

        # Get the points from the ramp
        t = np.linspace(0, 1, 1000)
        at = np.linspace(0, 1, 1000)
        points = self.ramp.evaluate_list(at)

        # Get the arc length of the ramp at each point using distance function
        arc_lengths = [0]
        for i in range(1, len(points)):
            arc_lengths.append(arc_lengths[i-1] + self.distance(points[i-1], points[i]))

        # Normalize the arc lengths
        arc_lengths = np.array(arc_lengths) / arc_lengths[-1]

        # Invert the arc lengths to get the parameterization
        at_t = np.interp(at, arc_lengths, t)

        # Get the points from the ramp using the parameterization
        self.path = self.ramp.evaluate_list(at_t)

        # Truncate the front and back of the path based on the first element of each point (e.g. cut if L* is less than 0.2)
        # self.path = [point for point in self.path if point[0] > self.truncate_front and point[0] < self.truncate_back]


        # # Truncate the front and back of the path
        # self.path = self.path[round(self.truncate_front * len(self.path)):len(self.path) - round(self.truncate_back * len(self.path))]
        #         self.path = [self.lab_to_rgb(p).to_string(hex=True) for p in self.path]
        self.path = [self.lab_to_rgb(p).to_string(hex=True) for p in self.path]

        # convert to ListedColormap
        self.cmap = ListedColormap(self.path)

    def distance(self, p1, p2):
        return Color("lab({}% {} {} / 1)".format(*p1)).delta_e(Color("lab({}% {} {} / 1)".format(*p2)), method='2000')


class Query:
    """
    An abstract parent class that is useful for typing.
    
    A query is a question to the user.
    """
    def __init__(self):
        pass
        
    def copy(self):
        """Returns a deep copy of the query."""
        return deepcopy(self)
        
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


class DemonstrationQuery(Query):
    """
    A demonstration query is one where the initial state is given to the user, and they are asked to control the robot.
    
    Although not practical for optimization, this class is defined for coherence with other query types.
    
    Parameters:
        initial_state (numpy.array): The initial state of the environment.
        
    Attributes:
        initial_state (numpy.array): The initial state of the environment.
    """
    def __init__(self, initial_state: np.array):
        super(DemonstrationQuery, self).__init__()
        self.initial_state = initial_state
        
        
class Demonstration(QueryWithResponse):
    """
    The trajectory generated by the DemonstrationQuery, along with the DemonstrationQuery that prompted the user
    with the initial state.
    
    For preference-based reward learning initialized with demonstrations, this class should be used (without
    actually querying the user). First, the demonstration should be collected as a :class:`.Trajectory`
    object. Then, a :class:`.Demonstration` instance should be created with this trajectory without specifying
    the query parameter, in which case it is automatically assigned as the initial state of the trajectory.
    
    Parameters:
        trajectory (Trajectory): The demonstrated trajectory.
        query (DemonstrationQuery): The query that led to the trajectory, i.e., the initial state of the trajectory.
        
    Attributes:
        trajectory (Trajectory): The demonstrated trajectory.
        features (numpy.array): The features of the demonstrated trajectory.
        
    Raises:
        AssertionError: if the initial state of the trajectory does not match with the query.
    """
    def __init__(self, trajectory: Trajectory, query: DemonstrationQuery = None):
        # It is not consistent to put the query as the second argument,
        # but let's keep it because the demonstrations are only passively collected.
        initial_state, _ = trajectory[0]
        if query is None:
            query = DemonstrationQuery(initial_state)
        else:
            assert(np.all(np.isclose(query.initial_state, initial_state))), 'Mismatch between the query and the response for the demonstration.'
        super(Demonstration, self).__init__(query)
        self.trajectory = trajectory
        self.features = trajectory.features


class PreferenceQuery(Query):
    """
    A preference query is one where the user is presented with multiple trajectories and asked for their favorite among them.
    
    Parameters:
        slate (TrajectorySet or List[Trajectory]): The set of trajectories that will be presented to the user.

    Attributes:
        K (int): The number of trajectories in the query.
        response_set (numpy.array): The set of possible responses to the query.
        
    Raises:
        AssertionError: if slate has less than 2 trajectories.
    """
    def __init__(self, slate: Union[TrajectorySet, List[Trajectory]]):
        super(PreferenceQuery, self).__init__()
        assert isinstance(slate, TrajectorySet) or isinstance(slate, list), 'Query constructor requires a TrajectorySet object for the slate.'
        self.slate = slate
        assert(self.K >= 2), 'Preference queries have to include at least 2 trajectories.'
    
    @property
    def slate(self) -> TrajectorySet:
        """Returns a :class:`.TrajectorySet` of the trajectories in the query."""
        return self._slate
    
    @slate.setter
    def slate(self, new_slate: Union[TrajectorySet, List[Trajectory]]):
        """Sets the slate of trajectories in the query."""
        self._slate = new_slate if isinstance(new_slate, TrajectorySet) else TrajectorySet(new_slate)
        self.K = self._slate.size
        self.response_set = np.arange(self.K)
        
    def visualize(self, delay: float = 0.) -> int:
        """Visualizes the query and interactively asks for a response.
        
        Args:
            delay (float): The waiting time between each trajectory visualization in seconds.
            
        Returns:
            int: The response of the user.
        """
        for i in range(self.K):
            print('Playing trajectory #' + str(i))
            time.sleep(delay)
            self.slate[i].visualize()
        selection = None
        while selection is None:
            selection = input('Which trajectory is the best? Enter a number: [0-' + str(self.K-1) + ']: ')
            if not isinteger(selection) or int(selection) not in self.response_set:
                selection = None
        return int(selection)
            

class Preference(QueryWithResponse):
    """
    A Preference feedback.
    
    Contains the :class:`.PreferenceQuery` the user responded to and the response.
    
    Parameters:
        query (PreferenceQuery): The query for which the feedback was given.
        response (int): The response of the user to the query.
        
    Attributes:
        response (int): The response of the user to the query.
        
    Raises:
        AssertionError: if the response is not in the response set of the query.
    """
    def __init__(self, query: PreferenceQuery, response: int):
        super(Preference, self).__init__(query)
        assert(response in self.query.response_set), 'Response ' + str(response) + ' is out of bounds for a slate size of ' + str(self.query.K) + '.'
        self.response = response


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
        super(WeakComparisonQuery, self).__init__()
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
        # for i in range(self.K):
        #     print('Playing trajectory #' + str(i))
        #     time.sleep(delay)
        #     self.slate[i].visualize()
        if self.chart is None:
            ramp1 = Ramping(self.slate[0].trajectory)
            ramp2 = Ramping(self.slate[1].trajectory)
            ramp1.execute()
            ramp2.execute()

            # Show the ramps side by side
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            im1=ax[0].imshow(data2d, cmap=ramp1.cmap)
            im2=ax[1].imshow(data2d, cmap=ramp2.cmap)
            # Show colorbars for both charts
            fig.colorbar(im1, ax=ax[0])
            fig.colorbar(im2, ax=ax[0])

            plt.show()

        selection = None
        while selection is None:
            selection = input('Which trajectory is the best? Enter a number (-1 for "About Equal"): ')
            if not isinteger(selection) or int(selection) not in self.response_set:
                selection = None
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
        super(WeakComparison, self).__init__(query, response)
        assert(response in self.query.response_set), 'Invalid response ' + str(response) +  ' for the weak comparison query.'
        self.response = response


class FullRankingQuery(Query):
    """
    A full ranking query is one where the user is presented with multiple trajectories and asked for a ranking from their most 
    preferred trajectory to the least.
    
    Parameters:
        slate (TrajectorySet or List[Trajectory]): The set of trajectories that will be presented to the user.

    Attributes:
        K (int): The number of trajectories in the query.
        response_set (numpy.array): The set of possible responses to the query, which is all :py:attr:`K`-combinations of the 
            trajectory indices in the slate.
        
    Raises:
        AssertionError: if slate has less than 2 trajectories.
    """
    def __init__(self, slate: Union[TrajectorySet, List[Trajectory]]):
        super(FullRankingQuery, self).__init__()
        assert isinstance(slate, TrajectorySet) or isinstance(slate, list), 'Query constructor requires a TrajectorySet object for the slate.'
        self.slate = slate
        assert(self.K >= 2), 'Ranking queries have to include at least 2 trajectories.'
    
    @property
    def slate(self) -> TrajectorySet:
        """Returns a :class:`.TrajectorySet` of the trajectories in the query."""
        return self._slate
    
    @slate.setter
    def slate(self, new_slate: Union[TrajectorySet, List[Trajectory]]):
        """Sets the slate of trajectories in the query."""
        self._slate = new_slate if isinstance(new_slate, TrajectorySet) else TrajectorySet(new_slate)
        self.K = self._slate.size
        self.response_set = np.array([list(tup) for tup in itertools.permutations(np.arange(self.K))])

    def visualize(self, delay: float = 0.) -> List[int]:
        """Visualizes the query and interactively asks for a response.
        
        Args:
            delay (float): The waiting time between each trajectory visualization in seconds.
            
        Returns:
            List[int]: The response of the user, as a list from the most preferred to the least.
        """
        for i in range(self.K):
            print('Playing trajectory #' + str(i))
            time.sleep(delay)
            self.slate[i].visualize()
        response = []
        i = 1
        while i < self.K:
            selection = None
            while selection is None:
                selection = input('Which trajectory is your #' + str(i) + ' favorite? Enter a number [0-' + str(self.K-1) + ']: ')
                if not isinteger(selection) or int(selection) < 0 or int(selection) >= self.K:
                    selection = None
                elif int(selection) in response:
                    print('You have already chosen trajectory ' + selection + ' before!')
                    selection = None
            response.append(int(selection))
            i += 1
        remaining_id = np.setdiff1d(self.response_set, response)
        response.append(remaining_id.item())
        return np.array(response)


class FullRanking(QueryWithResponse):
    """
    A Full Ranking feedback.
    
    Contains the :class:`.FullRankingQuery` the user responded to and the response.
    
    Parameters:
        query (FullRankingQuery): The query for which the feedback was given.
        response (numpy.array): The response of the user to the query, indices from the most preferred to the least.
        
    Attributes:
        response (numpy.array): The response of the user to the query, indices from the most preferred to the least.
        
    Raises:
        AssertionError: if the response is not in the response set of the query.
    """
    def __init__(self, query: FullRankingQuery, response: List[int]):
        super(FullRanking, self).__init__(query)
        assert(response in self.query.response_set), 'Invalid response ' + str(response) + ' for the ranking query of size ' + str(self.query.K) + '.'
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