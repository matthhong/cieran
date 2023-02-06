from aprel.assessing.metrics import cosine_similarity

from aprel.basics.environment import Environment
from aprel.basics.moving_dot_env import MovingDotEnv
from aprel.basics.trajectory import Trajectory, TrajectorySet

from aprel.learning.data_types import Query, QueryWithResponse
from aprel.learning.data_types import DemonstrationQuery, Demonstration
from aprel.learning.data_types import PreferenceQuery, Preference
from aprel.learning.data_types import WeakComparisonQuery, WeakComparison
from aprel.learning.data_types import FullRankingQuery, FullRanking
from aprel.learning.user_models import User, SoftmaxUser, HumanUser
from aprel.learning.belief_models import Belief, LinearRewardBelief, SamplingBasedBelief

from aprel.querying.acquisition_functions import mutual_information, volume_removal, disagreement, regret, random, thompson
from aprel.querying.query_optimizer import QueryOptimizer, QueryOptimizerDiscreteTrajectorySet

from aprel.utils.generate_trajectories import generate_trajectories_randomly
from aprel.utils.sampling_utils import uniform_logprior, gaussian_proposal
from aprel.utils.kmedoids import kMedoids
from aprel.utils.dpp import dpp_mode
from aprel.utils.batch_utils import default_query_distance
import aprel.utils.util_functions as util_funs

from gym.envs.registration import register

register(
    id='MovingDot-v1',
    entry_point='aprel.basics.moving_dot_env:MovingDotEnv'
)

register(
    id='ColorReacher-v1',
    entry_point='aprel.basics.color_reacher_env:ColorReacherEnv'
)

__version__ = "1.0.0"
