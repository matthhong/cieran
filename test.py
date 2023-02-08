# Making APReL work for networkx graphs

# Ideal workflow:
# trajectory_set = crowdsourced corpus
# features_func = feature function

import aprel
import matplotlib as mpl


from matplotlib.colors import ListedColormap

from geomdl import fitting

from coloraide import Color
import numpy as np
    

def main():
    env = aprel.Environment([69.33, -10.77, -24.79])
    env_name = 'Cieran'

    trajectory_set = aprel.generate_trajectories_randomly(env, num_trajectories=100,
                                                        max_episode_length=300,
                                                        file_name=env_name, seed=0)
    features_dim = len(trajectory_set[0].features)

    query_optimizer = aprel.QueryOptimizerDiscreteTrajectorySet(trajectory_set)

    true_user = aprel.HumanUser(delay=0.5)

    # params = {'weights': aprel.util_funs.get_random_normalized_vector(features_dim)}
    params = {'weights': [-1.0, -1.0, -1.0]}
    user_model = aprel.SoftmaxUser(params)
    belief = aprel.SamplingBasedBelief(user_model, [], params)
    print('Estimated user parameters: ' + str(belief.mean))
                                        
    query = aprel.WeakComparisonQuery(trajectory_set[:2])

    for query_no in range(10):
        queries, objective_values = query_optimizer.optimize('disagreement', belief, query, optimization_method='medoids', batch_size=2)
        print('Objective Value: ' + str(objective_values[0]))
        
        responses = true_user.respond(queries[0])
        belief.update(aprel.Preference(queries[0], responses[0]))
        print('Estimated user parameters: ' + str(belief.mean))

    # Qlearning
    env.reward_weights = belief.mean['weights']

    epochs = 1000
    Q = env.Q.copy()
    for i in range(epochs):
        env.run()
        # Test for convergence on Q table values
        if i % 100 == 0:
            print("Epoch {}".format(i))
            # print("Q table: {}".format(env.Q))
            # print("Q table diff: {}".format({k: env.Q[k] - Q[k] for k in env.Q.keys() & Q.keys()}))
            # print("Best path: {}".format(env.get_best_path()))
            Q = env.Q.copy()
        env.reset()

    # Get the path
    return env.get_best_path()


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
            try:
                self.ramp = fitting.interpolate_curve(self.control_points, 3, centripetal=True)
            except:
                self.ramp = fitting.interpolate_curve(self.control_points, 2, centripetal=True)
            
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


if __name__=="__main__":
    path = main()
    # [(72.45506299680495, 19.451625626588225, -14.244807378939939), (61.419906746804955, 20.967400660881793, -22.32320737893994), (54.876937996804955, 21.550391058686984, -26.076807378939947), (46.057357918679955, 23.985995387295446, -33.355527378939954), (37.909164559304955, 32.588342590465686, -32.49056737893994), (27.118148934304955, 36.90247153422425, -40.568967378939945)]
    ramper = Ramping(path[1:-1])
    ramper.execute()

    # Visualize the cmap using matplotlib using a simple colorbar
    fig, ax = mpl.pyplot.subplots(figsize=(6, 1))
    fig.subplots_adjust(bottom=0.5)

    cmap = ramper.cmap
    norm = mpl.colors.Normalize(vmin=0, vmax=1)

    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,

                                    norm=norm,

                                    orientation='horizontal')

    mpl.pyplot.show()


