import random
import math
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import numpy as np
import pickle
import os.path

class LearningAgent(Agent):
    """ An agent that learns to drive in the Smartcab world.
        This is the object you will be modifying. """

    # def __init__(self, env, learning=False, epsilon=1.0, alpha=0.5):
    def __init__(self, env, learning=False, epsilon=1.0, alpha=0.5, dictn={}):
        super(LearningAgent, self).__init__(env)     # Set the agent in the evironment 
        self.planner = RoutePlanner(self.env, self)  # Create a route planner
        self.valid_actions = self.env.valid_actions  # The set of valid actions

        print(learning, epsilon, alpha, dictn)
        # Set parameters of the learning agent
        self.learning = learning # Whether the agent is expected to learn

        # Create a Q-table which will be a dictionary of tuples
        self.Q = dictn

        self.epsilon = epsilon   # Random exploration factor
        self.alpha = alpha       # Learning factor

        # Set any additional class parameters as needed
        self.createQ()

        self.step = 1
        self.a = random.random() # Random no from [0.0,1) for the decaying func   epsilon = exp ^ (- a * t)  where t = step

    def reset(self, destination=None, testing=False):
        """ The reset function is called at the beginning of each trial.
            'testing' is set to True if testing trials are being used
            once training trials have completed. """
        # Select the destination as the new location to route to
        self.planner.route_to(destination)
        # Update epsilon using a decay function of your choice
        
        # For Non-Optimised Phase
        # self.epsilon = self.epsilon - 0.05 

        ######   Optimized   ######
        # -----------------------------------------------
        ## 1st epsilon dacay -> epsilon = exp(-at)

        # if random.random() < self.epsilon:

        #     self.epsilon = np.exp(-(self.a * self.step))
        #     self.step = self.step + 1

        # Rating -> A+, A
        # -----------------------------------------------
        ## 2nd epsilon decay -> epsilon = a ^ t

        if random.random() < self.epsilon:

            self.epsilon = np.power(self.a, self.step)
            self.step = self.step + 1

            self.alpha += 0.01 

        # Rating -> A+, A+ 
        # -----------------------------------------------
        ## 3rd epsilon decay -> epsilon = 1 / (t ^ 2)

        # if random.random() < self.epsilon:

        #     self.epsilon = 1.0 / np.power(self.step, 2)
        #     self.step = self.step + 1

        # Takes very long time
        # Rating -> A+, A 
        # -----------------------------------------------
        ## 4th epsilon decay -> epsilon = cos(at)

        # if random.random() < self.epsilon:

        #     self.epsilon = np.cos(self.a * self.step)
        #     self.step = self.step + 1

        # Rating -> A+, B
        # -----------------------------------------------

        #############     Winner -> 2nd      ##############

        # self.epsilon = np.power(self.a, self.step)
        # self.step = self.step + 1
        # self.alpha += 0.01


        # Update additional class parameters as needed
        # If 'testing' is True, set epsilon and alpha to 0
        # self.epsilon = 0
        # self.alpha = 0

        return None

    def build_state(self):
        """ The build_state function is called when the agent requests data from the 
            environment. The next waypoint, the intersection inputs, and the deadline 
            are all features available to the agent. """
        # Collect data about the environment
        waypoint = self.planner.next_waypoint() # The next waypoint 
        inputs = self.env.sense(self)           # Visual input - intersection light and traffic
        deadline = self.env.get_deadline(self)  # Remaining deadline
        # Set 'state' as a tuple of relevant data for the agent
        # When learning, check if the state is in the Q-table
        #   If it is not, create a dictionary in the Q-table for the current 'state'
        #   For each action, set the Q-value for the state-action pair to 0
        # state = (waypoint, inputs['light'], inputs['oncoming'])

        dict_state = {'light' : inputs['light'], 'oncoming' : inputs['oncoming'], 'direction' : waypoint}

        return dict_state

    def get_maxQ(self, state):
        """ The get_maxQ function is called when the agent is asked to find the
            maximum Q-value of all actions based on the 'state' the smartcab is in. """
        # Calculate the maximum Q-value of all actions for a given state
        state_key = (self.state['light'], self.state['oncoming'], self.state['direction'])

        Q_item = self.Q[state_key]
        max_Q = max(Q_item.values()) # find the max Q value

        return max_Q 

    def createQ(self):
        """ The createQ function is called when a state is generated by the agent. """
        # When learning, check if the 'state' is not in the Q-table
        # If it is not, create a new dictionary for that state
        #   Then, for each action available, set the initial Q-value to 0.0

        if len(self.Q) == 0 :
            print('Creating a new Empty Dict')
            valid_actions = self.valid_actions

            for light in ['green','red'] :
                for oncoming in valid_actions :
                    for direction in valid_actions:
                        self.Q[(light, oncoming, direction)] = {'forward':0.0, 'left':0.0, 'right':0.0, None:0.0}

        return

    def printDict(self):
        return self.Q

    def choose_action(self, state):
        """ The choose_action function is called when the agent is asked to choose
            which action to take, based on the 'state' the smartcab is in. """
        # Set the agent state and default action
        self.state = state
        self.next_waypoint = self.planner.next_waypoint()
        # When not learning, choose a random action
        # When learning, choose a random action with 'epsilon' probability
        #   Otherwise, choose an action with the highest Q-value for the current state

        state_key = (self.state['light'], self.state['oncoming'], self.state['direction'])
        Q_item = self.Q[state_key]

        max_Q = self.get_maxQ(state)

        actionss = []
        for key, value in Q_item.iteritems():
            if value == max_Q:
                actionss.append(key)
        action = random.choice(actionss)

        ### For non-learning phase
        # action = random.choice(self.valid_actions)

        return action

    def learn(self, state, action, reward):
        """ The learn function is called after the agent completes an action and
            receives an award. This function does not consider future rewards 
            when conducting learning. """
        # When learning, implement the value iteration update rule
        #   Use only the learning rate 'alpha' (do not use the discount factor 'gamma')
        
        learning_rate = self.alpha

        # transform state to the Q dictionary key
        state_key = (state['light'], state['oncoming'], state['direction'])

        inputs = self.env.sense(self)
        direction = self.planner.next_waypoint()

        # Q learning equation
        self.Q[state_key][action] = (1 - learning_rate) * self.Q[state_key][action] + learning_rate * reward


        #------------- Uncomment to use Discount Factor Gamma  ------------------------
        # gamma = 0.5
        # next_state_key = (inputs['light'], inputs['oncoming'], state['direction'])
        # max_Q = max(self.Q[next_state_key].values())
        # self.Q[state_key][action] = (1 - learning_rate) * self.Q[state_key][action] + learning_rate * (reward + gamma * max_Q)
        #------------- Uncomment to use Discount Factor Gamma  ------------------------

        ###### Using Gamma made results worse in this case
        ###### Rating came down from A+, A+  to  A+, C

        return

    def update(self):
        """ The update function is called when a time step is completed in the 
            environment for a given trial. This function will build the agent
            state, choose an action, receive a reward, and learn if enabled. """

        state = self.build_state()          # Get current state
        # self.createQ(state)                 # Create 'state' in Q-table
        action = self.choose_action(state)  # Choose an action
        # if action != 'None' :
        reward = self.env.act(self, action) # Receive a reward
        self.learn(state, action, reward)   # Q-learn

        return
        

def run():
    isOptimized = True
    isLearning = True
    
    dict_name = 'Q_learned_dict'
    dict_name_pikle = 'Q_learned_dict.pkl'

    def load_dict():
        with open(dict_name + '.pkl', 'rb') as f:
            return pickle.load(f)

    def save_dict(obj):
        with open( dict_name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    """ Driving function for running the simulation. 
        Press ESC to close the simulation, or [SPACE] to pause the simulation. """
    
    ##############
    # Create the environment
    # Flags:
    #   verbose     - set to True to display additional output from the simulation
    #   num_dummies - discrete number of dummy agents in the environment, default is 100
    #   grid_size   - discrete number of intersections (columns, rows), default is (8, 6)
    env = Environment()
        
    ##############
    # Create the driving agent
    # Flags:
    #    * learning   - set to True to force the driving agent to use Q-learning
    #    * epsilon - continuous value for the exploration factor, default is 1
    #    * alpha   - continuous value for the learning rate, default is 0.5
    Q_dict = {}

    if (isLearning and os.path.isfile(dict_name_pikle) and isOptimized) :
        print('inside_if_condition')
        Q_dict = load_dict()
    print(len(Q_dict))

    l = LearningAgent(env, learning=isLearning, alpha=0.6, epsilon=0.20, dictn=Q_dict)
    agent = env.create_agent_new(l)#, learning=True, alpha=0.5, epsilon=0.015)
    
    ##############
    # Follow the driving agent
    # Flags:
    #   enforce_deadline - set to True to enforce a deadline metric
    env.set_primary_agent(agent, enforce_deadline=True)

    ##############
    # Create the simulation
    # Flags:
    #   update_delay - continuous time (in seconds) between actions, default is 2.0 seconds
    #   display      - set to False to disable the GUI if PyGame is enabled
    #   log_metrics  - set to True to log trial and simulation results to /logs
    #   optimized    - set to True to change the default log file name
    sim = Simulator(env, update_delay=0.01, log_metrics=True, optimized=isOptimized, display=False)
    
    ##############
    # Run the simulator
    # Flags:
    #   tolerance  - epsilon tolerance before beginning testing, default is 0.05 
    #   n_test     - discrete number of testing trials to perform, default is 0
    sim.run(n_test=50, tolerance=0.01)
   
    new_dict = LearningAgent.printDict(l) 
    
    if (isLearning and (not isOptimized)) :
        save_dict(new_dict)
        print('Directory Saved')

if __name__ == '__main__':
    run()
