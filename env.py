import numpy as np
import matplotlib.pyplot as plt
import _env
import time

class World(_env.Hidden):

    def __init__(self):

        self.nRows = 4
        self.nCols = 5
        self.stateInitial = [4]
        self.regular_states = [3,4,5,6,7,8,9,11,13,14,15,16,18,19]
        self.stateTerminals = [1, 2,  10, 12, 17, 20]
        self.stateObstacles = []
        self.stateHoles = [1, 2,  10, 12, 20]
        self.stateGoal = [17]
        self.nStates = 20
        self.nActions = 4

        self.observation = 4  # initial state

    @staticmethod
    def _truncate(n, decimals=0):
        multiplier = 10 ** decimals
        return int(n * multiplier) / multiplier

    def _plot_world(self):

        nRows = self.nRows
        nCols = self.nCols
        stateObstacles = self.stateObstacles
        stateTerminals = self.stateTerminals
        stateGoal      = self.stateGoal
        coord = [[0, 0], [nCols, 0], [nCols, nRows], [0, nRows], [0, 0]]
        xs, ys = zip(*coord)
        plt.plot(xs, ys, "black")
        for i in stateObstacles:
            (I, J) = np.unravel_index(i-1, shape=(nRows, nCols), order='F')
            I = I + 1
            J = J
            coord = [[J, nRows - I],
                     [J + 1, nRows - I],
                     [J + 1, nRows - I + 1],
                     [J, nRows - I + 1],
                     [J, nRows - I]]
            xs, ys = zip(*coord)
            plt.fill(xs, ys, "0.3")
            plt.plot(xs, ys, "black")
        for i in stateTerminals:
            #print("stateTerminal", i)
            (I, J) = np.unravel_index(i-1, shape=(nRows, nCols), order='F')
            I = I + 1
            J = J
            #print("I,J = ", I,J)
            coord = [[J, nRows - I],
                     [J + 1, nRows - I],
                     [J + 1, nRows - I + 1],
                     [J, nRows - I + 1],
                     [J, nRows - I]]
            xs, ys = zip(*coord)
            #print("coord", xs,ys)
            plt.fill(xs, ys, "0.6")
            plt.plot(xs, ys, "black")
        for i in stateGoal:
            (I, J) = np.unravel_index(i-1, shape=(nRows, nCols), order='F')
            I = I + 1
            J = J
            #print("I,J = ", I,J)
            coord = [[J, nRows - I],
                     [J + 1, nRows - I],
                     [J + 1, nRows - I + 1],
                     [J, nRows - I + 1],
                     [J, nRows - I]]
            xs, ys = zip(*coord)
            #print("coord", xs,ys)
            plt.fill(xs, ys, "0.9")
            plt.plot(xs, ys, "black")
        plt.plot(xs, ys, "black")
        X, Y = np.meshgrid(range(nCols + 1), range(nRows + 1))
        plt.plot(X, Y, 'k-')
        plt.plot(X.transpose(), Y.transpose(), 'k-')

    @staticmethod
    def _truncate(n, decimals=0):
        multiplier = 10 ** decimals
        return int(n * multiplier) / multiplier

    def plot(self):
        """
        plot function
        :return: None
        """
        nStates = self.nStates
        nRows = self.nRows
        nCols = self.nCols
        self._plot_world()
        states = range(1, nStates + 1)
        k = 0
        for i in range(nCols):
            for j in range(nRows, 0, -1):
                plt.text(i + 0.5, j - 0.5, str(states[k]), fontsize=26, horizontalalignment='center', verticalalignment='center')
                k += 1
        plt.title('gridworld', size=16)
        plt.axis("equal")
        plt.axis("off")
        plt.show()

    def plot_value(self, valueFunction, title):

        """
        plot state value function V

        :param policy: vector of values of size nStates x 1
        :return: None
        """

        nStates = self.nStates
        nRows = self.nRows
        nCols = self.nCols
        stateObstacles = self.stateObstacles
        self._plot_world()
        k = 0
        for i in range(nCols):
            for j in range(nRows, 0, -1):
                if k + 1 not in stateObstacles:
                    plt.text(i + 0.5, j - 0.5, str(self._truncate(np.round(valueFunction[k],4),3)), fontsize=16, horizontalalignment='center', verticalalignment='center')
                k += 1
        # label states by numbers
        states = range(1, nStates + 1)
        k = 0
        for i in range(nCols):
            for j in range(nRows, 0, -1):
                plt.text(i + 0.92, j - 0.92, str(states[k]), fontsize=11, horizontalalignment='right',verticalalignment='bottom')
                k += 1

        plt.title(f'gridworld - {title}', size=16)
        plt.axis("equal")
        plt.axis("off")
        plt.savefig(title, dpi=500)
        plt.show()

    def plot_policy(self, policy, title):

        """
        plot (stochastic) policy

        :param policy: matrix of policy of size nStates x nActions
        :return: None
        """
        # remove values below 1e-6
        policy = policy * (np.abs(policy) > 1e-6)


        nStates = self.nStates
        nActions = self.nActions
        nRows = self.nRows
        nCols = self.nCols
        stateObstacles = self.stateObstacles
        stateTerminals = self.stateTerminals
        #policy = policy.reshape(nRows, nCols, order="F").reshape(-1, 1)
        # generate mesh for grid world
        X, Y = np.meshgrid(range(nCols + 1), range(nRows + 1))
        # generate locations for policy vectors
        #print("X = ", X)
        X1 = X.transpose()
        X1 = X1[:-1, :-1]
        #print("X1 = ", X1)
        Y1 = Y.transpose()
        Y1 = Y1[:-1, :-1]
        #print("Y1 =", Y1)
        X2 = X1.reshape(-1, 1) + 0.5
        #print("X2 = ", X2)
        Y2 = np.flip(Y1.reshape(-1, 1)) + 0.5
        #print("Y2 = ", Y2)
        # reshape to matrix
        X2 = np.kron(np.ones((1, nActions)), X2)
        #print("X2 after kron = ", X2)
        Y2 = np.kron(np.ones((1, nActions)), Y2)
        #print("X2 = ",X2)
        #print("Y2 = ",Y2)
        # define an auxiliary matrix out of [1,2,3,4]
        mat = np.cumsum(np.ones((nStates , nActions)), axis=1).astype("int64")
        #print("mat = ", mat)
        # if policy vector (policy deterministic) turn it into a matrix (stochastic policy)
        #print("policy.shape[1] =", policy.shape[1])
        if policy.shape[1] == 1:
            policy = (np.kron(np.ones((1, nActions)), policy) == mat)
            policy = policy.astype("int64")
            print("policy inside", policy)
        # no policy entries for obstacle and terminal states
        index_no_policy = stateObstacles + stateTerminals
        index_policy = [item - 1 for item in range(1, nStates + 1) if item not in index_no_policy]
        #print("index_policy", index_policy)
        #print("index_policy[0]", index_policy[0:2])
        mask = (policy > 0) * mat
        #print("mask", mask)
        #mask = mask.reshape(nRows, nCols, nCols)
        #X3 = X2.reshape(nRows, nCols, nActions)
        #Y3 = Y2.reshape(nRows, nCols, nActions)
        #print("X3 = ", X3)
        # print arrows for policy
        # [N, E, S, W] = [up, right, down, left] = [pi, pi/2, 0, -pi/2]
        alpha = np.pi - np.pi / 2.0 * mask
        #print("alpha", alpha)
        #print("mask ", mask)
        #print("mask test ", np.where(mask[0, :] > 0)[0])
        self._plot_world()
        for i in index_policy:
            #print("ii = ", ii)
            ax = plt.gca()
            #j = int(ii / nRows)
            #i = (ii + 1 - j * nRows) % nCols - 1
            #index = np.where(mask[i, j] > 0)[0]
            index = np.where(mask[i, :] > 0)[0]
            #print("index = ", index)
            #print("X2,Y2", X2[ii, index], Y2[ii, index])
            h = ax.quiver(X2[i, index], Y2[i, index], np.cos(alpha[i, index]), np.sin(alpha[i, index]), color='b')
            #h = ax.quiver(X3[i, j, index], Y3[i, j, index], np.cos(alpha[i, j, index]), np.sin(alpha[i, j, index]),0.3)

        # label states by numbers
        states = range(1, nStates + 1)
        k = 0
        for i in range(nCols):
            for j in range(nRows, 0, -1):
                plt.text(i + 0.92, j - 0.92, str(states[k]), fontsize=11, horizontalalignment='right', verticalalignment='bottom')
                k += 1
        plt.title(f'gridworld - {title}', size=16)
        plt.axis("equal")
        plt.axis("off")
        plt.savefig(title,dpi=500)
        plt.show()

    def plot_qvalue(self, Q, title):
        """
        plot Q-values

        :param Q: matrix of Q-values of size nStates x nActions
        :return: None
        """
        nRows = self.nRows
        nCols = self.nCols
        stateHoles = self.stateHoles
        stateGoal = self.stateGoal
        stateObstacles = self.stateObstacles

        fig = plt.plot(1)

        self._plot_world()
        k = 0
        for i in range(nCols):
            for j in range(nRows, 0, -1):
                if k + 1 not in stateHoles + stateObstacles + stateGoal:
                    #print("Q = ", Q)
                    plt.text(i + 0.5, j - 0.15, str(self._truncate(Q[k, 0], 3)), fontsize=7,
                             horizontalalignment='center', verticalalignment='top', multialignment='center')
                    plt.text(i + 0.95, j - 0.5, str(self._truncate(Q[k, 1], 3)), fontsize=7,
                             horizontalalignment='right', verticalalignment='center', multialignment='right')
                    plt.text(i + 0.5, j - 0.85, str(self._truncate(Q[k, 2], 3)), fontsize=7,
                             horizontalalignment='center', verticalalignment='bottom', multialignment='center')
                    plt.text(i + 0.05, j - 0.5, str(self._truncate(Q[k, 3], 3)), fontsize=7,
                             horizontalalignment='left', verticalalignment='center', multialignment='left')
                    # plot cross
                    plt.plot([i, i + 1], [j - 1, j], 'black', lw=0.5)
                    plt.plot([i + 1, i], [j - 1, j], 'black', lw=0.5)
                k += 1

        plt.title(f'gridworld - {title}', size=16)
        plt.axis("equal")
        plt.axis("off")
        plt.savefig(title, dpi=500)
        plt.show()

    def get_nrows(self):

        return self.nRows

    def get_ncols(self):

        return self.nCols

    def get_stateTerminals(self):

        return self.stateTerminals

    def get_stateHoles(self):

        return self.stateHoles

    def get_stateObstacles(self):

        return self.stateObstacles

    def get_stateGoal(self):

        return self.stateGoal

    def get_nstates(self):

        return self.nStates

    def get_nactions(self):

        return self.nActions

    def step(self,action):

        nStates = self.nStates
        stateGoal = self.get_stateGoal()
        stateTerminals = self.get_stateTerminals()

        state = self.observation[0]


        # generate reward and transition model
        p_success = 0.8
        r = -0.04
        
        Pr = self.transition_model
        R = self.reward

        prob = np.array(Pr[state-1, :, action])
        #print("prob =", prob)
        next_state = np.random.choice(np.arange(1, nStates + 1), p = prob)
        #print("state = ", state)
        #print("next_state inside = ", next_state)
        #print("action = ", action)
        reward = R[state-1, next_state-1, action]
        #print("reward = ", R[:, :, 0])
        observation = next_state

        #if (next_state in stateTerminals) or (self.nsteps >= self.max_episode_steps):
        if (next_state in stateTerminals):
            done = True
        else:
            done = False

        self.observation = [next_state]


        return observation, reward, done

    def reset(self, *args):


        nStates = self.nStates

        if not args:

            observation = self.stateInitial #[np.random.choice([3,4,5,6,7,8,9,11,13,14,15,16,19,18])]
        else:
            observation = []
            while not (observation):
                observation = np.setdiff1d(np.random.choice(np.arange(1, nStates +  1, dtype = int)), self.stateHoles + self.stateObstacles + self.stateGoal)
        self.observation = observation

    def render(self):

        nStates = self.nStates
        nActions = self.nActions
        nRows = self.nRows
        nCols = self.nCols
        stateHoles = self.stateHoles
        stateGoal = self.stateGoal

        observation = self.observation #observation
        state = observation[0]


        J, I = np.unravel_index(state - 1, (nRows, nCols), order='F')



        J = (nRows -1) - J



        circle = plt.Circle((I+0.5,J+0.5), 0.28, color='black')
        fig = plt.gcf()
        ax = fig.gca()
        ax.add_artist(circle)

        self.plot()

    def close(self):
        plt.pause(0.3) #0.5
        plt.close()

    def show(self):
        plt.ion()
        plt.show()

    def initialize_Q(self):
        Q = np.ones((self.nStates, self.nActions)) * (1 / self.nActions)
        for state in self.get_stateTerminals():
            state -= 1
            Q[state, :] = np.zeros((self.nActions))
        return Q

        

    def get_our_transition_model(self,p_success):
        self.our_transition_model = {'3':{'0':[2,3,7], '1':[7,2,4], '2':[4,3,7], '3':[3,2,4]}, '4':{'0':[3,4,8], '1':[8,3,4], '2':[4,4,8], '3':[4,4,3]}, '5':{'0':[5,1,9], '1':[9,6,5], '2':[6,1,9], '3':[1,6,5]}, '6':{'0':[5,2,10], '1':[10,5,7], '2':[7,2,10], '3':[2,5,7]}, '7':{'0':[6,3,11], '1':[11,6,8], '2':[8,3,11], '3':[3,6,8]}, '8':{'0':[7,4,12], '1':[12,8,7], '2':[8,4,12], '3':[4,8,7]}, '9':{'0':[9,5,13], '1':[13,10,9], '2':[10,5,13], '3':[5,10,9]}, '11':{'0':[10,7,15], '1':[15,10,12], '2':[12,7,15], '3':[7,10,12]}, '13':{'0':[13,9,17], '1':[17,14,13], '2':[14,9,17], '3':[9,13,14]}, '14':{'0':[13,10,18], '1':[18,13,15], '2':[15,10,18], '3':[10,13,15]}, '15':{'0':[14,11,19], '1':[19,16,14], '2':[16,11,19], '3':[11,16,14]}, '16':{'0':[15,12,20], '1':[20,15,16], '2':[16,12,20], '3':[12,16,15]}, '18':{'0':[17,14,18], '1':[18,19,17], '2':[19,18,14], '3':[14,19,17]}, '19':{'0':[18,19,15], '1':[19,20,18], '2':[20,19,15], '3':[15,20,18]}}  
        self.p_success = p_success
        self.p_fail = (1-p_success)/2
        
    def get_our_reward_model(self,r, p_success):
        self.our_reward = {'3':r, '4':r, '5':r, '6':r, '7':r, '8':r, '9':r, '11':r, '13':r, '14':r, '15':r, '16':r, '18':r, '19':r, '1':-1+r, '2':-1+r, '10':-1+r, '12':-1+r, '20':-1+r, '17':1+r}  



    # Question 1
    def policy_iteration(self,theta, gamma):
        '''

        :param theta: the threshold between two values functions that represents we have converged
        :return: optimal policy, Q - state/action value function of the optimal policy, V - state value function of the optimal policy
        '''
        def policy_evaluation(policy, theta):
            '''
            this function evaluate a certain policy
            :param policy: given policy
            :param theta: threshold
            :return: V - state value function
            '''
            # initialize the value function
            V = np.zeros((self.nStates))
            probs_vec = [self.p_success, self.p_fail, self.p_fail]
            while True:
                delta = float(0)

                for state in self.regular_states:
                    old_v = V[state-1]
                    new_v = 0
                    for action in range(self.nActions):
                        current_sum = 0
                        
                        for next_state,prob in zip(self.our_transition_model[str(state)][str(action)],probs_vec):
                                current_sum += (self.our_reward[str(next_state)] + gamma * V[next_state-1]) * prob
                        new_v += current_sum * policy[state-1,action]
                    V[state-1] = new_v
                    delta = max(delta,np.abs(new_v-old_v))

                if delta < theta:
                    break
            return V

        def policy_improvement(V):
            '''
            :param V: Given value matrix
            :return: greedy policy
            '''
            probs_vec = [self.p_success, self.p_fail, self.p_fail]
            Q = np.zeros((self.nStates,self.nActions))
            for state in self.regular_states:
                for action in range(self.nActions):
                    Q[state-1,action] = np.sum([(self.our_reward[str(next_state)] + gamma * V[next_state-1]) * prob for next_state,prob in zip(self.our_transition_model[str(state)][str(action)],probs_vec)])
            chosen_actions = np.argmax(Q, axis=1)
            policy = np.zeros_like(Q)
            policy = np.eye(self.nActions)[chosen_actions]
            return policy

        def get_action_value_function(V):
            '''
            :param V: the Value function
            :return: Q: state action value function
            '''
            probs_vec = [self.p_success, self.p_fail, self.p_fail]
            Q = np.zeros((self.nStates,self.nActions))
            for state in self.regular_states:
                for action in range(self.nActions):
                    Q[state-1,action] = np.sum([(self.our_reward[str(next_state)] + gamma * V[next_state-1]) * prob for next_state,prob in zip(self.our_transition_model[str(state)][str(action)],probs_vec)])
            return Q

        # Initialize arbitraly policy
        self.policy = np.ones((self.nStates, self.nActions)) * 1 / self.nActions
        p_success = 0.8
        r = -0.04
        self.get_our_transition_model(p_success)
        self.get_our_reward_model(r, p_success)
        policy_stable = False

        while policy_stable is False:
            V = policy_evaluation(self.policy, theta)
            policy = policy_improvement(V)
            if (policy == self.policy).all():
                policy_stable = True
            self.policy = policy

        V = policy_evaluation(self.policy, theta)

        return self.policy, V, get_action_value_function(V)

    # Question 2
    def mc_control(self, num_episodes, alpha, gamma, optimal_V=None):
        '''
        :param num_episodes: the number of episodes that we are running
        :param alpha: the step size
        :return: policy, Q - state/action value function, V - state value function
        '''

        def epsilon_greedy(policy,state, epsilon):
            '''
            choose action following epsilon greedy exploration.
            :param policy: the current policy
            :param state: the current state
            :param epsilon: the current epsilon which decides if we choose random action or to choose action from our policy
            :return: chosen action
            '''
            if np.random.uniform() < epsilon:
                action = np.random.randint(self.nActions, size=1)[0]
            else:
                action =  np.argmax(policy[state, :])

            return action

        def get_policy(Q):
            chosen_actions = np.argmax(Q, axis=1)
            policy = np.eye(self.nActions)[chosen_actions]
            return policy

        def get_epsilon(n_episode):
            epsilon = 1/(n_episode * epsilon_decay + 1)
            return epsilon

        def calculate_RMS_distance():
            if optimal_V is not None:
                V = np.max(Q, axis=1)
                rms_error = (np.sqrt(np.mean((V - optimal_V) ** 2)))
                return rms_error
            return 0



        # end_epsilon = 0.01
        epsilon_decay = 0.1

        Q = self.initialize_Q()

        p_success = 0.8
        r = -0.04
        self.get_transition_model(p_success)
        self.get_reward_model(r,p_success)
        RMS_ERROR_VEC = np.zeros((num_episodes))

        for episode in range(num_episodes):
            policy = get_policy(Q)
            self.reset()
            # generate trajectory
            trajectory = []
            done = False
            state = self.observation[0]

            first_visit_indicators = np.zeros((self.nStates, self.nActions))
            epsilon = get_epsilon(episode)
            while not done:
                state -= 1
                # choose action
                action = epsilon_greedy(policy, state, epsilon)

                # check if we have already visited this state
                if first_visit_indicators[state,action] == 0:
                    first_visit = True
                    first_visit_indicators[state, action] = 1

                else:
                    first_visit = False

                # apply the action to the enviornemnt
                next_state,reward,done = self.step(action)

                # save the data to the current trajectory
                trajectory.append((state,action,reward,first_visit))

                # now update the current state to be the next state
                state = next_state



            trajectory.reverse()
            G_t = 0
            for state, action, reward, first_visit in trajectory:
                G_t = reward + gamma * G_t
                if first_visit is True:
                    Q[state,action] = Q[state,action] + alpha * (G_t - Q[state,action])

            RMS_ERROR_VEC[episode] = calculate_RMS_distance()

        return get_policy(Q), Q, np.max(Q,axis=1), RMS_ERROR_VEC






            # Question 1

    # Question 3
    def sarsa(self,num_episodes,alpha,gamma, optimal_V = None):

        def epsilon_greedy(Q,state, epsilon):
            '''
            choose action following epsilon greedy exploration.
            :param policy: the current policy
            :param state: the current state
            :param epsilon: the current epsilon which decides if we choose random action or to choose action from our policy
            :return: chosen action
            '''
            if np.random.uniform() < epsilon:
                action = np.random.randint(self.nActions, size=1)[0]
            else:
                action = np.argwhere(Q[state,:] == np.amax(Q[state,:]))
                number_of_actions = action.size
                if number_of_actions > 1:
                    prob = [1 / number_of_actions] * number_of_actions
                    action = np.random.choice(np.squeeze(action), p=prob)
                else:
                    action = action[0][0]

            return action

        def get_policy(Q):
            chosen_actions = np.argmax(Q, axis=1)
            policy = np.eye(self.nActions)[chosen_actions]
            return policy

        def get_epsilon(n_episode):
            epsilon = 1 / (n_episode * epsilon_decay + 1)
            return epsilon

        def calculate_RMS_distance():
            if optimal_V is not None:
                V = np.max(Q, axis=1)
                rms_error = (np.sqrt(np.mean((V - optimal_V) ** 2)))
                return rms_error
            return 0

        epsilon_decay = 0.1

        Q = self.initialize_Q()

        p_success = 0.8
        r = -0.04
        self.get_transition_model(p_success)
        self.get_reward_model(r, p_success)

        RMS_ERROR_VEC = np.zeros((num_episodes))

        for episode in range(num_episodes):
            done = False
            self.reset()
            # observe the current state
            state = self.observation[0]
            # calculate the epsilon
            epsilon = get_epsilon(episode)
            # get first action
            action = epsilon_greedy(Q,state,epsilon)
            state -= 1
            while not done:
                # apply the action to the enviornemnt
                next_state,reward,done = self.step(action)
                next_state -= 1
                # choose next action
                next_action = epsilon_greedy(Q,next_state,epsilon)
                # update the q
                Q[state,action] += alpha * (reward + gamma * Q[next_state,next_action] - Q[state,action])
                state = next_state
                action = next_action
            RMS_ERROR_VEC[episode] = calculate_RMS_distance()

        return get_policy(Q), Q, np.max(Q,axis=1), RMS_ERROR_VEC

    # Question 4
    def q_learning(self,num_episodes,alpha, gamma, optimal_V = None):

        def get_epsilon(n_episode):
            if n_episode < 1e3:
                return 1
            epsilon = 1 / ((n_episode-1e3) * epsilon_decay + 1)
            return epsilon

        def epsilon_greedy(Q,state, epsilon):
            '''
            choose action following epsilon greedy exploration.
            :param policy: the current policy
            :param state: the current state
            :param epsilon: the current epsilon which decides if we choose random action or to choose action from our policy
            :return: chosen action
            '''
            if np.random.uniform() < epsilon:
                action = np.random.randint(self.nActions, size=1)[0]
            else:
                action = np.argwhere(Q[state, :] == np.amax(Q[state, :]))
                number_of_actions = action.size
                if number_of_actions > 1:
                    prob = [1 / number_of_actions] * number_of_actions
                    action = np.random.choice(np.squeeze(action), p=prob)
                else:
                    action = action[0][0]

            return action

        def get_policy(Q):
            chosen_actions = np.argmax(Q, axis=1)
            policy = np.eye(self.nActions)[chosen_actions]
            return policy

        def calculate_RMS_distance():
            if optimal_V is not None:
                V = np.max(Q, axis=1)
                rms_error = (np.sqrt(np.mean((V - optimal_V) ** 2)))
                return rms_error
            return 0

        epsilon_decay = 0.0001

        Q = self.initialize_Q()

        p_success = 0.8
        r = -0.04
        self.get_transition_model(p_success)
        self.get_reward_model(r, p_success)

        RMS_ERROR_VEC = np.zeros((num_episodes))

        for episode in range(num_episodes):
            done = False
            self.reset()
            # observe the current state
            state = self.observation[0]
            # calculate the epsilon
            epsilon = get_epsilon(episode)
            state -= 1
            while not done:
                # Choose action
                action = epsilon_greedy(Q, state, epsilon)
                # apply the action to the enviornemnt
                next_state, reward,done = self.step(action)
                next_state -= 1
                # update the q state-action value function
                Q[state,action] += alpha * (reward + gamma * np.max(Q[next_state,:]) - Q[state,action])
                state = next_state

            RMS_ERROR_VEC[episode] = calculate_RMS_distance()

        return get_policy(Q), Q, np.max(Q,axis=1), RMS_ERROR_VEC
