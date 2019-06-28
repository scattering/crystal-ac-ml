"""
Code adapted frm:
https://towardsdatascience.com/reinforcement-learning-w-keras-openai-actor-critic-models-f084612cfd69
solving pendulum using actor-critic model
"""

import gym
import sys,os
import numpy as np 
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers.merge import Add, Multiply
from keras.optimizers import Adam
import keras.backend as K

import tensorflow as tf

import random
from collections import deque

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

class ActorCritic:

    def __init__(self, env, sess):
        self.env  = env
        self.sess = sess

        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = .9995
        self.gamma = .95
        self.tau   = .125

        # ===================================================================== #
        #                               Actor Model                             #
        # Chain rule: find the gradient of chaging the actor network params in  #
        # getting closest to the final value network predictions, i.e. de/dA    #
        # Calculate de/dA as = de/dC * dC/dA, where e is error, C critic, A act #
        # ===================================================================== #

        self.memory = deque(maxlen=2000)
        self.actor_state_input, self.actor_model = self.create_actor_model()
        _, self.target_actor_model = self.create_actor_model()

        self.actor_critic_grad = tf.placeholder(tf.float32, 
            [None, 198]) # where we will feed de/dC (from critic)

        actor_model_weights = self.actor_model.trainable_weights
        self.actor_grads = tf.gradients(self.actor_model.output, 
            actor_model_weights, -self.actor_critic_grad) # dC/dA (from actor)

        grads = list(zip(self.actor_grads, actor_model_weights))
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)

        # ===================================================================== #
        #                              Critic Model                             #
        # ===================================================================== #

        self.critic_state_input, self.critic_action_input, \
            self.critic_model = self.create_critic_model()
        _, _, self.target_critic_model = self.create_critic_model()

        self.critic_grads = tf.gradients(self.critic_model.output, 
            self.critic_action_input) # where we calcaulte de/dC for feeding above

        # Initialize for later gradient calculations
        self.sess.run(tf.initialize_all_variables())

    # ========================================================================= #
    #                              Model Definitions                            #
    # ========================================================================= #

    def create_actor_model(self):
        state_input = Input(shape=self.env.observation_space.shape)
        h1 = Dense(24, activation='relu')(state_input)
        h2 = Dense(48, activation='relu')(h1)
        h3 = Dense(24, activation='relu')(h2)
        output = Dense(self.env.action_space.n, activation='relu')(h3)  #edited shape

        model = Model(input=state_input, output=output)
        adam  = Adam(lr=0.001)
        model.compile(loss="mse", optimizer=adam)
        return state_input, model

    def create_critic_model(self):
        state_input = Input(shape=self.env.observation_space.shape)
        state_h1 = Dense(24, activation='relu')(state_input)
        state_h2 = Dense(48)(state_h1)

        action_input = Input(shape=(198,))
        action_h1 = Dense(48)(action_input)

        merged    = Add()([state_h2, action_h1])
        merged_h1 = Dense(24, activation='relu')(merged)
        output = Dense(1, activation='relu')(merged_h1)
        model  = Model(input=[state_input,action_input], output=output)

        adam  = Adam(lr=0.001)
        model.compile(loss="mse", optimizer=adam)
        return state_input, action_input, model

    # ========================================================================= #
    #                               Model Training                              #
    # ========================================================================= #

    def remember(self, cur_state, action, reward, new_state, done):
        self.memory.append([cur_state, action, reward, new_state, done])

    def _train_actor(self, samples):
        for sample in samples:
            cur_state, action, reward, new_state, _ = sample
            predicted_action = self.actor_model.predict(cur_state)
            grads = self.sess.run(self.critic_grads, feed_dict={
                self.critic_state_input:  cur_state,
                self.critic_action_input: predicted_action
            })[0]

            self.sess.run(self.optimize, feed_dict={
                self.actor_state_input: cur_state,
                self.actor_critic_grad: grads
            })

    def _train_critic(self, samples):
        for sample in samples:
            cur_state, action, reward, new_state, done = sample
            if not done:
                target_action = self.target_actor_model.predict(new_state)
                future_reward = self.target_critic_model.predict(
                    [new_state, target_action])[0][0]
                reward += self.gamma * future_reward
            self.critic_model.fit([cur_state, np.array(action)], np.array([reward]), verbose=0)

    def train(self):
        batch_size = 32
        if len(self.memory) < batch_size:
            return

        rewards = []
        samples = random.sample(self.memory, batch_size)
        self._train_critic(samples)
        self._train_actor(samples)

    # ========================================================================= #
    #                         Target Model Updating                             #
    # ========================================================================= #

    def _update_actor_target(self):
        actor_model_weights  = self.actor_model.get_weights()
        actor_target_weights = self.target_actor_model.get_weights()

        for i in range(len(actor_target_weights)):
            actor_target_weights[i] = actor_model_weights[i]
        self.target_actor_model.set_weights(actor_target_weights)

    def _update_critic_target(self):
        critic_model_weights  = self.critic_model.get_weights()
        critic_target_weights = self.critic_target_model.get_weights()

        for i in range(len(critic_target_weights)):
            critic_target_weights[i] = critic_model_weights[i]
        self.critic_target_model.set_weights(critic_target_weights)

    def update_target(self):
        self._update_actor_target()
        self._update_critic_target()

    # ========================================================================= #
    #                              Model Predictions                            #
    # ========================================================================= #

    def act(self, cur_state, actions_taken):
        self.epsilon *= self.epsilon_decay
        if np.random.random() < self.epsilon:
                        #Make a random distribution across the action space, to match the format of a prediction
                        #Remove previously taken actions, so there will be no repeats
                        #the highest remaining action will not have been taken before
            action_choices = np.random.uniform(0, 1, self.env.action_space.n) - actions_taken
        else:
            prediction = self.actor_model.predict(cur_state)[0]
                        #normalize prediction so no element is >1, and remove all previously chosen elments
            action_choices = prediction/(abs(prediction[np.argmax(prediction)])+1) - actions_taken*2

        #Choose the highest predicted action (ones that have been taken before will be out of the running
        action = np.argmax(action_choices)

        #Construct the action vector
        action_vec = np.zeros(self.env.action_space.n)
        action_vec[action] = 1

        #Update which actions have been taken
        actions_taken[action] = 1

        return action_vec, actions_taken

def main(args):
    sess = tf.Session()
    K.set_session(sess)
    reward_scale=float(args[0])
    mydir=args[1] #/wrk/jpr6
    env = gym.make("hkl-v0", reward_scale=reward_scale)
    env = env.unwrapped
    actor_critic = ActorCritic(env, sess)

    
    
    num_trials = 1000

    cur_state = env.reset()
    episode = 0
    done = False
    totreward = 0

        #Log data
    rewards = []
    chisqs = []
    zvals = []
    steps = []
    single_eps_chis = []
    hkls = []

    while episode < num_trials:
            #Start an episode

            actions_taken = np.zeros(env.action_space.n)

            while done is False:
                #take an action

                #Choose action
                cur_state = cur_state.reshape((1, env.observation_space.shape[0]))
                action, actions_taken = actor_critic.act(cur_state, actions_taken)
                action = action.reshape((1, env.action_space.n))

                #Take action
                action=action.tolist()
                new_state, reward, done, info = env.step(np.argmax(action))
                totreward += reward

                new_state = new_state.reshape((1, env.observation_space.shape[0]))

                #train
                actor_critic.remember(cur_state, action, reward, new_state, done)
                actor_critic.train()

                cur_state = new_state

                single_eps_chis.append(info.get("chi"))
                hkls.append(info.get("hkl"))

            #reset
            cur_state = env.reset()
            episode +=1
            done = False

            if ((episode % 10) == 0):
                rewards.append(totreward)
                chisqs.append(info.get("chi"))
                zvals.append(info.get("z"))
                steps.append(episode)

            totreward = 0

            if((episode % 10) == 0):
                plt.plot(single_eps_chis)
                plt.xlabel("Measurements Taken")
                plt.ylabel("Chi Squared Value")
                plt.title("Z: " + str(info.get("z")))
                plt.savefig(os.path.join(mydir,'ac-025-chi-in-eps-' + str(episode) + '.png'))
                plt.close()

            single_eps_chis = []

            if((episode % 10) == 0):
                file = open(os.path.join(mydir, "ac-025-hklLog-invchi2" + str(episode) + ".txt"), "w")
                file.write("episode: " + str(episode))
                file.write(str(hkls))
                file.close()

            hkls = []

            if((episode % 10) == 0):

                plt.scatter(steps, rewards)
                plt.xlabel("Episodes")
                plt.ylabel("Reward")
                plt.savefig(os.path.join(mydir,'ac-reward-invchi25.png'))
                plt.close()

                plt.scatter(steps, chisqs)
                plt.xlabel("Episodes")
                plt.ylabel("Final Chi Squared Value")
                plt.savefig(os.path.join(mydir,'ac-chi-invchi25.png'))
                plt.close()

                plt.scatter(steps, zvals)
                plt.xlabel("Episodes")
                plt.ylabel("Z Value")
                plt.savefig(os.path.join(mydir,'ac-z-invchi25.png'))
                plt.close()

if __name__ == "__main__":
    main(sys.argv[1:])

