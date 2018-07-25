"""
https://towardsdatascience.com/reinforcement-learning-w-keras-openai-actor-critic-models-f084612cfd69
solving pendulum using actor-critic model
"""

import gym
import numpy as np 
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers.merge import Add, Multiply
from keras.optimizers import Adam
import keras.backend as K

import tensorflow as tf

import random
from collections import deque

# determines how to assign values to each state, i.e. takes the state
# and action (two-input model) and determines the corresponding value
class ActorCritic:
	def __init__(self, env, sess):
		self.env  = env
		self.sess = sess

		self.learning_rate = 0.001
		self.epsilon = 1.0
		self.epsilon_decay = .995
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
 #               print("act grads", self.actor_grads)
		grads = zip(self.actor_grads, actor_model_weights)
		self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)
#                print("opt", self.optimize)
		# ===================================================================== #
		#                              Critic Model                             #
		# ===================================================================== #

		self.critic_state_input, self.critic_action_input, \
			self.critic_model = self.create_critic_model()
		_, _, self.target_critic_model = self.create_critic_model()
                print("--------------",self.critic_model.output, self.critic_action_input)

		self.critic_grads = tf.gradients(self.critic_model.output, 
			self.critic_action_input) # where we calcaulte de/dC for feeding above

 #               print("critic grads", self.critic_grads)
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
#		action_h1 = Dense(24, activation='relu')(action_input)
		action_h1 = Dense(48)(action_input)

		merged    = Add()([state_h2, action_h1])
		merged_h1 = Dense(24, activation='relu')(merged)
 		output = Dense(1, activation='relu')(merged_h1)
		model  = Model(input=[state_input,action_input], output=output)

#		model = Sequential()
#		model.add(Dense(24, input_shape=self.env.observation_space.shape))
#		model.add(Activation('relu'))
#		model.add(Dense(48)
#		model.add(Activation('relu'))
#		model.add(Dense(1))
#		model.add(Activation('relu'))

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
#                        print(cur_state, "<- state", predicted_action, self.critic_state_input, self.critic_action_input)
			grads = self.sess.run(self.critic_grads, feed_dict={
				self.critic_state_input:  cur_state,
				self.critic_action_input: predicted_action
			})[0]

#                        print(grads)
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
#                        print("past if", cur_state, action, np.array([action]), reward)
			self.critic_model.fit([cur_state, action], np.array([reward]), verbose=2)
		
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

	def act(self, cur_state):
		self.epsilon *= self.epsilon_decay
		if np.random.random() < self.epsilon:
                        action = self.env.action_space.sample()
                        actionArr = np.zeros(self.env.action_space.n)
                        actionArr[action] = 1
                        return actionArr
		return self.actor_model.predict(cur_state)

def main():
	sess = tf.Session()
	K.set_session(sess)
	env = gym.make("hkl-v0")
	env = env.unwrapped
	actor_critic = ActorCritic(env, sess)

	num_trials = 500
	trial_len  = 500

	cur_state = env.reset()
#       action = env.action_space.sample()
#        ep = 0
        done = False
        totreward = 0
#	chisq = 0
#	z = 0
#	hkl = 0
#        while ep < num_trials:
	for ep in range(num_trials):
	    print("Episode #" + str(ep))

#	    file = open("ac_results" + str(ep) + ".txt", "w") 
#	    file.write("HKL\t\tReward\ttotReward\tchisq\tz approx")

            while done is False:

                cur_state = cur_state.reshape((1, env.observation_space.shape[0]))
                action = actor_critic.act(cur_state)
                action = action.reshape((1, env.action_space.n))
#                print(action)

                new_state, reward, done, _ = env.step(np.argmax(action))
                totreward += reward
#		file.write(str(hkl) + "\t\t" + str(reward) + "\t" + str(totreward) + "\t" + str(chisq) + "\t" + str(z))
#                print(action, reward)
                new_state = new_state.reshape((1, env.observation_space.shape[0]))

                actor_critic.remember(cur_state, action, reward, new_state, done)
                actor_critic.train()

                cur_state = new_state

#	    file.close()
            print(totreward)
            totreward = 0
#	    env.episodeNum += 1
	    env.epStep()
            env.reset()
#            ep +=1
            done = False


if __name__ == "__main__":
	main()

