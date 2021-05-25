from random import random
from scipy.stats import entropy
import numpy as np


class Agent:
    def __init__(self, env, modality_weights):
        self.env = env

        self.n_states = 3
        self.n_modalities = 3

        # Observation counters per state per modality:
        self.counts = np.array(
            [
                [  # mod 0
                    [1, 1, 1],  # state 0
                    [1, 1, 1],  # state 1
                    [1, 1, 1]   # state 2
                ],
                [  # mod 1
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1]
                ],
                [  # mod 2
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1]
                ]
            ]
        )

        self.state_visits = np.array(
            [3, 3, 3]
        )

        self.dog_multiplier = 500


        # Predicted modality values per state:
        self.pred_outcome_given_state_per_modality = np.array(
            [
                [  # mod 0
                    [0.4, 0.3, 0.3],
                    [0.4, 0.3, 0.3],
                    [0.4, 0.3, 0.3]
                ],
                [  # mod 1
                    [0.4, 0.3, 0.3],
                    [0.4, 0.3, 0.3],
                    [0.4, 0.3, 0.3]
                ],
                [  # mod 2
                    [0.4, 0.3, 0.3],
                    [0.4, 0.3, 0.3],
                    [0.4, 0.3, 0.3]
                ]
            ]
        )

        # Preferred modality values:
        self.expected_outcome_per_modality = np.array(
            [
                [0.9, 0.0999, 0.00001],
                [0.9, 0.0999, 0.00001],
                [0.8, 0.1999, 0.00001]
            ]
        )

        # importance of each modality:
        #self.modality_weights = np.array(
        #    [0.334, 0.333, 0.333]
        #)
        self.modality_weights = np.array(modality_weights)

    def act(self):
        action = self.best_policy()
        observation = np.array(self.env.act(action))
        self.update_predicted_outcomes(observation, action)

    def update_predicted_outcomes(self, observation, state):
        observation_weight = 1
        if observation[1] == 2:
            observation_weight = self.dog_multiplier
        # update counts of observations:
        for i in range(self.n_modalities):
            self.counts[i, state, observation[i]] += observation_weight
        # update counts of visited states:
        self.state_visits[state] += observation_weight
        for m in range(self.n_modalities):
            self.pred_outcome_given_state_per_modality[m] = (self.counts[m].T/self.state_visits).T

    def best_policy(self):
        # Calculate qualities of actions:
        qs = [self.softmax(np.array([self.quality(action, modality) for action in range(self.n_states)])) for modality in range(self.n_modalities)]

        # Weigh the qualities of all modalities:
        qualities = np.dot(self.modality_weights, np.array(qs))

        # Draw action from the qualities:
        take_action = np.random.choice(self.n_states, p=qualities)

        # Return best action:
        return take_action

    def quality(self, act, mod):
        # predicted outcome of modalities given the outcome state when applying action:
        pred_outcome = self.pred_outcome_given_state_per_modality[mod][act]
        # Get the extrinsic and epistemic values for this policy:
        ext_val = self.extrinsic(pred_outcome, mod)
        epist_val = self.epistemic(pred_outcome, act, mod)
        return ext_val + epist_val

    def extrinsic(self, pred_outcome, mod):
        # calculate extrinsic value of the predicted outcome outcome given internal expected outcome:
        ext_val = np.sum(pred_outcome * np.log(self.expected_outcome_per_modality[mod]))
        return ext_val

    def epistemic(self, pred_outcome, act, mod):
        pred_state = np.zeros(3)
        pred_state[act] = 1

        posterior = np.multiply(pred_state, self.pred_outcome_given_state_per_modality[mod].T)

        post_sum = np.sum(posterior, axis=1)

        posterior = posterior / post_sum[:, None]

        # Calculate the expected entropy
        pred = pred_state * np.ones(posterior.shape)
        exp_ent = np.sum(pred_outcome * entropy(qk=pred, pk=posterior, axis=1))
        return exp_ent

    def softmax(self, x):
        f_x = np.exp(x) / np.sum(np.exp(x))
        return f_x
