from random import random
from scipy.stats import entropy
import numpy as np


class Agent:
    def __init__(self, environment):
        self.environment = environment
        self.observation = None
        self.n_modalities = 3
        # Coordinates of current state:
        self.state = [0, 0]
        # World limits:
        self.world_height = self.environment.world_height
        self.world_width = self.environment.world_width
        # Number of states:
        self.n_states = self.world_height * self.world_width
        #
        self.states = np.zeros(
            (
                self.n_modalities,
                self.world_height,
                self.world_width
            )
        )
        # print(self.states)

        # Predicted food given state: (starts Uniform)
        self.pred_food_given_state = np.array(
            [[1/self.n_states for i in range(self.world_width)] for j in range(self.world_height)]
        )
        print(self.pred_food_given_state)
        # Expected food outcome:
        self.expected_food_outcome = None  # How?

    def get_pred_state(self, move):
        # Apply move:
        predicted_state = self.state + move
        # Keep states within world bounds:
        predicted_state[0] = min(max(0, predicted_state[0]), self.world_height)
        predicted_state[1] = min(max(0, predicted_state[1]), self.world_width)
        # Return predicted state:
        return predicted_state

    def interact(self, action):
        self.state, self.observation = self.environment.act(action)
        print('observation: ', self.observation)
        print('location: ', self.state)

    def test_aa(self):
        # Define the distribution over future states P(s_t | pi)
        self.pred_state = [0.5, 0.1, 0.4]

        # Define the distribution over observation given a state P(o_t | s_t)
        self.pred_outcome_given_state = [
            [0.55, 0.15, 0.3],
            [0.5, 0.49, 0.01],
            [0.3, 0.6, 0.1]
        ]

        # Define the exepected out P(o | m)
        self.expected_outcome = [0.49, 0.01, 0.5]

        # There are two ways to determine the quality of a policy
        self.q1 = self.quality(self.pred_state, self.pred_outcome_given_state, self.expected_outcome)
        self.q2 = self.quality_alt(self.pred_state, self.pred_outcome_given_state, self.expected_outcome)

        # Show that these are the same
        print(self.q1, self.q2, self.q1 == self.q2)

    def quality(self, pred_state, pred_outcome_given_state, expected_outcome):
        # Change lists to arrays
        pred_state = np.array(pred_state)
        pred_outcome_given_state = np.array(pred_outcome_given_state)
        expected_outcome = np.array(expected_outcome)

        # Determine predicted outcome
        pred_outcome = np.dot(pred_state, pred_outcome_given_state)

        # Determine the extrinsic and epistemic value
        extrinsic = np.sum(pred_outcome * np.log(expected_outcome))

        epistemic = self.epistemic_value(pred_state, pred_outcome_given_state, pred_outcome)

        return extrinsic + epistemic

    def epistemic_value(self, pred_state, likelihoods, pred_outcome):
        # Calculate the posterior for each possible observation
        posterior = np.multiply(pred_state, likelihoods.T)
        post_sum = np.sum(posterior, axis=1)
        posterior = posterior / post_sum[:, None]

        # Calculate the expected entropy
        pred = pred_state * np.ones(posterior.shape)
        exp_ent = np.sum(pred_outcome * entropy(qk=pred, pk=posterior, axis=1))
        return exp_ent

    def quality_alt(self, pred_state, pred_outcome_given_state, expected_outcome):
        # Change lists to arrays
        pred_state = np.array(pred_state)
        pred_outcome_given_state = np.array(pred_outcome_given_state)
        expected_outcome = np.array(expected_outcome)

        # Determine predicted outcome
        pred_outcome = np.dot(pred_state, pred_outcome_given_state)

        # Calculate predicted uncertainty as the expectation
        # of the entropy of the outcome, weighted by the
        # probability of that outcome
        pred_ent = np.sum(pred_state * entropy(pred_outcome_given_state, axis=1))

        # Calculate predicted divergence as the Kullback-Leibler
        # divergence between the predicted outcome and the expected outcome
        pred_div = entropy(pk=pred_outcome, qk=expected_outcome)

        # Return the sum of the negatives of these two
        return -pred_ent - pred_div








