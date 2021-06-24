import numpy as np


class Environment:
    def __init__(self, dog_day, dog_modalities):

        # Means of observations per modality per state
        self.probabilities = np.array(
            [
                [  # mod 0
                    [0.6, 0.4, 0],  # state 0
                    [0.9, 0.1, 0],  # state 1
                    [0.3, 0.7, 0]   # state 2
                ],
                [  # mod 1
                    [0.6, 0.4, 0],
                    [0.9, 0.1, 0],
                    [0.3, 0.7, 0]
                ],
                [  # mod 2
                    [0.6, 0.4, 0],
                    [0.9, 0.1, 0],
                    [0.3, 0.7, 0]
                ]
            ]
        )

        # Number of observations until dog appears at state 1
        self.dog_day = dog_day

        # day of actual dog encounter
        self.dog_encounter_day = -1

        # number of modalities affected by the dog encounter:
        self.dog_modalities = dog_modalities

        # Boolean to keep track of dog encounter
        self.has_had_dog_day = False

        # Action that will lead to dog encounter
        self.dog_street = 1

        # Counter of observations and actions by the agent:
        self.days = 0

        # Log of agent moves:
        self.log = []

    def act(self, action):
        """Generates an observation based on an action"""
        # Generate observations based on the action:
        observation = np.array([np.random.choice(3, p=self.probabilities[mod][action]) for mod in range(3)])
        # Check for dog encounter:
        if self.days >= self.dog_day and action == self.dog_street and not self.has_had_dog_day:
            # Overwrite outcome with the dog outcome (outcome 2) for each affected modality:
            for i in range(self.dog_modalities):
                observation[i] = 2
            self.has_had_dog_day = True
            self.dog_encounter_day = self.days
        # Keep log of the day and action:
        self.days += 1
        self.log.append(action)
        # Return observations to the agent:
        return observation

    def report(self):
        """Returns the action log"""
        return {"log": self.log, "dog_encounter": self.dog_encounter_day}
