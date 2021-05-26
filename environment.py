import numpy as np


class Environment:
    def __init__(self, dog_day):

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

        # Boolean to keep track of dog encounter
        self.has_had_dog_day = False

        # Action that will lead to dog encounter
        self.dog_street = 1

        # Counter of observations and actions by the agent:
        self.days = 0

        # Log of agent moves:
        self.log = []

    def act(self, action):
        observation = np.array([np.random.choice(3, p=self.probabilities[mod][action]) for mod in range(3)])
        if self.days >= self.dog_day and action == self.dog_street and not self.has_had_dog_day:
            observation = [2, 2, 2]
            self.has_had_dog_day = True
            self.dog_encounter_day = self.days
            print("Oh no, a dog! scary!")
            print("action taken:", action)
        self.days += 1
        self.log.append(action)
        return observation

    def report(self):
        return {"log": self.log, "dog_encounter": self.dog_encounter_day}
