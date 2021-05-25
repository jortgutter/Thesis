import numpy as np

# TODO: two alternatives of world representation. Sketch them, and set up simply in python
# TODO: schematic drawing of how agent works, what distributions are used?
# TODO: als niet lukt om te beginnen, schrijf op wat je dan wel doet!

class Environment:
    def __init__(self):
        # TODO: Add doghouse? location around dog that has a certain observational value for some modality so agent can associate location with dog
        self.world_items = {
            'X': -1,  # Wall
            '.': 0,   # Path
        }
        self.world = None
        self.world_width = 0
        self.world_height = 0

        # Positions:
        self.food = None
        self.dog = None
        self.agent = None
        self.agent_start = None

        # flag for affect-biased attention trigger:
        self.scare = False

        # Hyper parameters:
        self.l = 0.1  # lambda, scales translation distribution of food distance
        self.temp_mean = .5
        self.sight_mean = .5
        self.temp_std = .05
        self.sight_std = .05

        # move vectors:
        self.moves = {
            'left':  [0, -1],
            'up':    [-1, 0],
            'right': [0, 1],
            'down':  [1, 0],
            'stay':  [0, 0]
        }

        # Log of all the agents movements:
        self.path_log = []

    def translate(self, c):
        return self.world_items[c]

    def get_neighbours(self, state):
        neighbours = [
            [state[0] + 1, state[1]],
            [state[0] - 1, state[1]],
            [state[0], state[1] + 1],
            [state[0], state[1] - 1]
        ]
        return neighbours

    def display_world(self, world):
        for i in range(len(world)):
            for j in range(len(world[0])):
                val = world[i][j]
                if 0 <= val < 10:
                    print(' ', end='')
                if val == -1:
                    val = '##'
                print(val, ' ', sep='', end='')
            print()

    def dist_to_food(self, world, food):
        queue = [food]
        while len(queue) >= 1:
            state = queue.pop()

            new_dist = world[state[0]][state[1]] + 1
            neighbours = self.get_neighbours(state)
            for n in neighbours:
                if 0 <= n[0] < len(world) and 0 <= n[1] < len(world[0]):
                    if (world[n[0]][n[1]] > new_dist or world[n[0]][n[1]] == 0) and n != food:
                        world[n[0]][n[1]] = new_dist
                        queue.append(n)
        return world

    def load_world(self, filename):
        f = open(filename, "r")
        lines = f.read().split('\n')

        # Translate world states from file:
        world = [[self.translate(val) for val in row] for row in lines[3:]]
        self.world_width = len(world[0])
        self.world_height = len(world)

        # Read agent, food and dog locations:
        self.agent_start = [int(x) for x in lines[0].split(' ')]    # Agents starting location
        self.agent = self.agent_start                               # Set agents location to the start
        self.food = [int(x) for x in lines[1].split(' ')]           # Food location
        self.dog = [int(x) for x in lines[2].split(' ')]            # dog location

        # TODO: perhaps determine location of dog later, when preferred path of agent is known.

        # Translate path states to distances from food:
        self.world = self.dist_to_food(world, self.food)
        self.display_world(world)

    def observe(self, row, col):
        dist = self.world[row][col]
        smell = np.e**(-self.l*dist)
        # TODO: add slight variation to smell

        temp = np.random.normal(self.temp_mean, self.temp_std)
        sight = np.random.normal(self.sight_mean, self.sight_std)
        # Handle unlikely event of dog:
        if self.scare and self.agent == self.dog:
            temp = 0
            sight = 0
            smell = 0
            self.scare = False

        return self.agent, [smell, temp, sight]

    def act(self, move):
        # TODO: make move functionality maybe slightly non-deterministic?
        # Perform move:
        new_pos = np.add(self.agent, self.moves[move])
        new_pos[0] = max(0, new_pos[0])
        new_pos[0] = min(self.world_height - 1, new_pos[0])
        new_pos[1] = max(0, new_pos[1])
        new_pos[1] = min(self.world_width - 1, new_pos[1])
        # If new position is a valid world pos:
        if self.world[new_pos[0]][new_pos[1]] >= 0:
            self.agent = new_pos

        # Log agents position:
        print('new agent position: ', self.agent)
        self.path_log.append(self.agent)

        # Return observation of new state (and position of the agent):
        return self.observe(self.agent[0], self.agent[1])

    def enable_dog(self):
        self.scare = True

    def plot_log(self):
        # TODO: plot the current agent history in some way
        pass

    def reset(self):
        """
        Resets the agent to its starting position, clears log
        """
        self.agent = self.agent_start
        self.path_log = []
