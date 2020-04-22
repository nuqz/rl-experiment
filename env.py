import sys
import curses

import gym
import numpy as np

STAY = np.zeros(2, dtype=np.int)
NORTH = np.array([0, -1])
EAST = np.array([1, 0])
SOUTH = np.array([0, 1])
WEST = np.array([-1, 0])

ACTIONS = ['stay', 'north', 'east', 'south', 'west']
ACTIONS_MAP = dict(enumerate(ACTIONS))

MAP_SIZE = np.array([10, 10])
MAX_STEPS = 100


def distance(p1, p2):
    return np.sqrt(np.sum((p2 - p1) ** 2))


ZERO = np.array([0, 0])
MAX_DISTANCE = distance(ZERO, MAP_SIZE-1)


class State:
    def __init__(self, map_size=MAP_SIZE):
        self.steps = 0

        width, height = self.map_size = map_size
        # self.map_size = map_size

        self.target_position = np.random.randint(0, width, 2)
        self.player_position = np.random.randint(0, width, 2)

        self.map = self._generate_map(map_size)

    def _generate_map(self, size=MAP_SIZE):
        gmap = np.zeros(size)
        return gmap

    def move_player(self, direction_name):
        direction = getattr(sys.modules[__name__], direction_name)
        self.player_position += direction


class SimpleEnv(gym.Env):

    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Discrete(len(ACTIONS))
        self.last_action = self.last_reward = self.last_distance = 0

    def step(self, action):
        if action not in ACTIONS_MAP:
            raise AssertionError('invalid action')

        action_name = ACTIONS_MAP[action]

        self.last_action = action
        self.state.move_player(action_name.upper())
        self.state.steps += 1

        reward = 0.
        if action_name != 'stay':
            new_distance = distance(
                self.state.player_position, self.state.target_position)
            # delta = (MAX_DISTANCE - new_distance) / MAX_DISTANCE
            # if self.last_distance <= new_distance:
            #     delta = -delta
            reward += (self.last_distance - new_distance) / 2
            if reward < 0:
                reward *= 1.5
            self.last_distance = new_distance

        # too_long = self.state.steps > MAX_STEPS
        # if too_long:
        #     reward += -1.
        too_long = False

        on_target = np.array_equal(
            self.state.player_position, self.state.target_position)
        if on_target:
            reward += 1.

        mw, mh = self.state.map_size
        px, py = self.state.player_position
        out_of_map = px < 0 or px >= mw or py < 0 or py >= mh
        if out_of_map:
            reward += -1.

        done = out_of_map or on_target or too_long

        reward /= 2.
        self.last_reward = reward

        return self.state, reward, done, {}

    def reset(self):
        self.last_action = self.last_reward = 0
        self.state = State()
        return self.state

    def render(self, scr, mode='human'):
        scr.clear()
        scr.addstr(
            'Step: #{step:03d} / Last reward: {reward:01.2f} / Last action: {action}'.format(
                step=self.state.steps,
                reward=self.last_reward,
                action=ACTIONS_MAP[self.last_action]))

        width, height = self.state.map_size
        width += 2
        height += 2
        xlim, ylim = width - 1, height - 1

        gmap = curses.newwin(height, width, 1, 1)

        for x in range(0, xlim):
            gmap.addch(0, x, ord('-'))
            gmap.addch(ylim, x, ord('-'))

        for y in range(0, ylim):
            gmap.addch(y, 0, ord('|'))
            gmap.addch(y, xlim,  ord('|'))

        for x in range(0, xlim):
            for y in range(0, ylim):
                if x != 0 and x != xlim and y != 0 and y != ylim:
                    gmap.addch(y, x, ord('.'))

        px, py = self.state.player_position + 1
        if px >= 0 and py >= 0 and px < xlim and py < ylim:
            gmap.addch(py, px, ord('*'))

        tx, ty = self.state.target_position + 1
        ch = ord('O')
        if tx == px and ty == py:
            ch = ord('X')
        gmap.addch(ty, tx, ch)

        scr.addstr(13, 0, 'Player @ [{x}, {y}]'.format(x=px-1, y=py-1))
        scr.refresh()
        gmap.refresh()

    def close(self):
        pass


gym.register('SimpleEnv-v0', entry_point=SimpleEnv)
