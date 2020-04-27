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
ACTIONS_RENDER_MAP = {
    0: ord('*'),
    1: ord('^'),
    2: ord('>'),
    3: ord('v'),
    4: ord('<'),
}

MAP_SIZE = np.array([10, 10])
MAX_STEPS = 50


def distance(p1, p2):
    return np.sqrt(np.sum((p2 - p1) ** 2))


ZERO = np.array([0, 0])
MAX_DISTANCE = distance(ZERO, MAP_SIZE-1)


class State:
    def __init__(self, map_size=MAP_SIZE):
        self.steps = self.last_action = self.last_reward = 0
        width, height = self.map_size = map_size

        self.map = self._generate_map()

        self._map_win_width, self._map_win_height = self.map_size + 2
        self._map_win_xlim, self._map_win_ylim = self.map_size + 1

        inner_width = self._map_win_width - 2
        self._map_win_horizontal_border = '+' + ('-' * inner_width) + '+'
        self._map_win_horizontal_filler = '|' + ('.' * inner_width) + '|'

        lowest_dim = width if width < height else height
        self.target_position = np.random.randint(0, lowest_dim, 2)
        self.player_position = np.random.randint(0, lowest_dim, 2)
        self._update_positions()

    def _update_positions(self):
        self._tx, self._ty = self.target_position + 1
        self._px, self._py = self.player_position + 1

    def _generate_map(self):
        gmap = np.zeros(self.map_size)
        return gmap

    def move_player(self, direction_name):
        direction = getattr(sys.modules[__name__], direction_name)
        self.player_position += direction

    def is_player_on_target(self):
        return np.array_equal(self.player_position, self.target_position)

    def render(self):
        self._update_positions()

        map_win = self._render_map()
        self._render_player(map_win)
        self._render_target(map_win)

        return map_win

    def _render_map(self):
        map_win = curses.newwin(self._map_win_height,
                                self._map_win_width+1, 1, 0)
        for y in range(0, self._map_win_height):
            s = self._map_win_horizontal_filler
            if y == 0 or y == self._map_win_ylim:
                s = self._map_win_horizontal_border
            map_win.addstr(y, 0, s)

        return map_win

    def _is_player_within_window(self):
        return (self._px >= 0 and self._py >= 0 and
                self._px <= self._map_win_xlim and self._py <= self._map_win_ylim)

    def _render_player(self, map_win):
        if self._is_player_within_window():
            map_win.addch(self._py, self._px,
                          ACTIONS_RENDER_MAP[self.last_action])

    def _render_target(self, map_win):
        ch = ord('O')
        if self.is_player_on_target():
            ch = ord('X')
        map_win.addch(self._ty, self._tx, ch)


class SimpleEnv(gym.Env):

    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Discrete(len(ACTIONS))
        self.last_distance = 0

    def step(self, action):
        if action not in ACTIONS_MAP:
            raise AssertionError('invalid action')

        self.state.steps += 1
        self.state.last_action = action

        action_name = ACTIONS_MAP[action]
        self.state.move_player(action_name.upper())

        reward = 0.
        if action_name != 'stay':
            new_distance = distance(
                self.state.player_position, self.state.target_position)
            delta = self.last_distance - new_distance
            if delta < 0 and np.abs(delta) < 1:
                reward -= 0.25
            elif delta < 0:
                reward -= 0.5
            elif delta > 0 and delta < 1:
                reward += 0.25
            else:
                reward += 0.5
            self.last_distance = new_distance

        too_long = self.state.steps > MAX_STEPS
        if too_long:
            reward -= 1.
        # too_long = False

        on_target = self.state.is_player_on_target()
        if on_target:
            reward += 0.5

        mw, mh = self.state.map_size
        px, py = self.state.player_position
        out_of_map = px < 0 or px >= mw or py < 0 or py >= mh
        if out_of_map:
            reward -= 0.5

        done = out_of_map or on_target or too_long

        self.state.last_reward = reward

        return self.state, reward, done, {}

    def reset(self):
        self.state = State()
        self.last_distance = distance(
            self.state.player_position, self.state.target_position)
        return self.state

    def render(self, scr, mode='human'):
        scr.clear()
        scr.addstr(
            'Step: #{step:03d} / Last reward: {reward:+01.2f} / Last distance: {distance:01.2f}'.format(
                step=self.state.steps,
                reward=self.state.last_reward,
                distance=self.last_distance))
        scr.refresh()

        map_win = self.state.render()
        map_win.refresh()

    def close(self):
        pass


gym.register('SimpleEnv-v0', entry_point=SimpleEnv)
