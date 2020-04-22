import argparse
import curses
import random
import time

import gym
import torch

import env

WEIGHTS_PATH = 'weights.pt'
MAX_EPISODES = 10 ** 5
VISUAL_INTERVAL = 10 ** 2

args_parser = argparse.ArgumentParser()
args_parser.add_argument('--load', dest='load', type=bool)
args = args_parser.parse_args()


e = gym.make('SimpleEnv-v0')


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(100, 16)
        # self.l2 = torch.nn.Linear(32, 16)
        self.l3 = torch.nn.Linear(16, e.action_space.n)

    def forward(self, state):
        x = torch.Tensor(state.map)
        px, py = state.player_position
        tx, ty = state.target_position

        x[px, py] = -1
        x[tx, ty] = 1

        x = x.view(-1)
        x = torch.nn.functional.relu(self.l1(x))
        # x = torch.nn.functional.relu(self.l2(x))
        x = torch.nn.functional.relu(self.l3(x))

        return x


net = Net()
print(net)
if args.load:
    print('Loading weights')
    net.load_state_dict(torch.load(WEIGHTS_PATH))
    net.eval()


# Hypers
alpha = 3e-2    # learning rate
gamma = 7e-1    # discount
epsilon = 1.5e-1  # greedyness

optimizer = torch.optim.SGD(net.parameters(), lr=alpha)


def main(scr):
    epochs = 0
    wins = 0
    for t in range(1, MAX_EPISODES):
        visualize_episode = t != 0 and t % VISUAL_INTERVAL == 0

        state = e.reset()
        penalties, reward, done = 0, 0, False

        while not done:
            optimizer.zero_grad()

            if random.uniform(0, 1) < epsilon:
                action = e.action_space.sample()
            else:
                action = torch.argmax(net(state)).item()

            next_state, reward, done, info = e.step(action)

            if not done:
                old_value = net(state)
                next_max = torch.max(net(next_state))

                cost = (old_value - (reward + gamma * next_max)) ** 2
                cost.mean().backward()
                optimizer.step()

                if reward == -1:
                    penalties += 1

            state = next_state
            epochs += 1

            if visualize_episode:
                e.render(scr)
                scr.addstr(14, 0, '=' * 40)
                scr.addstr(15, 0,
                           'Epochs: {epoch} / Episodes: {episode} / Wins: {wins} ({win_rate:2.2f}%)'.format(
                               epoch=epochs, episode=t, wins=wins, win_rate=(wins/VISUAL_INTERVAL)*100))

                # p = curses.newpad(200, 500)
                # p.addstr(0, 0, '{}'.format(list(net.parameters())[3]))
                # p.refresh(0, 0, 4, 14, 12, curses.COLS-1)

                scr.refresh()
                time.sleep(0.25)

        if reward > 0:
            wins += 1

        if visualize_episode:
            if reward <= 0:
                scr.addstr(3, 14, '---- LOSE ----')
            else:
                scr.addstr(3, 14, '++++ WIN +++++')
            scr.refresh()
            wins = 0

    torch.save(net.state_dict(), WEIGHTS_PATH)


if __name__ == '__main__':
    curses.wrapper(main)
