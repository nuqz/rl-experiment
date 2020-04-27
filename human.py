#!/usr/bin/env python

import curses
import os

import gym

import env

ENV = gym.make('SimpleEnv-v0')
KEY_MAP = {
    258: 3,
    259: 1,
    260: 4,
    261: 2,
}

def main(scr):
    _, done = ENV.reset(), False
    scr.clear()
    while not done:
        ENV.render(scr)
        ch = scr.getch()
        if ch in KEY_MAP:
            action = KEY_MAP[ch]
            state, reward, done, info = ENV.step(action)
        else:
            break
    ENV.render(scr)
    scr.getch()

if __name__ == '__main__':
    curses.wrapper(main)
