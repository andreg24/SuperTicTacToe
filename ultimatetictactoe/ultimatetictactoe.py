from __future__ import annotations

import os

import gymnasium
import numpy as np
import pygame
from gymnasium import spaces
from gymnasium.utils import EzPickle
import torch

from pettingzoo import AECEnv
from pettingzoo.utils import AgentSelector, wrappers

from utils.board import UltimateTicTacToeBoard, Status, BoardTransformation, BoardRotation, BoardReflection
from utils.board_utils import relative_to_absolute
from utils.render_utils import get_image, get_font, render_tui

DEFAULT_BOARD_SIZE = 500

BOARD_IMG = "board.jpg"
CROSS_IMG = "cross.jpg"
CIRCLE_IMG = "circle.jpg"


def get_symbol(value):
    if value == 0:
        return None
    elif value == 1:
        return "cross"
    elif value == 2:
        return "circle"
    else:
        # tie
        return None


def env(**kwargs):
    env = raw_env(**kwargs)
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class raw_env(AECEnv, EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array", "tui"],
        "name": "supertictactoe_v0",
        "is_parallelizable": False,
        "render_fps": 2,
    }

    def __init__(self, render_mode: str | None = None, num_envs: int = 1, board_size: int | None = None):
        EzPickle.__init__(
            self,
            render_mode,
            board_size if board_size is not None else DEFAULT_BOARD_SIZE,
        )
        super().__init__()

        self.num_envs = num_envs
        self.board = torch.zeros((self.num_envs, 9, 9))

        # ---agents---
        self.agents = ["player_1", "player_2"]
        self.possible_agents = self.agents[:]  # PettingZoo requirement
        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        # ---spaces---
        self.action_spaces = {name: spaces.Discrete(81) for name in self.agents}
        self.observation_spaces = {
            a: spaces.Dict(
                {
                    "observation": spaces.Box(
                        low=0, high=1, shape=(self.num_envs, 2, 9, 9), dtype=np.int8
                    ),
                    "action_mask": spaces.Box(
                        low=0, high=1, shape=(81,), dtype=np.int8
                    ),
                    "turn": spaces.Discrete(2),
                }
            )
            for a in self.agents
        }

        # ---returns---
        self.rewards = {a: 0.0 for a in self.agents}  # not accessed by last()
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}

        # ---rendering--
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.board_size = board_size if board_size is not None else DEFAULT_BOARD_SIZE
        self.screen = None

        # other render related things TO DO

        if self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # TODO in the future
        # history
        self.last_action = None
        self.board_history = []  # TODO
        self.actions_history = []  # TODO
        self.num_moves = 0  # TODO

        # stats
        self.status = None

    def observe(self, agent):
        # last() calls this function
        board_vals = np.array(self.board.cells).reshape(9, 9)

        observation = np.empty((2, 9, 9), dtype=np.int8)

        # previous version
        # cur_player = self.possible_agents.index(agent)
        # opp_player = (cur_player + 1) % 2
        # observation[0, :, :] = np.equal(board_vals, cur_player + 1)
        # observation[1, :, :] = np.equal(board_vals, opp_player + 1)
        observation[0, :, :] = np.equal(board_vals, 1)
        observation[1, :, :] = np.equal(board_vals, 2)

        return {
            "observation": observation,
            "action_mask": self._get_mask(agent),
            "turn": self.agents.index(self.agent_selection),
        }

    def _get_mask(self, agent):
        # Per the documentation, the mask of any agent other than the
        # currently selected one is all zeros.
        mask = np.zeros(81, dtype=np.int8)

        if agent == self.agent_selection:
            mask[self.board.legal_moves()] = 1
        return mask

    def action_mask(self, agent):
        return self._get_mask(agent)

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    # action in this case is a value from 0 to 8 indicating position to move on tictactoe board
    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            return self._was_dead_step(action)

        self.board.play_turn(self.agents.index(self.agent_selection), int(action))

        status = self.board.game_status()
        if status != Status.GAME_NOT_OVER:
            if status == Status.TIE:
                difference = self.board.super_cells.count(
                    1
                ) - self.board.super_cells.count(2)
                self.rewards[self.agents[0]] = difference * 0.1
                self.rewards[self.agents[1]] = -difference * 0.1
            else:
                winner = status.value - 1  # either TTT_PLAYER1_WIN or TTT_PLAYER2_WIN
                loser = winner ^ 1  # 0 -> 1; 1 -> 0
                self.rewards[self.agents[winner]] = 1
                self.rewards[self.agents[loser]] = -1

            # once either play wins or there is a draw, game over, both players are done
            self.terminations = {a: True for a in self.agents}
            self._accumulate_rewards()

        self.agent_selection = self._agent_selector.next()
        self.last_action = action

        if self.render_mode in ["human", "tui"]:
            self.render()

    def reset(self, seed=None, options={}):
        if options and "board" in options and options["board"]:
            self.board = options["board"].copy()
        else:
            self.board = UltimateTicTacToeBoard()

        self.agents = self.possible_agents[:]
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.reset()
        if options and "next_player" in options and options["next_player"] != 1:
            self.agent_selection = self._agent_selector.next()
            self.board.current_player = self.agents.index(self.agent_selection)

        self.rewards = {name: 0 for name in self.agents}
        self._cumulative_rewards = {name: 0 for name in self.agents}
        self.terminations = {name: False for name in self.agents}
        self.truncations = {name: False for name in self.agents}
        self.infos = {name: {} for name in self.agents}

        self.board_history = []
        self.actions_history = []
        self.num_moves = 0
        self.last_action = None

        if self.render_mode is not None and self.screen is None:
            pygame.init()

        if self.render_mode == "human":
            self.screen = pygame.display.set_mode((self.board_size, self.board_size))
            pygame.display.set_caption("Super Tic-Tac-Toe")
        elif self.render_mode == "rgb_array":
            self.screen = pygame.Surface((self.board_size, self.board_size))

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return
        # elif self.render_mode in {"human", "rgb_array"}:
        #     return self._render_gui()

        if self.render_mode == "human":
          # --- Setup ---
          tile_size = self.board_size // 12
          tile_size_big = self.board_size // 4

          # --- Draw board background ---
          board_img = get_image(os.path.join("img", "board.jpg"))
          board_img = pygame.transform.scale(board_img, (self.board_size, self.board_size))
          self.screen.blit(board_img, (0, 0))

          board_state = list(map(get_symbol, self.board.cells))
          super_board_state = list(map(get_symbol, self.board.super_cells))

          # --- Draw small marks (9x9) ---
          mark_pos = 0
          for x in range(9):
              for y in range(9):
                  mark = board_state[mark_pos]
                  mark_pos += 1

                  if mark is None:
                      continue

                  mark_img = get_image(os.path.join("img", mark + ".png"))
                  if self.last_action is not None and self.last_action == mark_pos-1:
                      pass
                  else:
                      mark_img.set_alpha(100)

                  mark_img = pygame.transform.scale(mark_img, (tile_size, tile_size))

                  self.screen.blit(
                      mark_img,
                      (
                          (self.board_size / 8.8) * y + (self.board_size / 190),
                          (self.board_size / 8.8) * x + (self.board_size / 190),
                      ),
                  )
          # big symbols
          mark_pos = 0
          for x in range(3):
              for y in range(3):
                  mark = super_board_state[mark_pos]
                  mark_pos += 1

                  if mark is None:
                      continue

                  mark_img = get_image(os.path.join("img", mark + ".png"))
                  mark_img = pygame.transform.scale(mark_img, (tile_size_big, tile_size_big))


                  self.screen.blit(
                      mark_img,
                      (
                          (self.board_size / 2.9) * y + (self.board_size / 30),
                          (self.board_size / 2.9) * x + (self.board_size / 30),
                      ),
                  )
          pygame.display.update()
          self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
          observation = np.array(pygame.surfarray.pixels3d(self.screen))
          return np.transpose(observation, axes=(1, 0, 2))
        elif self.render_mode == "tui":
          render_tui(self.board.cells)

    def apply_transformation(self, transformation: BoardTransformation):
        self.board.apply_transformation(transformation)
        if self.last_action != -1:
            self.last_action = transformation.pos_transform(self.last_action, 9)


    def save_snapshot(self, path: str = "game.png", dir: str = "img"):
        pygame.image.save(self.screen, dir + path)
