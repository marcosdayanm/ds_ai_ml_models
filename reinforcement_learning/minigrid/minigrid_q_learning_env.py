from __future__ import annotations

import numpy as np
import pandas as pd


from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Wall
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.actions import Actions
import matplotlib.pyplot as plt


import random
import time
import pickle

class QLearningEnv(MiniGridEnv):
    def __init__(
        self,
        size=10,
        episodes=1000,
        max_steps: int | None = None,
        # Diccionario Q-table: key=(x,y,dir), value=np.array con valores Q por acción
        Q: dict = {},     
        **kwargs,
    ):
        self.size = size
        self.key_positions = []
        self.lava_positions = []

        self.Q = Q
        self.history = []

        # ésto está configurado para un grid de tamaño 10x10
        self.alpha = 0.2   # tasa de aprendizaje, entre más alta más brisco el aprendizaje
        self.gamma = 0.95  # factor de descuento, entre más chico más valora recompensas inmediatas
        self.epsilon = 0.9  # exploración inicial, entre más bajo, menos exploración
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01   # entre más bajo menos exploración
        self.episodes = episodes

        self.agent_pos=(1,1)

        mission_space = MissionSpace(mission_func=self._gen_mission)
        
        if max_steps is None:
            max_steps = 3 * size**2
        
        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "Reach the goal"

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        
        self.goal_pos = (width - 2, height - 2)
        self.put_obj(Goal(), *self.goal_pos)

        # comentar para quitar los muros
        self._place_walls(width, height)
        
        self._place_agent()
        
        self.mission = "Reach the goal"
    
    def _place_walls(self, width, height):
        # print("Placing walls")
        # Vertical walls
        for y in range(1, self.size-1):
            self.put_obj(Wall(), self.size // 2, y)
        
        # Horizontal walls
        for x in range(1, self.size-1):
            self.put_obj(Wall(), x, self.size//2)
        
        # Create openings in the walls
        openings = [(width//2,width//4),(width//2,width-(width//4)),(width//4,height//2),(width-(width//4),height//2),]
        
        for x, y in openings:
            self.grid.set(x, y, None)

    def _place_agent(self):
        # print("Executing place agent")
        # self.agent_pos=(5,1)
        # self.agent_dir=0

        # return
        while True:
            # print("placing agent")
            x = random.randint(1, self.size - 2)
            y = random.randint(1, self.size - 2)
            pos = (x, y)

            # print(pos)
            # Check if the position is empty (not wall, lava, floor, or goal)
            if (self.grid.get(*pos) is None and
                pos != self.goal_pos):
                self.agent_pos = pos
                self.agent_dir = random.randint(0, 3)  # Random direction
                break

    def reset(self, **kwargs):
        self.stepped_floors = set()
        obs = super().reset(**kwargs)
        return obs

    def step(self, action: Actions):
        prev_pos=self.agent_pos
        prev_dir=self.agent_dir
        obs, reward, terminated, truncated, info = super().step(action)

        # Penalización por movimiento normal
        if(self.grid.get(*self.agent_pos) is None):
            reward=-0.05
        
        # recompensa cuando se mueve en dirección a la meta
        if(self.agent_pos[0] > prev_pos[0] or self.agent_pos[1] > prev_pos[1]):
            reward += 0.02  #type: ignore

        # penalización grande por chocar contra la pared
        if(prev_dir==self.agent_dir and prev_pos == self.agent_pos):
            # print("!", end="")
            reward=-1.0
        
        # Recompensa por llegar a la meta
        if isinstance(self.grid.get(*self.agent_pos), Goal):
            reward = 100
            terminated = True
            # print("Reached the goal")
        
        return obs, reward, terminated, truncated, info
    
    def get_Q_actions_from_state(self, state) -> np.ndarray:
        if state not in self.Q:
            self.Q[state] = np.zeros(len(self.actions))  # Inicializa con ceros
        return self.Q[state]

    def get_state(self) -> tuple[int, int, int]:
        return (self.agent_pos[0], self.agent_pos[1], self.agent_dir)
    
    def train(self):
        for ep in range(self.episodes):
            self.reset()
            done = False
            truncated = False
            total_reward = 0.0

            while not (done or truncated):
                state = self.get_state()
                q_values = self.get_Q_actions_from_state(state)

                # epsilon greedy
                if np.random.rand() < self.epsilon:
                    action = self.action_space.sample()
                else:
                    action = np.argmax(q_values)

                obs, reward, done, truncated, info = self.step(action)  #type: ignore
                next_state = self.get_state()
                next_q = self.get_Q_actions_from_state(next_state)

                # Actualización Q-learning
                self.Q[state][action] += self.alpha * (reward + self.gamma * np.max(next_q) - self.Q[state][action])

                total_reward += float(reward)

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay) # cada vez se va bajando más el epsilon para q cada vez explore menos
            self.history.append({
                "episode": ep + 1,
                "epsilon": self.epsilon,
                "total_reward": total_reward,
                "states_learned": len(self.Q),
                "reached_goal": done,  # True si llegó, False si se truncó por tiempo
                "truncated": truncated
            })
            status = "GOAL" if done else "TIMEOUT"
            if not done:
                print(f"Ep {ep+1}: Reward={total_reward:.2f}, Eps={self.epsilon:.2f} {status}")

    def plot_history(self):
        df = pd.DataFrame(self.history)
        plt.figure()
        plt.plot(df["episode"], df["total_reward"], label="Total Reward")
        plt.plot(df["episode"], df["states_learned"], label="States Learned")
        plt.xlabel("Episode")
        plt.legend()
        plt.title("Q-Learning Progress")
        plt.show()
    
    def dump_qtable(self, filename="q_learning_table.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(self.Q, f)
        print(f"Q-table saved to {filename}")

    def load_qtable(self, filename="q_learning_table.pkl"):
        with open(filename, "rb") as f:
            self.Q = pickle.load(f)
        print(f"Q-table loaded from {filename}")

    def test_human(self):
        self.render_mode = "human"
        for _ in range(5):
            done = False
            obs = self.reset()
            time.sleep(.3)
            while not done:
                state = self.get_state()
                action = np.argmax(self.get_Q_actions_from_state(state))  # usa lo aprendido
                obs, reward, done, truncated, info = self.step(action)  #type: ignore
                time.sleep(0.05)





if __name__ == "__main__":
    plot_history = False
    skip_training = False
    skip_testing = False
    size = 10
    episodes = 500


    # Training 
    if not skip_training:
        print(f"Training agent for {episodes} episodes on {size}x{size} grid...")
        env = QLearningEnv(render_mode=None, size=size, episodes=episodes)
        env.train()
        env.dump_qtable()

        if plot_history:
            env.plot_history()

    # testing con UI
    if not skip_testing:
        print(f"\nTesting agent with human rendering...")
        env_human = QLearningEnv(size=size)
        env_human.load_qtable()
        env_human.test_human()
