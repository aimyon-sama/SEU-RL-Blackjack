from __future__ import annotations

import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from blackjack import Player, analyze_card


FIRST_ACTIONS = ("Double", "Surrender", "Continue")
PLAY_ACTIONS = ("Hit", "Stand")


def card_value(card: int) -> int:
    value = card % 13 + 1
    return 10 if value > 10 else value


def pair_value(cards: list[int]) -> int:
    if len(cards) != 2:
        return 0
    a, b = card_value(cards[0]), card_value(cards[1])
    return a if a == b else 0


class BasicStrategyPlayer(Player):
    """Quiet baseline player, equivalent to the strategy in demo.py."""

    def __init__(self, id: int, base_bet: int = 100):
        super().__init__(id)
        self.base_bet = base_bet

    def reset(self):
        self.cards = []

    def act(self, obs: dict[str, Any]):
        stage = obs["stage"]
        if stage == "meta":
            self.pos = obs["position"]
            self.chips = obs["chips"]
        elif stage == "bet":
            return min(self.base_bet, self.chips)
        elif stage == "opening":
            self.cards = list(obs[self.pos])
            self.bet = obs[f"bet_{self.pos}"]
            self.dealer = card_value(obs["dealer"])
        elif stage == "first_act":
            points, _, is_soft, _ = analyze_card(self.cards)
            if is_soft:
                return "Double" if 13 <= points <= 18 and 4 <= self.dealer <= 6 else "Continue"
            if points == 9 and 3 <= self.dealer <= 6:
                return "Double"
            if points == 10 and 2 <= self.dealer <= 9:
                return "Double"
            if points == 11 and 2 <= self.dealer <= 10:
                return "Double"
            if points == 16 and self.dealer in (1, 9, 10):
                return "Surrender"
            if points == 17 and self.dealer == 1:
                return "Surrender"
            return "Continue"
        elif stage == "act":
            points, _, is_soft, _ = analyze_card(self.cards)
            if is_soft:
                if points >= 19:
                    return "Stand"
                if points == 18:
                    return "Stand" if 2 <= self.dealer <= 6 else "Hit"
                return "Hit"
            if points >= 17:
                return "Stand"
            if 13 <= points <= 16:
                return "Stand" if 2 <= self.dealer <= 6 else "Hit"
            return "Hit"
        elif stage == "deal":
            self.cards.append(obs["card"])
        elif stage == "finish":
            self.chips += obs["chips"]


class QLearningBlackJackPlayer(Player):
    def __init__(
        self,
        id: int,
        alpha: float = 0.08,
        gamma: float = 0.98,
        epsilon: float = 0.1,
        epsilon_min: float = 0.02,
        epsilon_decay: float = 0.9995,
        base_bet: int = 100,
        train: bool = True,
        seed: int | None = None,
    ):
        super().__init__(id)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.base_bet = base_bet
        self.train = train
        self.random = random.Random(seed)
        self.q: dict[tuple[Any, ...], np.ndarray] = defaultdict(lambda: np.zeros(3, dtype=np.float64))
        self.reset()

    def reset(self):
        self.cards: list[int] = []
        self.pending: tuple[tuple[Any, ...], str] | None = None
        self.round_bet = 1
        self.total_delta = 0
        self.rounds_finished = 0
        self.results: dict[str, int] = defaultdict(int)

    def legal_actions(self, stage: str) -> tuple[str, ...]:
        if stage == "first":
            actions = list(FIRST_ACTIONS)
            if self.chips < self.round_bet * 2 and "Double" in actions:
                actions.remove("Double")
            return tuple(actions)
        return PLAY_ACTIONS

    def state(self, stage: str) -> tuple[Any, ...]:
        points, not_bust, is_soft, is_blackjack = analyze_card(self.cards)
        points = min(points, 22)
        return (
            stage,
            points,
            int(not_bust),
            int(is_soft),
            int(is_blackjack),
            self.dealer,
            len(self.cards),
            pair_value(self.cards),
            min(self.chips // max(self.base_bet, 1), 20),
        )

    def action_index(self, action: str) -> int:
        if action in FIRST_ACTIONS:
            return FIRST_ACTIONS.index(action)
        return PLAY_ACTIONS.index(action)

    def q_values(self, state: tuple[Any, ...]) -> np.ndarray:
        values = self.q[state]
        if state[0] == "play":
            return values[:2]
        return values

    def choose(self, state: tuple[Any, ...], legal: tuple[str, ...]) -> str:
        if self.train and self.random.random() < self.epsilon:
            return self.random.choice(legal)
        values = self.q_values(state).copy()
        all_actions = FIRST_ACTIONS if state[0] == "first" else PLAY_ACTIONS
        illegal = set(all_actions) - set(legal)
        for action in illegal:
            values[self.action_index(action)] = -np.inf
        return all_actions[int(np.argmax(values))]

    def update(self, reward: float, next_state: tuple[Any, ...] | None):
        if not self.train or self.pending is None:
            self.pending = None
            return
        state, action = self.pending
        idx = self.action_index(action)
        bootstrap = 0.0 if next_state is None else float(np.max(self.q_values(next_state)))
        target = reward + self.gamma * bootstrap
        self.q[state][idx] += self.alpha * (target - self.q[state][idx])
        self.pending = None

    def act(self, obs: dict[str, Any]):
        stage = obs["stage"]
        if stage == "meta":
            self.pos = obs["position"]
            self.chips = obs["chips"]
            self.card_sets = obs.get("cardsets")
        elif stage == "bet":
            return min(self.base_bet, self.chips)
        elif stage == "opening":
            self.cards = list(obs[self.pos])
            self.round_bet = max(int(obs[f"bet_{self.pos}"]), 1)
            self.dealer = card_value(obs["dealer"])
            self.pending = None
        elif stage == "first_act":
            state = self.state("first")
            action = self.choose(state, self.legal_actions("first"))
            self.pending = (state, action)
            return action
        elif stage == "act":
            state = self.state("play")
            self.update(0.0, state)
            action = self.choose(state, PLAY_ACTIONS)
            self.pending = (state, action)
            return action
        elif stage == "deal":
            self.cards.append(obs["card"])
        elif stage == "finish":
            reward = float(obs["chips"]) / float(self.round_bet)
            self.update(reward, None)
            self.chips += obs["chips"]
            self.total_delta += obs["chips"]
            self.rounds_finished += 1
            self.results[obs["state"]] += 1
            if self.train:
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        elif stage == "dealer_card":
            pass

    def save(self, path: str | Path):
        data = {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "base_bet": self.base_bet,
            "q": {k: v.tolist() for k, v in self.q.items()},
        }
        with Path(path).open("wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str | Path, id: int = 0, train: bool = False):
        with Path(path).open("rb") as f:
            data = pickle.load(f)
        player = cls(
            id=id,
            alpha=data.get("alpha", 0.08),
            gamma=data.get("gamma", 0.98),
            epsilon=0.0 if not train else data.get("epsilon", 0.0),
            epsilon_min=data.get("epsilon_min", 0.02),
            epsilon_decay=data.get("epsilon_decay", 0.9995),
            base_bet=data.get("base_bet", 100),
            train=train,
        )
        player.q = defaultdict(lambda: np.zeros(3, dtype=np.float64))
        for key, value in data["q"].items():
            player.q[key] = np.array(value, dtype=np.float64)
        return player
