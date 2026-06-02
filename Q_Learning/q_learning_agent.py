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
BET_ACTIONS = ("minimum", "half_base", "base", "double_base")
MAX_ACTIONS = max(len(FIRST_ACTIONS), len(PLAY_ACTIONS), len(BET_ACTIONS))


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
            self.last_round_history = obs.get("last_round_history", {})
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
        final_reward_weight: float = 0.4,
        chip_reward_weight: float = 0.02,
        survival_reward_weight: float = 0.25,
        elimination_penalty_weight: float = 0.5,
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
        self.final_reward_weight = final_reward_weight
        self.chip_reward_weight = chip_reward_weight
        self.survival_reward_weight = survival_reward_weight
        self.elimination_penalty_weight = elimination_penalty_weight
        self.train = train
        self.random = random.Random(seed)
        self.q: dict[tuple[Any, ...], np.ndarray] = defaultdict(lambda: np.zeros(MAX_ACTIONS, dtype=np.float64))
        self.reset()

    def reset(self):
        self.cards: list[int] = []
        self.pending: tuple[tuple[Any, ...], str] | None = None
        self.round_bet = 1
        self.initial_chips = 0
        self.last_round_history: dict[Any, Any] = {}
        self.episode_actions: list[tuple[tuple[Any, ...], str]] = []
        self.total_delta = 0
        self.rounds_finished = 0
        self.results: dict[str, int] = defaultdict(int)

    def legal_actions(self, stage: str) -> tuple[str, ...]:
        if stage == "bet":
            return BET_ACTIONS
        if stage == "first":
            actions = list(FIRST_ACTIONS)
            if self.chips < self.round_bet * 2 and "Double" in actions:
                actions.remove("Double")
            return tuple(actions)
        return PLAY_ACTIONS

    def last_history_features(self) -> tuple[int, int, int, int, int]:
        actions = [action for history in self.last_round_history.values() for action, _ in history]
        return (
            len(self.last_round_history),
            actions.count("Hit"),
            actions.count("Double"),
            actions.count("Surrender"),
            actions.count("Burst"),
        )

    def bet_amount(self, action: str) -> int:
        amounts = {
            "minimum": 1,
            "half_base": max(self.base_bet // 2, 1),
            "base": max(self.base_bet, 1),
            "double_base": max(self.base_bet * 2, 1),
        }
        return min(amounts[action], max(self.chips, 1))

    def state(self, stage: str) -> tuple[Any, ...]:
        if stage == "bet":
            return (
                stage,
                min(self.chips // max(self.base_bet, 1), 20),
                getattr(self, "pos", -1),
                *self.last_history_features(),
            )
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
        if action in BET_ACTIONS:
            return BET_ACTIONS.index(action)
        if action in FIRST_ACTIONS:
            return FIRST_ACTIONS.index(action)
        return PLAY_ACTIONS.index(action)

    def q_values(self, state: tuple[Any, ...]) -> np.ndarray:
        values = self.q[state]
        if state[0] == "bet":
            return values[:len(BET_ACTIONS)]
        if state[0] == "play":
            return values[:len(PLAY_ACTIONS)]
        return values[:len(FIRST_ACTIONS)]

    def choose(self, state: tuple[Any, ...], legal: tuple[str, ...]) -> str:
        if self.train and self.random.random() < self.epsilon:
            return self.random.choice(legal)
        values = self.q_values(state).copy()
        if state[0] == "bet":
            all_actions = BET_ACTIONS
            if np.all(values == 0):
                return "base"
        elif state[0] == "first":
            all_actions = FIRST_ACTIONS
        else:
            all_actions = PLAY_ACTIONS
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

    def remember(self, state: tuple[Any, ...], action: str):
        if self.train:
            self.episode_actions.append((state, action))

    def finish_game(self, rank: int, final_chips: int, player_count: int, survived: bool):
        if not self.train or not self.episode_actions:
            return
        if player_count > 1:
            rank_score = 1.0 - 2.0 * ((rank - 1) / (player_count - 1))
        else:
            rank_score = 0.0
        chip_score = (final_chips - self.initial_chips) / max(self.base_bet, 1)
        survival_score = 1.0 if survived else 0.0
        elimination_score = 1.0 if final_chips <= 0 else 0.0
        reward = (
            self.final_reward_weight * rank_score
            + self.chip_reward_weight * chip_score
            + self.survival_reward_weight * survival_score
            - self.elimination_penalty_weight * elimination_score
        )
        for state, action in self.episode_actions:
            idx = self.action_index(action)
            self.q[state][idx] += self.alpha * (reward - self.q[state][idx])
        self.episode_actions.clear()

    def act(self, obs: dict[str, Any]):
        stage = obs["stage"]
        if stage == "meta":
            self.pos = obs["position"]
            self.chips = obs["chips"]
            self.initial_chips = obs["chips"]
            self.card_sets = obs.get("cardsets")
        elif stage == "bet":
            self.last_round_history = obs.get("last_round_history", {})
            state = self.state("bet")
            action = self.choose(state, self.legal_actions("bet"))
            self.pending = (state, action)
            self.remember(state, action)
            return self.bet_amount(action)
        elif stage == "opening":
            self.cards = list(obs[self.pos])
            self.round_bet = max(int(obs[f"bet_{self.pos}"]), 1)
            self.dealer = card_value(obs["dealer"])
        elif stage == "first_act":
            state = self.state("first")
            self.update(0.0, state)
            action = self.choose(state, self.legal_actions("first"))
            self.pending = (state, action)
            self.remember(state, action)
            return action
        elif stage == "act":
            state = self.state("play")
            self.update(0.0, state)
            action = self.choose(state, PLAY_ACTIONS)
            self.pending = (state, action)
            self.remember(state, action)
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
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "base_bet": self.base_bet,
            "final_reward_weight": self.final_reward_weight,
            "chip_reward_weight": self.chip_reward_weight,
            "survival_reward_weight": self.survival_reward_weight,
            "elimination_penalty_weight": self.elimination_penalty_weight,
            "q": {k: v.tolist() for k, v in self.q.items()},
        }
        with path.open("wb") as f:
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
            final_reward_weight=data.get("final_reward_weight", 0.4),
            chip_reward_weight=data.get("chip_reward_weight", 0.02),
            survival_reward_weight=data.get("survival_reward_weight", 0.25),
            elimination_penalty_weight=data.get("elimination_penalty_weight", 0.5),
            train=train,
        )
        player.q = defaultdict(lambda: np.zeros(MAX_ACTIONS, dtype=np.float64))
        for key, value in data["q"].items():
            arr = np.array(value, dtype=np.float64)
            if arr.shape[0] < MAX_ACTIONS:
                arr = np.pad(arr, (0, MAX_ACTIONS - arr.shape[0]))
            player.q[key] = arr
        return player
