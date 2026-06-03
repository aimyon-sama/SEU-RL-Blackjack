from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from blackjack import BlackJack
from q_learning_agent import BasicStrategyPlayer, QLearningBlackJackPlayer


def apply_training_config(player: QLearningBlackJackPlayer, args: argparse.Namespace, seed: int | None = None):
    player.alpha = args.alpha
    player.gamma = args.gamma
    player.epsilon = args.epsilon
    player.epsilon_min = args.epsilon_min
    player.epsilon_decay = args.epsilon_decay
    player.base_bet = args.base_bet
    player.final_reward_weight = args.final_reward_weight
    player.chip_reward_weight = args.chip_reward_weight
    player.survival_reward_weight = args.survival_reward_weight
    player.elimination_penalty_weight = args.elimination_penalty_weight
    player.burst_penalty_weight = args.burst_penalty_weight
    player.train = True
    if seed is not None:
        player.random.seed(seed)


def make_learner(player_id: int, args: argparse.Namespace, seed: int) -> QLearningBlackJackPlayer:
    if args.init_model is not None:
        player = QLearningBlackJackPlayer.load(args.init_model, id=player_id, train=True)
        apply_training_config(player, args, seed)
        return player
    return QLearningBlackJackPlayer(
        player_id,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        base_bet=args.base_bet,
        final_reward_weight=args.final_reward_weight,
        chip_reward_weight=args.chip_reward_weight,
        survival_reward_weight=args.survival_reward_weight,
        elimination_penalty_weight=args.elimination_penalty_weight,
        burst_penalty_weight=args.burst_penalty_weight,
        train=True,
        seed=seed,
    )


def make_self_play_players(args: argparse.Namespace) -> list[QLearningBlackJackPlayer]:
    players = [make_learner(player_id, args, args.seed + player_id) for player_id in range(args.players)]
    shared_q = players[0].q
    for player in players[1:]:
        player.q = shared_q
    return players


def make_basic_opponent_players(args: argparse.Namespace) -> list[QLearningBlackJackPlayer | BasicStrategyPlayer]:
    return [make_learner(0, args, args.seed)] + [BasicStrategyPlayer(i + 1, args.base_bet) for i in range(args.players - 1)]


def make_history_pool_players(args: argparse.Namespace) -> list[QLearningBlackJackPlayer | BasicStrategyPlayer]:
    opponent_models = args.opponent_models or []
    if 1 + len(opponent_models) > args.players:
        raise ValueError("--players must be at least 1 + len(--opponent-models)")

    players: list[QLearningBlackJackPlayer | BasicStrategyPlayer] = [make_learner(0, args, args.seed)]
    for model in opponent_models:
        opponent = QLearningBlackJackPlayer.load(model, id=len(players), train=False)
        opponent.base_bet = args.base_bet
        players.append(opponent)
    while len(players) < args.players:
        players.append(BasicStrategyPlayer(len(players), args.base_bet))
    return players


def run_game(players: list[QLearningBlackJackPlayer | BasicStrategyPlayer], args: argparse.Namespace) -> np.ndarray:
    game = BlackJack(players, args.card_sets, args.guard, args.rounds_per_game, args.chips)
    if args.verbose:
        ranks = game.run()
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            ranks = game.run()
    for index, player in enumerate(players):
        if isinstance(player, QLearningBlackJackPlayer):
            position = game.player_position[index]
            final_chips = int(game.position_chips[position])
            survived = bool(game.pos_alivestate[position])
            player.finish_game(int(ranks[index]), final_chips, len(players), survived)
    return ranks


def main():
    parser = argparse.ArgumentParser(description="Train a Q-Learning blackjack player.")
    parser.add_argument("--mode", choices=("self-play", "basic-opponents"), default="self-play")
    parser.add_argument("--games", type=int, default=20000)
    parser.add_argument("--rounds-per-game", type=int, default=50)
    parser.add_argument("--players", type=int, default=3)
    parser.add_argument("--chips", type=int, default=1000)
    parser.add_argument("--base-bet", type=int, default=100)
    parser.add_argument("--card-sets", type=int, default=6)
    parser.add_argument("--guard", type=int, default=60)
    parser.add_argument("--alpha", type=float, default=0.08)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--epsilon", type=float, default=0.25)
    parser.add_argument("--epsilon-min", type=float, default=0.02)
    parser.add_argument("--epsilon-decay", type=float, default=0.9995)
    parser.add_argument("--final-reward-weight", type=float, default=0.5)
    parser.add_argument("--chip-reward-weight", type=float, default=0.03)
    parser.add_argument("--survival-reward-weight", type=float, default=0.05)
    parser.add_argument("--elimination-penalty-weight", type=float, default=0.2)
    parser.add_argument("--burst-penalty-weight", type=float, default=0.1)
    parser.add_argument("--init-model", type=Path)
    parser.add_argument("--opponent-models", type=Path, nargs="*", default=[])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--save", type=Path, default=Path(__file__).with_name("model") / "player1_q_model.pkl")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.players < 1:
        raise ValueError("--players must be at least 1")

    if args.opponent_models:
        players = make_history_pool_players(args)
    elif args.mode == "self-play":
        players = make_self_play_players(args)
    else:
        players = make_basic_opponent_players(args)
    learner = players[0]
    if args.init_model is not None:
        print(f"init_model={args.init_model}")
    if args.opponent_models:
        print(f"opponent_models={[str(model) for model in args.opponent_models]}")

    rank_sum = 0
    chips_sum = 0
    for game_index in range(1, args.games + 1):
        ranks = run_game(players, args)
        rank_sum += int(ranks[0])
        chips_sum += int(getattr(learner, "chips", 0))
        if game_index % max(args.games // 20, 1) == 0:
            avg_rank = rank_sum / game_index
            avg_chips = chips_sum / game_index
            avg_delta = learner.total_delta / max(learner.rounds_finished, 1)
            print(
                f"game={game_index}/{args.games} "
                f"mode={args.mode} "
                f"epsilon={learner.epsilon:.4f} "
                f"avg_rank={avg_rank:.3f} "
                f"avg_final_chips={avg_chips:.1f} "
                f"avg_round_delta={avg_delta:.3f} "
                f"q_states={len(learner.q)}"
            )

    learner.save(args.save)
    print(f"saved_model={args.save}")


if __name__ == "__main__":
    main()
