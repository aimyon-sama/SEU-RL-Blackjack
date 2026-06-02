from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from blackjack import BlackJack
from q_learning_agent import BasicStrategyPlayer, QLearningBlackJackPlayer


def run_game(player: QLearningBlackJackPlayer, args: argparse.Namespace) -> int:
    players = [player] + [BasicStrategyPlayer(i + 1, args.base_bet) for i in range(args.opponents)]
    game = BlackJack(players, args.card_sets, args.guard, args.rounds_per_game, args.chips)
    if args.verbose:
        rank = game.run()[0]
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            rank = game.run()[0]
    return int(rank)


def main():
    parser = argparse.ArgumentParser(description="Train a Q-Learning blackjack player.")
    parser.add_argument("--games", type=int, default=20000)
    parser.add_argument("--rounds-per-game", type=int, default=50)
    parser.add_argument("--opponents", type=int, default=2)
    parser.add_argument("--chips", type=int, default=1000)
    parser.add_argument("--base-bet", type=int, default=100)
    parser.add_argument("--card-sets", type=int, default=6)
    parser.add_argument("--guard", type=int, default=60)
    parser.add_argument("--alpha", type=float, default=0.08)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--epsilon", type=float, default=0.25)
    parser.add_argument("--epsilon-min", type=float, default=0.02)
    parser.add_argument("--epsilon-decay", type=float, default=0.9995)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--save", type=Path, default=Path(__file__).with_name("player1_q_model.pkl"))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    player = QLearningBlackJackPlayer(
        0,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        base_bet=args.base_bet,
        train=True,
        seed=args.seed,
    )

    rank_sum = 0
    for game_index in range(1, args.games + 1):
        rank_sum += run_game(player, args)
        if game_index % max(args.games // 20, 1) == 0:
            avg_rank = rank_sum / game_index
            avg_delta = player.total_delta / max(player.rounds_finished, 1)
            print(
                f"game={game_index}/{args.games} "
                f"epsilon={player.epsilon:.4f} "
                f"avg_rank={avg_rank:.3f} "
                f"avg_round_delta={avg_delta:.3f} "
                f"q_states={len(player.q)}"
            )

    player.save(args.save)
    print(f"saved_model={args.save}")


if __name__ == "__main__":
    main()
