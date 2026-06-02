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


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained Q-Learning blackjack player.")
    parser.add_argument("--model", type=Path, default=Path(__file__).with_name("player1_q_model.pkl"))
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--rounds-per-game", type=int, default=50)
    parser.add_argument("--opponents", type=int, default=2)
    parser.add_argument("--chips", type=int, default=1000)
    parser.add_argument("--base-bet", type=int, default=100)
    parser.add_argument("--card-sets", type=int, default=6)
    parser.add_argument("--guard", type=int, default=60)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    player = QLearningBlackJackPlayer.load(args.model, id=0, train=False)
    player.base_bet = args.base_bet

    rank_sum = 0
    for _ in range(args.games):
        players = [player] + [BasicStrategyPlayer(i + 1, args.base_bet) for i in range(args.opponents)]
        game = BlackJack(players, args.card_sets, args.guard, args.rounds_per_game, args.chips)
        if args.verbose:
            rank = game.run()[0]
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                rank = game.run()[0]
        rank_sum += int(rank)

    rounds = max(player.rounds_finished, 1)
    print(f"games={args.games}")
    print(f"avg_rank={rank_sum / args.games:.3f}")
    print(f"rounds_finished={player.rounds_finished}")
    print(f"total_delta={player.total_delta}")
    print(f"avg_round_delta={player.total_delta / rounds:.3f}")
    print(f"results={dict(sorted(player.results.items()))}")


if __name__ == "__main__":
    main()
