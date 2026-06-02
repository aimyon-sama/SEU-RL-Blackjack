from __future__ import annotations

import argparse
import contextlib
import io
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from blackjack import BlackJack
from q_learning_agent import BasicStrategyPlayer, QLearningBlackJackPlayer


def player_name(path: Path) -> str:
    return path.stem


def main():
    parser = argparse.ArgumentParser(description="Compare blackjack models in the same table.")
    parser.add_argument("--models", type=Path, nargs="+", required=True)
    parser.add_argument("--basic-opponents", type=int, default=0)
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--rounds-per-game", type=int, default=50)
    parser.add_argument("--chips", type=int, default=1000)
    parser.add_argument("--base-bet", type=int, default=100)
    parser.add_argument("--card-sets", type=int, default=6)
    parser.add_argument("--guard", type=int, default=60)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    stats = {
        player_name(model): {
            "rank_sum": 0,
            "chips_sum": 0,
            "rounds": 0,
            "delta": 0,
            "results": defaultdict(int),
        }
        for model in args.models
    }

    for _ in range(args.games):
        model_players = [
            QLearningBlackJackPlayer.load(model, id=index, train=False)
            for index, model in enumerate(args.models)
        ]
        for player in model_players:
            player.base_bet = args.base_bet
        basic_players = [
            BasicStrategyPlayer(len(model_players) + index, args.base_bet)
            for index in range(args.basic_opponents)
        ]
        players = model_players + basic_players
        game = BlackJack(players, args.card_sets, args.guard, args.rounds_per_game, args.chips)
        if args.verbose:
            ranks = game.run()
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                ranks = game.run()

        for index, model in enumerate(args.models):
            name = player_name(model)
            player = model_players[index]
            final_chips = int(game.position_chips[game.player_position[index]])
            stats[name]["rank_sum"] += int(ranks[index])
            stats[name]["chips_sum"] += final_chips
            stats[name]["rounds"] += player.rounds_finished
            stats[name]["delta"] += player.total_delta
            for state, count in player.results.items():
                stats[name]["results"][state] += count

    print(f"games={args.games}")
    print(f"players={len(args.models) + args.basic_opponents}")
    for name, item in stats.items():
        rounds = max(item["rounds"], 1)
        print(f"\nmodel={name}")
        print(f"avg_rank={item['rank_sum'] / args.games:.3f}")
        print(f"avg_final_chips={item['chips_sum'] / args.games:.3f}")
        print(f"avg_round_delta={item['delta'] / rounds:.3f}")
        print(f"rounds_finished={item['rounds']}")
        print(f"results={dict(sorted(item['results'].items()))}")


if __name__ == "__main__":
    main()
