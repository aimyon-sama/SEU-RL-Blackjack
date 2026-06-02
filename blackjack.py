import numpy
import abc


class Deck:
    def __init__(self, card_sets: int):
        """
        card_sets: packs of cards
        """
        self.card_sets = card_sets

    def reset(self):
        self.card_count = numpy.full((52,), self.card_sets)
        self.remains = numpy.sum(self.card_count)

    def deal(self):
        card = numpy.random.choice(
            52, p=self.card_count/self.remains)
        self.card_count[card] -= 1
        self.remains -= 1
        return card

    suits = ["♠", "♥", "♣", "♦"]
    points = ["A ", "2 ", "3 ", "4 ", "5 ", "6 ",
              "7 ", "8 ", "9 ", "10", "J ", "Q ", "K "]

    @classmethod
    def visualize(cls, card: int):
        return cls.card_map[card]


Deck.card_map = {
    i: f"{Deck.suits[i//13]}{Deck.points[i % 13]}" for i in range(52)}


class Player:
    def __init__(self, id):
        self.id = id

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def act(self, obs):
        pass


def analyze_card(card_list: list[int]) -> tuple[int, bool, bool, bool]:
    """
    card_list: list of card indices
    return points,notburst,is_soft,is_blackjack
    """
    cards = numpy.array(card_list, dtype=numpy.int32) % 13+1
    cards[cards > 10] = 10
    if len(cards) == 2 and ((cards[0] == 1 and cards[1] == 10) or (cards[1] == 1 and cards[0] == 10)):
        return 21, True, True, True
    have_ace = numpy.any(cards == 1)
    points = cards.sum()
    if have_ace and points <= 11:
        points += 10
        return points, True, True, False
    return points, points <= 21, False, False


def rank_desc_dense(arr):
    unique_sorted = numpy.sort(numpy.unique(arr))[::-1]
    rank_map = {val: i+1 for i, val in enumerate(unique_sorted)}
    return numpy.array([rank_map[x] for x in arr])


class BlackJack:
    def __init__(self, players: list[Player], card_sets: int, guard: int, rounds: int, chips):
        self.players = players
        self.player_number = len(players)
        self.deck = Deck(card_sets)
        self.guard = guard
        self.rounds = rounds
        self.chips = chips

    def reset(self):
        self.deck.reset()
        self.need_reset_deck = False
        self.position_chips = numpy.full(
            (self.player_number,), self.chips, dtype=numpy.int32)

        self.pos_alivestate = numpy.full(
            (self.player_number,), True, dtype=bool)

        self.player_position = numpy.random.permutation(self.player_number)
        self.position_map = {
            self.player_position[k]: k for k in range(self.player_number)}
        for pos, p in zip(self.player_position, self.players):
            p.act({"stage": "meta", "position": pos,
                  "chips": self.chips, "cardsets": self.deck.card_sets})
        self.history_info = {}

    def alive_players(self):
        for pos in range(self.player_number):
            index = self.position_map[pos]
            if not self.pos_alivestate[pos]:
                continue
            yield self.players[index], pos

    def deal_opening(self):
        self.position_bet = numpy.zeros(
            (self.player_number,), dtype=numpy.int32)
        for player, pos in self.alive_players():
            bet = player.act(
                {"stage": "bet", "last_round_history": self.history_info})
            if bet > self.position_chips[pos]:
                bet = self.position_chips[pos]
            elif bet <= 0:
                bet = 1
            self.position_bet[pos] = bet
        self.history_info={}

        self.opening_state = {"stage": "opening"}
        self.player_card = {}
        for _, pos in self.alive_players():
            card_a = self.deck.deal()
            card_b = self.deck.deal()
            self.opening_state[pos] = (card_a, card_b)
            self.opening_state[f"bet_{pos}"] = self.position_bet[pos]
            self.player_card[pos] = [card_a, card_b]
        self.dealer_open_card = self.deck.deal()
        self.dealer_hidden_card = self.deck.deal()
        self.opening_state["dealer"] = self.dealer_open_card
        self.dealer_card = [self.dealer_open_card, self.dealer_hidden_card]
        for p, _ in self.alive_players():
            p.act(self.opening_state)

    def check_deck(self):
        if self.deck.remains <= self.guard:
            self.deck.reset()
            for p, _ in self.alive_players():
                p.act({"stage": "deck_reset"})

    def player_act(self, history_info, player: Player, pos: int):
        history = []
        first_act = player.act(
            {"stage": "first_act", "history": history_info})
        if first_act == "Double":
            position_bet = self.position_bet[pos]*2
            remain_chips = self.position_chips[pos]
            if position_bet <= remain_chips:
                self.position_bet[pos] = position_bet
                card = self.deck.deal()
                self.player_card[pos].append(card)
                history.append(("Double", card))
                player.act({"stage": "deal", "card": card})
                points, notburst, _, _ = analyze_card(self.player_card[pos])
                if notburst:
                    history.append(("Stand", points))
                    self.player_points[pos] = points
                else:
                    history.append(("Burst", points))
                    self.eliminated[pos] = True
                    player.act(
                        {"stage": "finish", "chips": -self.position_bet[pos], "state": "Burst"})
                    self.position_chips[pos] -= self.position_bet[pos]
                return history
        elif first_act == "Surrender":
            position_bet = self.position_bet[pos]
            lose_chips = position_bet//2
            self.position_bet[pos] = lose_chips
            points, _, _, _ = analyze_card(self.player_card[pos])
            history.append(("Surrender", points))
            self.eliminated[pos] = True
            player.act({"stage": "finish", "chips": -
                       lose_chips, "state": "Surrender"})
            self.position_chips[pos] -= lose_chips
            return history
        done = False
        while not done:
            player_action = player.act({"stage": "act"})
            if player_action == "Stand":
                done = True
                points, _, _, _ = analyze_card(self.player_card[pos])
                history.append(("Stand", points))
            elif player_action == "Hit":
                card = self.deck.deal()
                player.act({"stage": "deal", "card": card})
                self.player_card[pos].append(card)
                history.append(("Hit", card))
                points, notburst, _, _ = analyze_card(self.player_card[pos])
                if not notburst:
                    history.append(("Burst", points))
                    done = True
                    self.eliminated[pos] = True
                    player.act({"stage": "finish", "chips": -
                               self.position_bet[pos], "state": "Burst"})
                    self.position_chips[pos] -= self.position_bet[pos]
        return history

    def dealer_turn(self):
        points, notburst, _, _ = analyze_card(self.dealer_card)
        while points < 17:
            card = self.deck.deal()
            self.dealer_card.append(card)
            points, notburst, _, _ = analyze_card(self.dealer_card)
        return points, notburst

    def run(self):
        self.reset()
        self.pos_rank = numpy.zeros((self.player_number,), dtype=numpy.int32)
        for r in range(self.rounds):
            print(f"round:{r}")
            if not numpy.any(self.pos_alivestate):
                break
            self.deal_opening()
            _, _, _, dealer_bj = analyze_card(self.dealer_card)
            if dealer_bj:
                for player, pos in self.alive_players():
                    _, _, _, player_bj = analyze_card(self.player_card[pos])
                    player.act({"stage": "dealer_card",
                               "card": self.dealer_card})
                    if player_bj:
                        player.act({"stage": "finish", "chips": 0,
                                   "state": "DealerBlackJack"})
                        self.position_bet[pos] = 0
                    else:
                        player.act({"stage": "finish", "chips": -self.position_bet[pos],
                                   "state": "DealerBlackJack"})
                        self.position_chips[pos] -= self.position_bet[pos]
                continue
            self.eliminated = numpy.zeros((self.player_number,), dtype=bool)
            self.player_points = numpy.zeros(
                (self.player_number,), dtype=numpy.int32)
            for player, pos in self.alive_players():
                history = self.player_act(self.history_info, player, pos)
                self.history_info[pos] = history

            dpoint, dnotburst = self.dealer_turn()
            for player, pos in self.alive_players():
                if not self.eliminated[pos]:
                    player.act({"stage": "dealer_card",
                               "card": self.dealer_card})
                    ppoint, _, _, pbj = analyze_card(self.player_card[pos])
                    if pbj:
                        win_chips = self.position_bet[pos]
                        win_chips = int(1.5*win_chips)
                        player.act(
                            {"stage": "finish", "chips": win_chips, "state": "Win"})
                        self.position_chips[pos] += win_chips
                        continue
                    if not dnotburst or dpoint < ppoint:
                        win_chips = self.position_bet[pos]
                        player.act(
                            {"stage": "finish", "chips": win_chips, "state": "Win"})
                        self.position_chips[pos] += win_chips
                    elif not dnotburst or dpoint == ppoint:
                        player.act(
                            {"stage": "finish", "chips": 0, "state": "tie"})
                    else:
                        player.act(
                            {"stage": "finish", "chips": -self.position_bet[pos], "state": "Lost"})
                        self.position_chips[pos] -= self.position_bet[pos]
            alive = self.position_chips > 0
            fail = ~alive & self.pos_alivestate
            if numpy.any(fail):
                self.pos_rank[fail] = r - self.rounds
            self.pos_alivestate &= alive
            self.check_deck()
        self.pos_rank += self.position_chips
        self.player_rank = numpy.array(
            [self.pos_rank[self.player_position[index]] for index in range(self.player_number)])
        return rank_desc_dense(self.player_rank)
