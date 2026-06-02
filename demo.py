from blackjack import Player, BlackJack, Deck, analyze_card


class ClassicalStradegy(Player):
    def __init__(self, id):
        super().__init__(id)

    def act(self, obs):
        obs_type = obs["stage"]
        if obs_type == "meta":
            """
            Observation 
            position: player position
            chips: player total chips

            No action
            """
            self.pos = obs["position"]
            self.chips = obs["chips"]
            print(f"My position:{self.pos}, My chips:{self.chips}")
        elif obs_type == "bet":
            """
            No Observation

            Action space: int [0,chips]
            """
            if self.chips < 100:
                bet = self.chips
            else:
                bet = 100
            print(
                f"Pos:{self.pos}, ID:{self.id}, I have {self.chips}, I bet {bet}")
            print(f"Last round {obs["last_round_history"]}")
            return bet
        elif obs_type == "opening":
            """
            Observation
            All players cards and bets and dealer's open card

            No action
            """
            self.my_card = list(obs[self.pos])
            self.my_bet = obs[f"bet_{self.pos}"]
            self.dealer = obs["dealer"] % 13+1
            if self.dealer > 10:
                self.dealer = 10
            print(
                f"Pos:{self.pos}, ID:{self.id}, My card:{[Deck.visualize(card) for card in self.my_card]}, Dealer card:{Deck.visualize(obs["dealer"])}")
        elif obs_type == "deck_reset":
            """
            No Observation

            No action
            """
            print(
                f"Pos:{self.pos}, ID:{self.id}, I know the deck is resetted")
        elif obs_type == "first_act":
            """
            Observation:
            Previous player act history

            Action space: "Double", "Surrender", "Continue"
            """
            history = obs["history"]
            print(
                f"Pos:{self.pos}, ID:{self.id}, Previous players:{history}")
            my_points, _, is_soft, _ = analyze_card(self.my_card)
            if is_soft:
                if 13 <= my_points <= 18 and 4 <= self.dealer <= 6:
                    print(
                        f"Pos:{self.pos}, ID:{self.id}, Double, Cards={[Deck.visualize(card) for card in self.my_card]}")
                    return "Double"
                else:
                    print(
                        f"Pos:{self.pos}, ID:{self.id}, Continue, Cards={[Deck.visualize(card) for card in self.my_card]}")
                    return "Continue"
            else:
                if my_points == 9 and 3 <= self.dealer <= 6:
                    print(
                        f"Pos:{self.pos}, ID:{self.id}, Double, Cards={[Deck.visualize(card) for card in self.my_card]}")
                    return "Double"
                elif my_points == 10 and 2 <= self.dealer <= 9:
                    print(
                        f"Pos:{self.pos}, ID:{self.id}, Double, Cards={[Deck.visualize(card) for card in self.my_card]}")
                    return "Double"
                elif my_points == 11 and 2 <= self.dealer <= 10:
                    print(
                        f"Pos:{self.pos}, ID:{self.id}, Double, Cards={[Deck.visualize(card) for card in self.my_card]}")
                    return "Double"
                elif my_points == 16 and (self.dealer == 1 or self.dealer == 9 or self.dealer == 10):
                    print(
                        f"Pos:{self.pos}, ID:{self.id}, Surrender, Cards={[Deck.visualize(card) for card in self.my_card]}")
                    return "Surrender"
                elif my_points == 17 and self.dealer == 1:
                    print(
                        f"Pos:{self.pos}, ID:{self.id}, Surrender, Cards={[Deck.visualize(card) for card in self.my_card]}")
                    return "Surrender"
                else:
                    print(
                        f"Pos:{self.pos}, ID:{self.id}, Continue, Cards={[Deck.visualize(card) for card in self.my_card]}")
                    return "Continue"
        elif obs_type == "act":
            """
            No Observation

            Action space: "Hit", "Stand"
            """
            my_points, _, is_soft, _ = analyze_card(self.my_card)
            if is_soft:
                if 19 <= my_points <= 21:
                    print(
                        f"Pos:{self.pos}, ID:{self.id}, Stand, Cards={[Deck.visualize(card) for card in self.my_card]}")
                    return "Stand"
                elif my_points == 18:
                    if 2 <= self.dealer <= 6:
                        print(
                            f"Pos:{self.pos}, ID:{self.id}, Stand, Cards={[Deck.visualize(card) for card in self.my_card]}")
                        return "Stand"
                    else:
                        print(
                            f"Pos:{self.pos}, ID:{self.id}, Hit, Cards={[Deck.visualize(card) for card in self.my_card]}")
                        return "Hit"
                else:
                    print(
                        f"Pos:{self.pos}, ID:{self.id}, Hit, Cards={[Deck.visualize(card) for card in self.my_card]}")
                    return "Hit"
            else:
                if 17 <= my_points <= 21:
                    print(
                        f"Pos:{self.pos}, ID:{self.id}, Stand, Cards={[Deck.visualize(card) for card in self.my_card]}")
                    return "Stand"
                elif 13 <= my_points <= 16:
                    if 2 <= self.dealer <= 6:
                        print(
                            f"Pos:{self.pos}, ID:{self.id}, Stand, Cards={[Deck.visualize(card) for card in self.my_card]}")
                        return "Stand"
                    else:
                        print(
                            f"Pos:{self.pos}, ID:{self.id}, Hit, Cards={[Deck.visualize(card) for card in self.my_card]}")
                        return "Hit"
                else:
                    print(
                        f"Pos:{self.pos}, ID:{self.id}, Hit, Cards={[Deck.visualize(card) for card in self.my_card]}")
                    return "Hit"
        elif obs_type == "deal":
            """
            Observation:
            card: card index

            No action
            """
            card = obs["card"]
            self.my_card.append(card)
            print(
                f"Pos:{self.pos}, ID:{self.id}, I get card {Deck.visualize(card)}. Now I have {[Deck.visualize(card) for card in self.my_card]}")
        elif obs_type == "finish":
            """
            Observation:
            chips: remaining chips, state

            No action
            """
            chips = obs["chips"]
            state = obs["state"]
            self.chips += chips
            print(
                f"Pos:{self.pos}, ID:{self.id}, I get {chips} chips, total {self.chips}, state:{state}")
        elif obs_type == "dealer_card":
            deal_card = obs["card"]
            print(
                f"Pos:{self.pos}, ID:{self.id}, I see dealer's card {[Deck.visualize(card) for card in deal_card]}")


class Interactive(Player):
    def __init__(self, id):
        super().__init__(id)

    def act(self, obs):
        obs_type = obs["stage"]
        if obs_type == "meta":
            """
            Observation 
            position: player position
            chips: player total chips

            No action
            """
            self.pos = obs["position"]
            self.chips = obs["chips"]
            print(f"My position:{self.pos}, My chips:{self.chips}")
        elif obs_type == "bet":
            """
            No Observation

            Action space: int [0,chips]
            """
            bet = input(f"Bet turn, I have {self.chips}, I bet:")
            return int(bet)
        elif obs_type == "opening":
            """
            Observation
            All players cards and bets and dealer's open card

            No action
            """
            self.cards = {}
            self.bets = {}
            for k, v in obs.items():
                if k == "stage":
                    continue
                elif k == "dealer":
                    self.dealer = v
                elif isinstance(k, str):
                    continue
                else:
                    self.cards[k] = list(v)
                    self.bets[k] = obs[f"bet_{k}"]
        elif obs_type == "deck_reset":
            """
            No Observation

            No action
            """
            print(f"I know the deck is resetted")
        elif obs_type == "first_act":
            """
            Observation:
            Previous player act history

            Action space: "Double", "Surrender", "Continue"
            """
            print("Opening:")
            for k in self.cards.keys():
                if k == self.pos:
                    print(f"Pos {k} (Me):", end="")
                else:
                    print(f"Pos {k}:", end="")
                print(
                    f"Bet {self.bets[k]} {[Deck.visualize(card) for card in self.cards[k]]}")
            print("History:")
            for k, h in obs["history"].items():
                format_h = []
                for s, a in h:
                    if s in ["Double", "Hit"]:
                        format_h.append((s, Deck.visualize(a)))
                    else:
                        format_h.append((s, f"Points: {a}"))
                print(f"Position {k}:{format_h}")
            my_action = input(
                "Select action from {Double, Surrender, Continue}")
            return my_action
        elif obs_type == "act":
            """
            No Observation

            Action space: "Hit", "Stand"
            """
            my_action = input("Select action from {Hit, Stand}")
            return my_action
        elif obs_type == "deal":
            """
            Observation:
            card: card index

            No action
            """
            card = obs["card"]
            print(f"New card:{Deck.visualize(card)}")
        elif obs_type == "finish":
            """
            Observation:
            chips: remaining chips, state

            No action
            """
            chips = obs["chips"]
            state = obs["state"]
            self.chips += chips
            print(
                f"Pos:{self.pos}, ID:{self.id}, I get {chips} chips, total {self.chips}, state:{state}")
        elif obs_type == "dealer_card":
            deal_card = obs["card"]
            print(
                f"Pos:{self.pos}, ID:{self.id}, I see dealer's card {[Deck.visualize(card) for card in deal_card]}")


if __name__ == "__main__":
    players = [ClassicalStradegy(
        0), ClassicalStradegy(1), ClassicalStradegy(2)]
    game = BlackJack(players, 6, 60, 50, 1000)
    rank = game.run()
    print(rank)
