import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import List, Tuple, Optional
import copy

from models import Card, Player, Pile, SUITS, RANKS
from actions import create_action, ActionType, Action
from sweep_game import SweepGame


ACTION_ENCODING_SIZE = 109

class SweepStateEncoder:
    """Converts game state to neural network input"""

    def __init__(self):
        # State components:
        # - Hand cards (52 dimensions - one-hot for each possible card)
        # - Table cards (52 dimensions)
        # - Table Piles info (5 x 55 (creators (2), doubled (1), cards (52)) dimensions for values 9-13, one hot for cards in each pile)
        # - Unseen Cards (52 dimensions)
        # - Score differential (1 dimension)
        # - Last to pick up (1 dimension)
        # - First to Play (1 dimension)
        # - Cards Left (Curr PLayer, Opponent) -> 2 dimensions
        # - Game phase (1 dimension: 0=first half, 1=second half)
        self.state_size = 52 + 52 + (5 * 55) + 52 + 1 + 1 + 1 + 2 + 1  # = 437

    def encode_cards(self, cards: List[Card]) -> np.ndarray:
        """Convert card list to one-hot encoding"""
        encoding = np.zeros(52)
        for card in cards:
            idx = self.card_to_index(card)
            encoding[idx] = 1
        return encoding

    def card_to_index(self, card: Card) -> int:
        """Convert card to index (0-51)"""
        suit_idx = SUITS.index(card.suit)
        rank_idx = RANKS.index(card.rank)
        return suit_idx * 13 + rank_idx

    def encode_state(self, game: SweepGame, player_idx: int) -> np.ndarray:
        """Encode full game state from player's perspective"""
        state = np.zeros(self.state_size)
        curr_player = game.players[player_idx]

        # Hand: 0-51
        # Table Cards: 52-103
        # Piles: 104-158, 159-213, 214-268, 269-323, 323-378
        # Unseen: 379-430
        # Score Differential: 431
        # Last to pick up: 432
        # First to Play: 433
        # Cards Left: 434-435
        # Game phase: 436

        # Player's hand
        hand_encoding = self.encode_cards(curr_player.hand)
        state[0:52] = hand_encoding

        # Table cards (excluding piles for now, just individual cards)
        table_cards = [item for item in game.table if isinstance(item, Card)]
        table_encoding = self.encode_cards(table_cards)
        state[52:104] = table_encoding

        # Pile information
        for i in range(5):
            pile_encoding = np.zeros(55)
            if (9 + i) in game.piles:
                pile = game.piles[9 + i]
                pile_encoding[0] = int(curr_player in pile.creators)
                pile_encoding[1] = int(game.players[1 - player_idx] in pile.creators)
                pile_encoding[2] = int(pile.doubled)
                pile_encoding[3:54] = self.encode_cards(pile.cards)
            start_idx = 104 + (55 * i)
            state[start_idx: start_idx + 55] = pile_encoding

        unseen_encoding = self.encode_cards(list(curr_player.unseen_cards))
        state[379:431] = unseen_encoding

        # Score differential (current player - opponent)
        score_diff = (
            curr_player.points - game.players[1 - player_idx].points
        )
        state[431] = score_diff / 100.0  # Normalize

        state[432] = game.last_to_pick_up
        state[433] = game.first_to_play
        state[434] = len(curr_player.hand)
        state[435] = len(game.players[1 - player_idx].hand)

        # Game phase
        state[436] = 0 if len(game.deck) > 0 else 1

        return state


class DQN(nn.Module):
    """Deep Q-Network for Sweep"""

    def __init__(self, input_size: int, output_size: int = 1, hidden_size: int = 256):
        super(DQN, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),  # Single Q-value output
        )

    def forward(self, x):
        return self.network(x)


class SweepDQNAgent:
    """DQN Agent for Sweep game"""

    def __init__(
        self,
        input_size: int,
        learning_rate: float = 0.001,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        memory_size: int = 10000,
    ):

        self.input_size = input_size
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Experience replay memory
        self.memory = deque(maxlen=memory_size)

        # Neural networks
        self.q_network = DQN(input_size, 1)  # Output single Q-value
        self.target_network = DQN(input_size, 1)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.encoder = SweepStateEncoder()

    def remember(self, state, action_features, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action_features, reward, next_state, done))

    def act(self, game: SweepGame, player_idx: int, valid_actions: List) -> int:
        """Choose action using epsilon-greedy policy"""
        if random.random() <= self.epsilon:
            return random.randrange(len(valid_actions))

        state = self.encoder.encode_state(game, player_idx)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        q_values = []
        for action in valid_actions:
            # Create action features (simplified)
            action_features = self.encode_action(action, game, player_idx)
            combined_input = torch.cat([state_tensor, action_features], dim=1)
            q_value = self.q_network(combined_input)
            q_values.append(q_value.item())

        return np.argmax(q_values)

    def encode_action(self, action: Action, game: SweepGame, player_idx: int) -> torch.Tensor:
        """Encode action as feature vector"""

        features = np.zeros(ACTION_ENCODING_SIZE)
        features[action.action_type.value] = 1.0
        features[3] = 1.0 if action.causes_sweep else 0.0
        features[4] = action.value / 13
        features[5:56] = self.encoder.encode_cards([action.played_card])
        features[57:108] = self.encoder.encode_cards(action.other_cards)

        return torch.FloatTensor(features).unsqueeze(0)

    def replay(self, batch_size: int = 32):
        """Train the model on a batch of experiences"""
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        action_features = torch.FloatTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])

        # Current Q values
        combined_input = torch.cat([states, action_features], dim=1)
        current_q_values = self.q_network(combined_input)

        # Next Q values from target network
        next_combined = torch.cat([next_states, action_features], dim=1)  # Simplified
        next_q_values = self.target_network(next_combined).detach()
        target_q_values = rewards + (0.99 * next_q_values * ~dones)

        loss = nn.MSELoss()(current_q_values, target_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())


class SweepTrainer:
    """Training environment for Sweep RL agent"""

    def __init__(self):
        self.encoder = SweepStateEncoder()
        self.agent = SweepDQNAgent(
            self.encoder.state_size + ACTION_ENCODING_SIZE
        )  # +109 for action features

    def calculate_reward(
        self, game: SweepGame, player_idx: int, prev_points: int, action
    ) -> float:
        """Calculate reward for an action"""
        # Basic reward based on points gained
        points_gained = game.players[player_idx].points - prev_points
        reward = points_gained

        # Bonus for sweeps
        if action.causes_sweep:
            reward += 50

        # Small penalty for throwing (encourages more strategic play)
        # if action.action_type == ActionType.THROW:
        #     reward -= 1

        # # Bonus for picking up high-value cards
        # if action.action_type == ActionType.PICK_UP:
        #     reward += len(action.other_cards) * 2

        return reward

    def train_episode(self) -> Tuple[int, int]:
        """Train one episode and return final scores"""
        game = SweepGame()
        game.initialize_round()

        # Simplified training - just play through one round
        game.first_move()

        episode_rewards = [0, 0]

        while any(len(p.hand) > 0 for p in game.players):
            player_idx = game.turn
            valid_actions = game.get_valid_actions()

            if len(valid_actions) == 0:
                print("NO VALID ACTIONS!!!")
                break

            # Get state before action
            state = self.encoder.encode_state(game, player_idx)
            prev_points = game.players[player_idx].points

            # Agent chooses action
            if game.players[player_idx].is_ai:
                action_idx = self.agent.act(game, player_idx, valid_actions)
                action = valid_actions[action_idx]
            else:
                # For training, make both players use the agent
                action_idx = self.agent.act(game, player_idx, valid_actions)
                action = valid_actions[action_idx]

            # Execute action
            action.execute(game)

            # Calculate reward
            reward = self.calculate_reward(game, player_idx, prev_points, action)
            episode_rewards[player_idx] += reward

            # Get next state
            next_state = self.encoder.encode_state(game, player_idx)

            # Store experience (simplified)
            action_features = (
                self.agent.encode_action(action, game, player_idx).numpy().flatten()
            )
            done = len(game.players[player_idx].hand) == 0

            self.agent.remember(state, action_features, reward, next_state, done)

            game.turn = 1 - game.turn

        # Train the agent
        self.agent.replay()

        return game.players[0].points, game.players[1].points

    def train(self, episodes: int = 1000, update_target_every: int = 100):
        """Train the agent for specified number of episodes"""
        scores = []

        for episode in range(episodes):
            p1_score, p2_score = self.train_episode()
            scores.append((p1_score, p2_score))

            if episode % update_target_every == 0:
                self.agent.update_target_network()

            if episode % 100 == 0:
                avg_p1 = np.mean([s[0] for s in scores[-100:]])
                avg_p2 = np.mean([s[1] for s in scores[-100:]])
                print(
                    f"Episode {episode}, Avg scores: P1={avg_p1:.1f}, P2={avg_p2:.1f}, Epsilon={self.agent.epsilon:.3f}"
                )

        return scores


# Usage example
if __name__ == "__main__":
    trainer = SweepTrainer()
    scores = trainer.train(episodes=1000)

    # Save the trained model
    torch.save(trainer.agent.q_network.state_dict(), "sweep_dqn_model.pth")
