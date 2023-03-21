import torch

from repaired.replay import (
    PrioritizedReplayDistribution,
    PrioritizedReplay
)
from repaired.level import Level

def test_prioritized_replay_distribution():
    level_1 = Level("level_1")
    level_2 = Level("level_2")
    level_3 = Level("level_3")

    score_levels = {level_1: 0.2, level_2: 0.1, level_3: 0.9}
    last_count_levels = {level_1: 3, level_2: 1, level_3: 4}
    last_episode = 4
    dist = PrioritizedReplayDistribution()

    prioritized_dist = dist.create(score_levels, last_count_levels, last_episode)

    assert isinstance(prioritized_dist, torch.Tensor)
    assert len(prioritized_dist) == len(score_levels)
    assert prioritized_dist.sum(dim=-1) == 1

def test_prioritized_replay_2():
    level_1 = Level("level_1")
    level_2 = Level("level_2")
    level_3 = Level("level_3")
    level_4 = Level("level_4")
    level_5 = Level("level_5")

    levels = [level_1, level_2, level_3, level_4, level_5]

    visited_levels = [level_1, level_3]
    score_levels = {level_1: 0.2, level_3: 0.1}
    last_episode_levels = {level_1: 3, level_3: 1}
    last_episode = 3

    replay = PrioritizedReplay(levels)

    next_level = replay.sample_next_level(
        visited_levels,
        score_levels,
        last_episode_levels,
        last_episode
    )

    assert isinstance(next_level, Level)
    assert next_level in levels

# @pytest.mark.skip(reason="Not implemented yet")
# def test_x_prioritized_replay():
#     levels = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
#     visited_levels = torch.tensor([])
#     score_levels = torch.tensor([])
#     last_episode_levels = torch.zeros_like(levels)

#     def random_init_level(levels):
#         return random.randint(0, levels.shape[0] - 1)

#     def update_history(
#         next_level,
#         episode, # the current episode that next_level was sampled
#         visited_levels, score_levels, last_episode_levels
#     ):
#         # global visited_levels
#         # global score_levels
#         # global last_episode_levels

#         visited_levels = torch.cat([visited_levels, next_level.unsqueeze(0)], dim=-1).long()
#         score_levels = torch.cat([score_levels, torch.randn(1)], dim=-1).long()
#         last_episode_levels[next_level] = episode

#         return visited_levels, score_levels, last_episode_levels

#     init_level = torch.tensor(random_init_level(levels))
#     episode = 0

#     visited_levels, score_levels, last_episode_levels = update_history(
#         init_level, episode,
#         visited_levels, score_levels, last_episode_levels
#     )

#     replay = PrioritizedReplay(levels)

#     while True:
#         next_level = replay.sample_next_level(
#             visited_levels, score_levels,
#             last_episode_levels, last_episode=episode
#         )

#         visited_levels, score_levels, last_episode_levels = update_history(
#             init_level, episode,
#             visited_levels, score_levels, last_episode_levels
#         )

#         pass