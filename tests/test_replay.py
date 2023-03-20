import torch

from repaired.replay import (
    PrioritizedReplayDistribution,
    PrioritizedReplay
)


def test_prioritized_replay_distribution():
    BATCH_SIZE = 1
    N_LEVELS = 10

    score_levels = torch.randn((BATCH_SIZE, N_LEVELS))
    last_count_levels = torch.randperm(n=score_levels.shape[-1])
    last_episode = torch.max(last_count_levels)
    dist = PrioritizedReplayDistribution()

    dist = dist.create(score_levels, last_count_levels, last_episode)

    assert dist.shape[0] == BATCH_SIZE
    assert dist.shape[1] == N_LEVELS
    assert torch.allclose(
        dist.sum(dim=-1), torch.ones(BATCH_SIZE),
        rtol=1e-03, atol=1e-05
    )

def test_experience_collection():
    BATCH_SIZE = 5
    N_LEVELS = 10

    levels = torch.tensor([1, 2, 3, 4, 5])
    visited_levels = torch.tensor([
        [1, 3],
        # [2, 4]
    ])
    level_scores = torch.tensor([
        [0.4, 0.1],
        # [0.3, 0.9]
    ])
    last_timestep_levels = torch.tensor([
        [1, 4],
        # [3, 2]
    ])
    last_episode = torch.tensor([5, 4])

    collection = PrioritizedReplay(
        levels,
        policy=None
    )

    next_level = collection.sample_next_level(
        visited_levels,
        level_scores,
        last_timestep_levels,
        last_episode
    )

    assert isinstance(next_level, torch.Tensor)
    assert next_level.numel() > 0