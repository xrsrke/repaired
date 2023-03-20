import torch

from repaired.replay import (
    PrioritizedReplayDistribution
)


def test_prioritized_replay_distribution():
    BATCH_SIZE = 3
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