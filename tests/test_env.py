"""Fast pytest checks for the Milestone 3 Gymnasium environment."""
import numpy as np
import pytest

from env.biped_env import BipedEnv


@pytest.fixture
def env():
    e = BipedEnv(seed=0)
    e.reset(seed=0)
    return e


def test_spaces(env):
    assert env.observation_space.shape == (47,)
    assert env.action_space.shape == (12,)


def test_reset_returns_valid_obs(env):
    obs, info = env.reset(seed=0)
    assert obs.shape == (47,)
    assert obs.dtype == np.float32
    assert np.all(np.isfinite(obs))
    assert "cmd" in info


def test_step_returns_valid_tuple(env):
    obs, reward, terminated, truncated, info = env.step(np.zeros(12, dtype=np.float32))
    assert obs.shape == (47,)
    assert np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "reward_terms" in info


def test_determinism():
    a = np.full(12, 0.1, dtype=np.float32)

    def rollout():
        e = BipedEnv(seed=5)
        obs, _ = e.reset(seed=5)
        for _ in range(5):
            obs, *_ = e.step(a)
        return obs

    np.testing.assert_array_equal(rollout(), rollout())


def test_zero_action_eventually_terminates(env):
    zero = np.zeros(12, dtype=np.float32)
    for _ in range(300):
        _, _, terminated, truncated, _ = env.step(zero)
        if terminated or truncated:
            return
    pytest.fail("episode never ended under zero action (expected a fall ~1.8s in)")
