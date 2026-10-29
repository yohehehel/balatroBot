"""Unit tests for bots.instance_pool."""

import pytest
from bots.instance_pool import InstancePool


def test_instance_pool_initialization():
    pool = InstancePool(num_instances=4, start_port=15000, headless=True, fast=True, no_save=True)
    assert pool.num_instances == 4
    assert pool.start_port == 15000
    assert pool.ports == [15000, 15001, 15002, 15003]
    assert pool.headless is True
    assert pool.fast is True
    assert pool.no_save is True


def test_instance_pool_single_instance_defaults():
    pool = InstancePool()
    assert pool.num_instances == 1
    assert pool.ports == [12346]


def test_instance_pool_platform_option(monkeypatch):
    pool_native = InstancePool(num_instances=2, platform="native")
    assert pool_native.platform == "native"

    monkeypatch.setenv("BALATROBOT_PLATFORM", "linux")
    pool_env = InstancePool(num_instances=1)
    assert pool_env.platform == "linux"


