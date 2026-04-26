#!/usr/bin/env python3
"""Test script demonstrating the new scenario registry features.

This script demonstrates:
1. read_only property for scenarios
2. Range queries for n_outcomes and n_negotiators
"""

from __future__ import annotations

from pathlib import Path

from negmas.registry import ScenarioRegistry, scenario_registry


def test_read_only_feature():
    """Test the read_only feature."""
    print("\n=== Testing read_only Feature ===\n")

    # Create a test registry
    registry = ScenarioRegistry()

    # Register a regular scenario
    path1 = Path("/tmp/editable_scenario")
    info1 = registry.register(path1, name="editable", read_only=False)
    print(f"Registered '{info1.name}' as editable (read_only={info1.read_only})")

    # Register a read-only scenario
    path2 = Path("/tmp/readonly_scenario")
    info2 = registry.register(path2, name="readonly", read_only=True)
    print(f"Registered '{info2.name}' as read-only (read_only={info2.read_only})")

    # The read_only property is informational only - both can be unregistered
    print(
        "\nNote: read_only is an informational property - it doesn't prevent unregistration"
    )

    # Unregister the editable one
    result = registry.unregister(path1)
    print(f"✓ Unregistered '{info1.name}' (read_only=False): {result}")

    # Unregister the read-only one - also works
    result = registry.unregister(path2)
    print(f"✓ Unregistered '{info2.name}' (read_only=True): {result}")

    # Check built-in scenarios are read-only
    print("\nChecking built-in scenarios...")
    builtin = scenario_registry.query_by_tag("builtin")
    if builtin:
        first_builtin = list(builtin.values())[0]
        print(
            f"Built-in scenario '{first_builtin.name}' has read_only={first_builtin.read_only}, source='{first_builtin.source}'"
        )

    # Query by read_only property
    print("\nQuerying by read_only property...")
    test_registry = ScenarioRegistry()
    test_registry.register(Path("/tmp/ro1"), name="ro1", read_only=True)
    test_registry.register(Path("/tmp/ro2"), name="ro2", read_only=True)
    test_registry.register(Path("/tmp/rw1"), name="rw1", read_only=False)

    readonly_scenarios = test_registry.query(read_only=True)
    print(f"Found {len(readonly_scenarios)} read-only scenarios")

    writable_scenarios = test_registry.query(read_only=False)
    print(f"Found {len(writable_scenarios)} writable scenarios")


def test_range_queries():
    """Test range query functionality."""
    print("\n=== Testing Range Queries ===\n")

    # Create a test registry with various scenarios
    registry = ScenarioRegistry()

    # Register scenarios with different properties
    scenarios = [
        (Path("/tmp/s1"), "Small-2P", 50, 2),
        (Path("/tmp/s2"), "Medium-2P", 100, 2),
        (Path("/tmp/s3"), "Large-2P", 200, 2),
        (Path("/tmp/s4"), "Small-3P", 75, 3),
        (Path("/tmp/s5"), "Medium-3P", 150, 3),
        (Path("/tmp/s6"), "Large-4P", 300, 4),
    ]

    for path, name, n_outcomes, n_negotiators in scenarios:
        registry.register(
            path, name=name, n_outcomes=n_outcomes, n_negotiators=n_negotiators
        )

    print("Registered scenarios:")
    for info in registry.values():
        print(
            f"  - {info.name}: {info.n_outcomes} outcomes, {info.n_negotiators} negotiators"
        )

    # Query 1: Exact value
    print("\nQuery 1: Scenarios with exactly 100 outcomes")
    results = registry.query(n_outcomes=100)
    for info in results.values():
        print(f"  - {info.name}: {info.n_outcomes} outcomes")

    # Query 2: Range
    print("\nQuery 2: Scenarios with 75-150 outcomes (inclusive)")
    results = registry.query(n_outcomes=(75, 150))
    for info in results.values():
        print(f"  - {info.name}: {info.n_outcomes} outcomes")

    # Query 3: No minimum
    print("\nQuery 3: Scenarios with at most 100 outcomes")
    results = registry.query(n_outcomes=(None, 100))
    for info in results.values():
        print(f"  - {info.name}: {info.n_outcomes} outcomes")

    # Query 4: No maximum
    print("\nQuery 4: Scenarios with at least 150 outcomes")
    results = registry.query(n_outcomes=(150, None))
    for info in results.values():
        print(f"  - {info.name}: {info.n_outcomes} outcomes")

    # Query 5: Combined ranges
    print(
        "\nQuery 5: Scenarios with 50-200 outcomes AND 2-3 negotiators (bilateral/trilateral)"
    )
    results = registry.query(n_outcomes=(50, 200), n_negotiators=(2, 3))
    for info in results.values():
        print(
            f"  - {info.name}: {info.n_outcomes} outcomes, {info.n_negotiators} negotiators"
        )


if __name__ == "__main__":
    test_read_only_feature()
    test_range_queries()
    print("\n✓ All tests completed successfully!\n")
