// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 The Hong Kong University of Science and Technology (Guangzhou), Humanoid Computing & Learning Lab.

#pragma once

#include "src/world/world_snapshot.h"

#include <array>
#include <limits>
#include <optional>

namespace decision {

/// Obstacle-aware heading and safety envelope produced by the local walk planner.
struct WalkPlan {
    double heading_deg{0.0};
    /// Multiplicative speed limit in [0, 1] derived from path clearance.
    double speed_scale{1.0};
    /// Clearance from the selected waypoint to the nearest hard-obstacle boundary.
    double minimum_clearance_m{std::numeric_limits<double>::infinity()};
    /// False means no collision-free escape/approach path was found this cycle.
    bool path_available{true};
};

/// Computes an obstacle-avoiding walk command envelope using grid A*.
///
/// All nearby players are swept obstacles unless `avoid_obstacles` is false.
/// Fresh opponent velocities are predicted over the planning horizon, and stale
/// observations grow their hard radius with age. The caller multiplies its base
/// speed by `speed_scale` and stops translation when `path_available` is false.
WalkPlan plan_walk(
    const std::array<double, 2>& self_pos,
    const std::array<double, 2>& target_pos,
    const world::WorldSnapshot& snapshot,
    int self_player_number,
    std::optional<double> opponent_x_threshold = std::nullopt,
    bool avoid_field_boundaries = true,
    bool avoid_obstacles = true);

}  // namespace decision
