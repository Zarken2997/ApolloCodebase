// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 The Hong Kong University of Science and Technology (Guangzhou), Humanoid Computing & Learning Lab.

#pragma once

#include "src/math/math_utils.h"
#include "src/world/world_snapshot.h"

#include <algorithm>
#include <array>
#include <vector>

namespace decision {

/// One opponent position with the motion and age information needed by planning.
struct KnownOpponent {
    std::array<double, 2> position_m{0.0, 0.0};
    std::array<double, 2> velocity_mps{0.0, 0.0};
    double age_s{0.0};
};

inline void append_distinct_opponent_position(
    std::vector<std::array<double, 2>>& positions,
    const std::array<double, 2>& position,
    double dedup_distance_m) {
    for (const auto& existing : positions) {
        if (math::planar_dist(existing, position) < dedup_distance_m) {
            return;
        }
    }
    positions.push_back(position);
}

/// Merges recent local and team-shared opponents, keeping the freshest local copy.
inline std::vector<KnownOpponent> collect_known_opponents(
    const world::WorldSnapshot& snapshot,
    double dedup_distance_m = 1.0,
    double recent_timeout_s = 2.0) {
    std::vector<KnownOpponent> opponents;
    opponents.reserve(snapshot.opponents.size() + snapshot.shared_opponents.size());

    auto append = [&](const world::PlayerObservation& opponent) {
        if (!opponent.seen && (snapshot.server_time - opponent.last_seen_time) > recent_timeout_s) {
            return;
        }
        const std::array<double, 2> position{
            opponent.position_m[0],
            opponent.position_m[1],
        };
        for (const auto& existing : opponents) {
            if (math::planar_dist(existing.position_m, position) < dedup_distance_m) {
                return;
            }
        }
        opponents.push_back({
            position,
            opponent.velocity_valid &&
                    std::max(0.0, snapshot.server_time - opponent.last_seen_time) <= 0.5
                ? std::array<double, 2>{opponent.velocity_mps[0], opponent.velocity_mps[1]}
                : std::array<double, 2>{0.0, 0.0},
            std::max(0.0, snapshot.server_time - opponent.last_seen_time),
        });
    };

    for (const auto& opponent : snapshot.opponents) {
        append(opponent);
    }
    for (const auto& opponent : snapshot.shared_opponents) {
        append(opponent);
    }
    return opponents;
}

/// Merges recent local and team-shared opponents, removing near duplicates.
inline std::vector<std::array<double, 2>> collect_known_opponent_positions(
    const world::WorldSnapshot& snapshot,
    double dedup_distance_m = 1.0,
    double recent_timeout_s = 2.0) {
    std::vector<std::array<double, 2>> positions;
    for (const auto& opponent : collect_known_opponents(
             snapshot, dedup_distance_m, recent_timeout_s)) {
        positions.push_back(opponent.position_m);
    }
    return positions;
}

}  // namespace decision
