// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 The Hong Kong University of Science and Technology (Guangzhou), Humanoid Computing & Learning Lab.

#pragma once

#include "src/decision/blackboard.h"
#include "src/decision/high_level_command.h"
#include "src/decision/role_manager.h"
#include "src/world/world_snapshot.h"

#include <optional>

namespace decision {

/// Interface for role-specific command generation.
class RoleBehavior {
public:
    virtual ~RoleBehavior() = default;
    virtual bool matches(const Blackboard& blackboard) const = 0;
    virtual HighLevelCommand make_command(
        const world::WorldSnapshot& snapshot,
        const Blackboard& blackboard) const = 0;
};

/// Persistent attacker state carried between decision cycles.
struct APState {
    bool dribble_ready{false};
    bool set_play_released{false};
    double previous_ball_distance{0.0};
};

/// Generates the active-player command and set-play handoff state.
class APBehavior final {
public:
    bool matches(const Blackboard& blackboard) const;
    HighLevelCommand make_command(
        const world::WorldSnapshot& snapshot,
        const Blackboard& blackboard,
        RoleManager& role_manager) const;
    void reset_state() const { state_ = {}; }
private:
    mutable APState state_;
};

/// Walk-to-formation behavior shared by CBM, ST, CBL, CBR, and CDM.
class SimpleRoleBehavior final : public RoleBehavior {
public:
    explicit SimpleRoleBehavior(int role_id)
        : role_id_(role_id) {}
    bool matches(const Blackboard& blackboard) const override;
    HighLevelCommand make_command(
        const world::WorldSnapshot& snapshot,
        const Blackboard& blackboard) const override;
private:
    int role_id_;
    mutable APState relay_state_;
};

/// Generates goalkeeper positioning and goal-kick commands.
class GKBehavior final : public RoleBehavior {
public:
    bool matches(const Blackboard& blackboard) const override;
    HighLevelCommand make_command(
        const world::WorldSnapshot& snapshot,
        const Blackboard& blackboard) const override;
    void reset_state() const {}
};

/// Clears persistent state owned by all role behavior instances.
void reset_role_behavior_state();

int current_role_from_blackboard(const Blackboard& blackboard);

/// Builds a speed-limited, obstacle-aware command for an absolute walk target.
///
/// This is the common entry point for role movement so set plays, goalkeeper
/// positioning, and open play all share the same collision envelope.
WalkCommand make_avoiding_walk_command(
    const std::array<double, 2>& target_position_m,
    const world::WorldSnapshot& snapshot,
    std::optional<double> opponent_x_threshold = std::nullopt,
    bool avoid_field_boundaries = true,
    bool orient_to_ball = true,
    std::optional<int> role_id = std::nullopt,
    bool suppress_heading_slowdown = false,
    bool avoid_obstacles = true);

/// Selects the behavior matching the current role, if one is available.
std::optional<HighLevelCommand> select_role_behavior(
    const world::WorldSnapshot& snapshot,
    const Blackboard& blackboard,
    RoleManager& role_manager);

}  // namespace decision
