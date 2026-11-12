// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 The Hong Kong University of Science and Technology (Guangzhou), Humanoid Computing & Learning Lab.

#include "src/decision/decision_manager.h"
#include "src/decision/high_level_command.h"
#include "src/math/math_utils.h"
#include "src/world/frame_normalizer.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <variant>

namespace {

world::WorldSnapshot make_shot_snapshot(double self_x, double ball_x) {
    world::WorldSnapshot snapshot;
    snapshot.team_name = "Student";
    snapshot.player_number = 2;
    snapshot.server_time = 10.0;
    snapshot.play_mode = world::PlayMode::PlayOn;
    snapshot.play_mode_group = world::PlayModeGroup::Other;
    snapshot.self.position_m = {self_x, 0.0, 0.7};
    snapshot.ball.visible = true;
    snapshot.ball.position_m = {ball_x, 0.0, 0.11};
    snapshot.teammates.resize(7);
    for (int i = 0; i < 7; ++i) {
        auto& mate = snapshot.teammates[static_cast<std::size_t>(i)];
        mate.player_number = i + 1;
        mate.is_teammate = true;
        mate.position_m = {-20.0 - static_cast<double>(i), 5.0, 0.7};
        mate.last_seen_time = -1.0;
    }
    auto& self_mate = snapshot.teammates[1];
    self_mate.seen = true;
    self_mate.position_m = snapshot.self.position_m;
    self_mate.last_seen_time = snapshot.server_time;
    return snapshot;
}

void require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

double command_world_x(
    const decision::WalkCommand& command,
    const world::WorldSnapshot& snapshot) {
    if (command.target_absolute) {
        return command.target_2d_m[0] - snapshot.self.position_m[0];
    }
    const double yaw_deg =
        world::FrameNormalizer::yaw_deg_from_quaternion_wxyz(
            snapshot.self.orientation_wxyz);
    return math::rotate_2d(command.target_2d_m, yaw_deg)[0];
}

double decide_world_x(
    decision::DecisionManager& manager,
    const world::WorldSnapshot& snapshot) {
    const auto command = manager.decide(snapshot);
    const auto* walk = std::get_if<decision::WalkCommand>(&command);
    require(walk != nullptr, "shot scenario must emit a WalkCommand");
    return command_world_x(*walk, snapshot);
}

void test_monotonic_shot_progress() {
    decision::DecisionManager manager;
    constexpr double kEpsilon = 1.0e-6;

    require(
        decide_world_x(manager, make_shot_snapshot(19.58, 20.18)) > 0.0,
        "initial shot approach must move toward the goal");
    require(
        decide_world_x(manager, make_shot_snapshot(23.0, 24.0)) > 0.0,
        "rolling push must continue toward the goal");
    require(
        decide_world_x(manager, make_shot_snapshot(26.8, 27.2)) >= -kEpsilon,
        "front boundary clamp must not reverse the shot");
    require(
        decide_world_x(manager, make_shot_snapshot(27.3, 27.0)) >= -kEpsilon,
        "lost push contact in the finish strip must not send the AP backward");
    require(
        decide_world_x(manager, make_shot_snapshot(27.1, 27.4)) >= -kEpsilon,
        "release near the goal plane must not reverse the shot");
    require(
        decide_world_x(manager, make_shot_snapshot(27.1, 28.0)) >= -kEpsilon,
        "ball beyond the goal line must not pull the AP backward");
}

}  // namespace

int main() {
    test_monotonic_shot_progress();
    std::cout << "shot boundary avoidance tests passed\n";
    return 0;
}
