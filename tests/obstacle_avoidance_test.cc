// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 The Hong Kong University of Science and Technology (Guangzhou), Humanoid Computing & Learning Lab.

#include "src/decision/field_geometry.h"
#include "src/decision/walk_planner.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

namespace {

using decision::WalkPlan;
using decision::field_geometry::Position2;

void require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

world::WorldSnapshot make_snapshot() {
    world::WorldSnapshot snapshot;
    snapshot.team_name = "Test";
    snapshot.player_number = 2;
    snapshot.server_time = 10.0;
    snapshot.self.position_m = {0.0, 0.0, 0.7};
    snapshot.ball.position_m = {0.0, 0.0, 0.11};
    snapshot.ball.visible = true;
    return snapshot;
}

world::PlayerObservation make_player(
    double x,
    double y,
    double vx = 0.0,
    double vy = 0.0) {
    world::PlayerObservation player;
    player.player_number = 1;
    player.seen = true;
    player.last_seen_time = 10.0;
    player.position_m = {x, y, 0.5};
    player.velocity_mps = {vx, vy, 0.0};
    player.velocity_valid = std::abs(vx) + std::abs(vy) > 1.0e-9;
    return player;
}

WalkPlan plan(
    const world::WorldSnapshot& snapshot,
    const Position2& target) {
    return decision::plan_walk(
        {0.0, 0.0},
        target,
        snapshot,
        snapshot.player_number);
}

void test_clear_path() {
    const auto snapshot = make_snapshot();
    const WalkPlan result = plan(snapshot, {4.0, 0.0});
    require(result.path_available, "clear path must be available");
    require(std::abs(result.heading_deg) < 1.0, "clear path must point at target");
    require(result.speed_scale > 0.99, "clear path must allow full speed");
}

void test_static_obstacle() {
    auto snapshot = make_snapshot();
    snapshot.opponents.push_back(make_player(2.0, 0.0));
    const WalkPlan result = plan(snapshot, {4.0, 0.0});
    require(result.path_available, "static-obstacle path must be available");
    require(std::abs(result.heading_deg) > 10.0, "static obstacle must cause a detour");
    require(result.speed_scale > 0.0, "static-obstacle detour must permit motion");
}

void test_moving_obstacle() {
    auto snapshot = make_snapshot();
    snapshot.opponents.push_back(make_player(2.2, 0.0, -2.0, 0.0));
    const WalkPlan result = plan(snapshot, {4.0, 0.0});
    require(result.path_available, "moving-obstacle path must be available");
    require(std::abs(result.heading_deg) > 10.0, "swept moving obstacle must cause a detour");
}

void test_occupied_goal() {
    auto snapshot = make_snapshot();
    snapshot.opponents.push_back(make_player(4.0, 0.0));
    const WalkPlan result = plan(snapshot, {4.0, 0.0});
    require(result.path_available, "occupied goal must resolve to a nearby feasible cell");
    require(std::abs(result.heading_deg) > 10.0, "occupied goal must not steer into the obstacle");
    require(result.speed_scale > 0.0, "occupied-goal detour must permit motion");
}

void test_no_path_stops() {
    auto snapshot = make_snapshot();
    constexpr int kRingCount = 6;
    constexpr double kRingRadiusM = 1.2;
    for (int i = 0; i < kRingCount; ++i) {
        const double angle = 2.0 * 3.14159265358979323846 *
                             static_cast<double>(i) / static_cast<double>(kRingCount);
        snapshot.opponents.push_back(make_player(
            kRingRadiusM * std::cos(angle),
            kRingRadiusM * std::sin(angle)));
    }
    const WalkPlan result = plan(snapshot, {4.0, 0.0});
    require(!result.path_available, "closed obstacle ring must report no path");
    require(result.speed_scale == 0.0, "closed obstacle ring must command zero speed");
}

void test_stale_obstacle_expansion() {
    auto fresh = make_snapshot();
    fresh.opponents.push_back(make_player(1.25, 0.0));
    auto stale = fresh;
    stale.opponents.front().seen = false;
    stale.opponents.front().last_seen_time = 8.5;

    const WalkPlan fresh_result = plan(fresh, {4.0, 0.0});
    const WalkPlan stale_result = plan(stale, {4.0, 0.0});
    require(
        !stale_result.path_available || stale_result.speed_scale <= fresh_result.speed_scale,
        "stale obstacle expansion must not increase allowed speed");
}

void test_field_target_clamp() {
    const Position2 clamped = decision::field_geometry::clamp_position_to_field({
        100.0,
        -100.0,
    });
    require(
        clamped[0] <= decision::field_geometry::kActualHalfLengthM,
        "forward target must stay inside field");
    require(
        clamped[1] >= -decision::field_geometry::kActualHalfWidthM,
        "lateral target must stay inside field");
}

void test_boundary_speed_limit() {
    auto snapshot = make_snapshot();
    snapshot.self.position_m = {26.6, 0.0, 0.7};
    const WalkPlan result = decision::plan_walk(
        {26.6, 0.0},
        {26.9, 0.0},
        snapshot,
        snapshot.player_number);
    require(result.path_available, "near-boundary path must remain available");
    require(result.speed_scale < 0.6, "field boundary must reduce approach speed");
}

}  // namespace

int main() {
    test_clear_path();
    test_static_obstacle();
    test_moving_obstacle();
    test_occupied_goal();
    test_no_path_stops();
    test_stale_obstacle_expansion();
    test_field_target_clamp();
    test_boundary_speed_limit();
    std::cout << "obstacle avoidance tests passed\n";
    return 0;
}
