#Onn Azulay 325759173
#PCA-Driven Constraint Penalty inside (E)KF EKF, Final Proj, Estimation Theory (086777)

#IMPORTANT - initial conditions must be feasible (not intersecting with constraints)


using LinearAlgebra, Statistics, ProgressMeter, Measures, DelimitedFiles
import Plots
import PyPlot

# ============================================================
# TYPES
# ============================================================

mutable struct AgentState
    x::Float64
    y::Float64
    vx::Float64
    vy::Float64
    s::Float64
end

mutable struct AgentBelief
    mean::Vector{Float64}      # [x, y, vx, vy]
    cov::Matrix{Float64}       # 4x4 covariance
end

struct CarModel
    c_drag::Float64
    c_slip::Float64
end

# ============================================================
# PCA ESTIMATOR SETTINGS / DIAGNOSTICS
# ============================================================

mutable struct PCAConstraintSettings
    retained_variance::Float64
    fixed_rank::Union{Nothing,Int}
    weighting_mode::Symbol       # :risk or :inverse
    alpha::Float64
    eps_eig::Float64
    exact_projection::Bool

    wall_activation_margin::Float64
    block_angle_margin::Float64
    block_radial_margin::Float64
end

mutable struct PCAConstraintDiagnostics
    active::Bool
    residual_mean::Vector{Float64}
    residual_cov::Matrix{Float64}
    eigvals::Vector{Float64}
    retained_rank::Int
    M::Matrix{Float64}
end

function empty_pca_diagnostics()
    return PCAConstraintDiagnostics(
        false,
        Float64[],
        zeros(0, 0),
        Float64[],
        0,
        zeros(0, 0)
    )
end

mutable struct BlockPassMemory
    active::Bool
    block_index::Int
    pass_side::Symbol   # :outward or :inward
end

mutable struct TwoAgentScenario
    blue_true::AgentState
    red_true::AgentState

    blue_self_belief::AgentBelief
    red_self_belief::AgentBelief

    blue_belief_on_red::AgentBelief
    red_belief_on_blue::AgentBelief

    blue_block_memory::BlockPassMemory
    red_block_memory::BlockPassMemory
end

mutable struct TwoAgentLog
    t::Vector{Float64}

    blue_true::Vector{Tuple{Float64,Float64}}
    red_true::Vector{Tuple{Float64,Float64}}

    blue_self_mean::Vector{Tuple{Float64,Float64}}
    red_self_mean::Vector{Tuple{Float64,Float64}}

    blue_on_red_mean::Vector{Tuple{Float64,Float64}}
    red_on_blue_mean::Vector{Tuple{Float64,Float64}}

    blue_self_cov::Vector{Matrix{Float64}}
    red_self_cov::Vector{Matrix{Float64}}

    blue_on_red_cov::Vector{Matrix{Float64}}
    red_on_blue_cov::Vector{Matrix{Float64}}

    tr_blue_self::Vector{Float64}
    tr_red_self::Vector{Float64}
    tr_blue_on_red::Vector{Float64}
    tr_red_on_blue::Vector{Float64}

    collision_cost_blue::Vector{Float64}
    collision_cost_red::Vector{Float64}

    distance_true::Vector{Float64}

    u_blue_hist::Vector{Vector{Float64}}
    u_red_hist::Vector{Vector{Float64}}

    blue_saw_red_hist::Vector{Bool}
    red_saw_blue_hist::Vector{Bool}

    blue_self_meas_hist::Vector{Bool}
    red_self_meas_hist::Vector{Bool}

    # PCA diagnostics — prediction
    pca_pred_blue_self_active::Vector{Bool}
    pca_pred_red_self_active::Vector{Bool}
    pca_pred_blue_on_red_active::Vector{Bool}
    pca_pred_red_on_blue_active::Vector{Bool}

    pca_pred_blue_self_rank::Vector{Int}
    pca_pred_red_self_rank::Vector{Int}
    pca_pred_blue_on_red_rank::Vector{Int}
    pca_pred_red_on_blue_rank::Vector{Int}

    pca_pred_blue_self_lambda_max::Vector{Float64}
    pca_pred_red_self_lambda_max::Vector{Float64}
    pca_pred_blue_on_red_lambda_max::Vector{Float64}
    pca_pred_red_on_blue_lambda_max::Vector{Float64}

    # PCA diagnostics — update
    pca_upd_blue_self_active::Vector{Bool}
    pca_upd_red_self_active::Vector{Bool}
    pca_upd_blue_on_red_active::Vector{Bool}
    pca_upd_red_on_blue_active::Vector{Bool}

    pca_upd_blue_self_rank::Vector{Int}
    pca_upd_red_self_rank::Vector{Int}
    pca_upd_blue_on_red_rank::Vector{Int}
    pca_upd_red_on_blue_rank::Vector{Int}

    pca_upd_blue_self_lambda_max::Vector{Float64}
    pca_upd_red_self_lambda_max::Vector{Float64}
    pca_upd_blue_on_red_lambda_max::Vector{Float64}
    pca_upd_red_on_blue_lambda_max::Vector{Float64}

    blue_block_clear_hist::Vector{Float64}
    red_block_clear_hist::Vector{Float64}

    blue_inner_wall_clear_hist::Vector{Float64}
    red_inner_wall_clear_hist::Vector{Float64}

    blue_outer_wall_clear_hist::Vector{Float64}
    red_outer_wall_clear_hist::Vector{Float64}

    blue_critical_moment_hist::Vector{Bool}
    red_critical_moment_hist::Vector{Bool}

    blue_block_wrong_side_hist::Vector{Bool}
    red_block_wrong_side_hist::Vector{Bool}

    blue_block_safe_side_hist::Vector{Bool}
    red_block_safe_side_hist::Vector{Bool}

    blue_block_near_sector_hist::Vector{Bool}
    red_block_near_sector_hist::Vector{Bool}

    du_blue_r_hist::Vector{Float64}
    du_red_r_hist::Vector{Float64}

    err_blue_self_pos::Vector{Float64}
    err_red_self_pos::Vector{Float64}
    err_blue_on_red_pos::Vector{Float64}
    err_red_on_blue_pos::Vector{Float64}

    future_min_clear_blue::Vector{Float64}
    future_min_clear_red::Vector{Float64}

    blue_block_event_id::Vector{Int}
    red_block_event_id::Vector{Int}

    blue_true_vx::Vector{Float64}
    blue_true_vy::Vector{Float64}
    red_true_vx::Vector{Float64}
    red_true_vy::Vector{Float64}

    blue_true_x::Vector{Float64}
    blue_true_y::Vector{Float64}
    red_true_x::Vector{Float64}
    red_true_y::Vector{Float64}

    blue_self_x::Vector{Float64}
    blue_self_y::Vector{Float64}
    red_self_x::Vector{Float64}
    red_self_y::Vector{Float64}

    blue_on_red_x::Vector{Float64}
    blue_on_red_y::Vector{Float64}
    red_on_blue_x::Vector{Float64}
    red_on_blue_y::Vector{Float64}

    distance_true_hist::Vector{Float64}

    blue_self_meas_hist_num::Vector{Float64}
    red_self_meas_hist_num::Vector{Float64}
    blue_saw_red_hist_num::Vector{Float64}
    red_saw_blue_hist_num::Vector{Float64}

    # nominal reference for stage16_vs_nominal
    nom_blue_self_x::Vector{Float64}
    nom_blue_self_y::Vector{Float64}
    nom_red_self_x::Vector{Float64}
    nom_red_self_y::Vector{Float64}

    # 2x2 covariance entries for offline ellipse reconstruction
    blue_self_P11::Vector{Float64}
    blue_self_P12::Vector{Float64}
    blue_self_P22::Vector{Float64}

    red_self_P11::Vector{Float64}
    red_self_P12::Vector{Float64}
    red_self_P22::Vector{Float64}

    blue_on_red_P11::Vector{Float64}
    blue_on_red_P12::Vector{Float64}
    blue_on_red_P22::Vector{Float64}

    red_on_blue_P11::Vector{Float64}
    red_on_blue_P12::Vector{Float64}
    red_on_blue_P22::Vector{Float64}


end

# ============================================================
# 2) PLANNER INTERFACE
# ============================================================

abstract type AbstractTwoAgentPlanner end

struct GreedyPlanner <: AbstractTwoAgentPlanner
end

# ============================================================
# STAGE 11 — NOMINAL TWO-PLAYER ROLLOUT STRUCTURES
# ============================================================

mutable struct NominalTwoPlayerStep
    blue_self_mean::Vector{Float64}
    red_self_mean::Vector{Float64}

    blue_on_red_mean::Vector{Float64}
    red_on_blue_mean::Vector{Float64}

    blue_self_cov::Matrix{Float64}
    red_self_cov::Matrix{Float64}

    blue_on_red_cov::Matrix{Float64}
    red_on_blue_cov::Matrix{Float64}

    u_blue::Vector{Float64}
    u_red::Vector{Float64}

    J_blue_stage::Float64
    J_red_stage::Float64
    J_blue_coll::Float64
    J_red_coll::Float64
end

mutable struct NominalTwoPlayerRollout
    steps::Vector{NominalTwoPlayerStep}
    J_blue_total::Float64
    J_red_total::Float64
    J_blue_stage_hist::Vector{Float64}
    J_red_stage_hist::Vector{Float64}
    J_blue_coll_hist::Vector{Float64}
    J_red_coll_hist::Vector{Float64}
end

# ============================================================
# STAGE 12 — LOCAL MODELS
# ============================================================

mutable struct Stage12LocalDynamicsModel
    G_b::Matrix{Float64}
    G_uB::Matrix{Float64}
    G_uR::Matrix{Float64}
    g0::Vector{Float64}
end

mutable struct Stage12LocalQuadraticCost
    c::Float64
    q::Vector{Float64}
    Q::Matrix{Float64}
    n_b::Int
    n_uB::Int
    n_uR::Int
end

# ============================================================
# STAGE 13 — ONE-STEP LOCAL GAME IMPROVEMENT
# ============================================================

mutable struct Stage13StepResult
    u_blue_nom::Vector{Float64}
    u_red_nom::Vector{Float64}

    du_blue::Vector{Float64}
    du_red::Vector{Float64}

    u_blue_new::Vector{Float64}
    u_red_new::Vector{Float64}

    blue_cost_nom::Float64
    red_cost_nom::Float64

    blue_cost_quad_pred::Float64
    red_cost_quad_pred::Float64
end

mutable struct Stage13ImprovedRollout
    base_rollout::NominalTwoPlayerRollout
    improved_rollout::NominalTwoPlayerRollout
    step_results::Vector{Stage13StepResult}
end

# ============================================================
# STAGE 14 — TRUE TWO-PLAYER BACKWARD PASS
# ============================================================

mutable struct Stage14ValueModel
    v::Float64
    v_b::Vector{Float64}
    V_bb::Matrix{Float64}
end

mutable struct Stage14BackwardStep
    j_blue::Vector{Float64}
    K_blue::Matrix{Float64}

    j_red::Vector{Float64}
    K_red::Matrix{Float64}

    cond_game_matrix::Float64
end

mutable struct Stage14BackwardPassResult
    steps::Vector{Stage14BackwardStep}
    blue_terminal_value::Stage14ValueModel
    red_terminal_value::Stage14ValueModel
end

# ============================================================
# STAGE 15 — FORWARD POLICY UPDATE + OUTER LOOP
# ============================================================

mutable struct Stage15IterationResult
    rollout::NominalTwoPlayerRollout
    backward_pass::Stage14BackwardPassResult
    J_blue_total::Float64
    J_red_total::Float64
end

mutable struct Stage15OuterLoopResult
    iterations::Vector{Stage15IterationResult}
    final_rollout::NominalTwoPlayerRollout
    final_backward_pass::Stage14BackwardPassResult
end

# ============================================================
# GLOBAL TUNERS
# ============================================================

# -----------------------------
# Simulation tuners
# -----------------------------
const dt = 0.02
const T_sim = 20.0
const N_test_default = 300

# -----------------------------
# Belief tuners
# -----------------------------
const belief_std_x0  = 0.20
const belief_std_y0  = 0.20
const belief_std_vx0 = 0.10
const belief_std_vy0 = 0.10

const q_pos = 1e-5
const q_vel_base = 0.005
const q_vel_gain = 0.002

const true_accel_noise_t = 0.10
const true_accel_noise_r = 0.15
const true_vel_noise_xy  = 0.02

# -----------------------------
# Measurement tuners
# -----------------------------
const meas_std_light = 0.12

# -----------------------------
# Track tuners
# -----------------------------
const R_inner = 30.0
const track_width = 12.0
const R_outer = R_inner + track_width
const R_center = 0.5 * (R_inner + R_outer)
const N_centerline = 300

# -----------------------------
# Vehicle tuners
# -----------------------------
const c_drag_default = 0.02
const c_slip_default = 0.20

# -----------------------------
# Initial-condition tuners
# -----------------------------
const theta0_default = 0.7
const r0_default = R_center
const v_t0_default = 4.0

# -----------------------------
# Stage-10 / 11 planning tuners
# -----------------------------
const v_des_default = 7.0

const inner_margin_default = 0.75
const outer_margin_default = 0.75
const block_clearance_default = 0.75

const a_t_max = 2.0 #originally 1.0
const a_r_max = 3.0

const H = 48

const H_base = H
const H_mid = H
const H_far = H
const H_max = H

const a_r_candidates_stage10 = collect(-a_r_max:0.25:a_r_max)
const a_t_candidates_stage10 = collect(0.0:0.05:a_t_max)

# -----------------------------
# Cost tuners
# -----------------------------

const w_prog  = 28.0
const w_wall  = 8.0
const w_block = 14.0
const w_coll_default = 16.0
const w_info  = 10.0
const w_u     = 0.20

const d_safe_coll_default = 1.0
const σ_coll_default = 0.9
const k_unc_coll_default = 1.2

const use_block_memory = Ref(true)

const angle_release_margin = 0.08
const extra_tail_steps = 10

const COV_INFLATION = Ref(1.0) #for debugging, in case the filter becomes too optimistic...

const w_preview_deadend_stage10 = 0.0
const boundary_penalty_eps = 1e-3

# -----------------------------
# PCA-constrained EKF tuners
# -----------------------------
const estimator_mode = Ref(:gupta)
# supported modes:
#   :gupta
#   :gupta_pca_full
#   :gupta_pca_trunc95
#   :gupta_pca_full_ep
#   :gupta_pca_trunc95_ep

const pca_settings = Ref(
    PCAConstraintSettings(
        0.95,       # retained_variance
        nothing,    # fixed_rank
        :risk,      # weighting_mode
        0.15,        # alpha
        1e-8,       # eps_eig
        false,      # exact_projection
        2.0,        # wall_activation_margin
        0.08,       # block_angle_margin
        2.0         # block_radial_margin
    )
)

const PCA_PRINT_DEBUG = Ref(false)
const PCA_PRINT_EVERY = Ref(25)
const sim_step = Ref(0)

# ============================================================
# PAPER-CONSISTENCY SWITCHES
# ============================================================

const enforce_first_order_beliefs_in_planner = Ref(true)


# Prediction PCA
const last_pca_pred_blue_self   = Ref(empty_pca_diagnostics())
const last_pca_pred_red_self    = Ref(empty_pca_diagnostics())
const last_pca_pred_blue_on_red = Ref(empty_pca_diagnostics())
const last_pca_pred_red_on_blue = Ref(empty_pca_diagnostics())

# Update PCA
const last_pca_upd_blue_self   = Ref(empty_pca_diagnostics())
const last_pca_upd_red_self    = Ref(empty_pca_diagnostics())
const last_pca_upd_blue_on_red = Ref(empty_pca_diagnostics())
const last_pca_upd_red_on_blue = Ref(empty_pca_diagnostics())

# ============================================================
# 3B) DIAGNOSTICS / BATCH-RUN TUNERS
# ============================================================

const N_WALL_ANCHORS  = 5
const N_BLOCK_ANCHORS = 5
const WALL_ANCHOR_SPAN = 0.035
const BLOCK_ANCHOR_SPAN = 0.040

const RUN_ALL_ESTIMATOR_MODES = true
const SAVE_DIAGNOSTIC_CSV = true
const SAVE_RAW_SERIALIZED = false   # optional, leave false unless you add JLD2/BSON
const DIAG_OUTPUT_DIR = "pca_diagnostics_outputs"

# Critical-event / future-risk tuners
const future_risk_horizon_steps = 15
const risky_clearance_threshold = 1.25
const critical_wall_margin = 1.5
const critical_block_margin = 1.5

# Event/anticipation tuning
const anticipation_lookback_steps = 12

# Optional control-smoothness diagnostic
const compute_du_diagnostics = true

# ============================================================
# TRACK GEOMETRY
# ============================================================

const centerline = [
    [R_center * cos(θ), R_center * sin(θ)]
    for θ in range(0, 2π, length = N_centerline + 1)
]

const forbidden_blocks = [
    (theta_center = π/2.3, half_angle = 0.16, side = "inner", radial_depth = 3.0),
    (theta_center = π,     half_angle = 0.18, side = "inner", radial_depth = 6.0),
    (theta_center = 11π/6, half_angle = 0.14, side = "inner", radial_depth = 10.0)
]

const light_width = 3.0

const light_zones = [
    (center = 0.0,   half_angle = 0.3, side = "inner"),
    (center = π/4,   half_angle = 0.3, side = "inner"),
    (center = 2π/3,  half_angle = 0.3, side = "outer"),
    (center = 6π/4,  half_angle = 0.3, side = "inner")
]

# ============================================================
# BASIC INITIALIZATION HELPERS
# ============================================================

function create_belief_from_state(
    state::AgentState;
    std_x::Float64 = belief_std_x0,
    std_y::Float64 = belief_std_y0,
    std_vx::Float64 = belief_std_vx0,
    std_vy::Float64 = belief_std_vy0
)
    mean = [state.x, state.y, state.vx, state.vy]
    cov = Matrix(Diagonal([std_x^2, std_y^2, std_vx^2, std_vy^2]))
    return AgentBelief(mean, cov)
end

function create_two_agent_scenario(
    blue_true::AgentState,
    red_true::AgentState
)
    blue_self_belief = create_belief_from_state(blue_true)
    red_self_belief  = create_belief_from_state(red_true)

    blue_belief_on_red = create_belief_from_state(red_true)
    red_belief_on_blue = create_belief_from_state(blue_true)

    blue_mem = BlockPassMemory(false, 0, :inward)
    red_mem  = BlockPassMemory(false, 0, :inward)

    return TwoAgentScenario(
        blue_true,
        red_true,
        blue_self_belief,
        red_self_belief,
        blue_belief_on_red,
        red_belief_on_blue,
        blue_mem,
        red_mem
    )
end

# ============================================================
# ANGLE / LOCAL GEOMETRY HELPERS
# ============================================================

function wrap_angle_pi(θ::Float64)
    return mod(θ + π, 2π) - π
end

function unit_vectors(x::Float64, y::Float64)
    r = hypot(x, y)

    if r < 1e-12
        t_hat = [0.0, 1.0]
        r_hat = [1.0, 0.0]
        θ = 0.0
        return t_hat, r_hat, θ
    end

    θ = atan(y, x)
    t_hat = [-sin(θ),  cos(θ)]
    r_hat = [ cos(θ),  sin(θ)]

    return t_hat, r_hat, θ
end

function angular_distance_to_block(x::Float64, y::Float64, block)
    θ = atan(y, x)
    return wrap_angle_pi(θ - block.theta_center)
end

function active_block_info(
    x::Float64,
    y::Float64;
    forbidden_blocks = forbidden_blocks,
    margin_angle::Float64 = 0.06
)
    θ = atan(y, x)
    θ = θ < 0 ? θ + 2π : θ
    r = hypot(x, y)

    for block in forbidden_blocks
        dθ_signed = wrap_angle_pi(θ - block.theta_center)
        dθ = abs(dθ_signed)

        if dθ <= block.half_angle + margin_angle
            if block.side == "inner"
                block_boundary_r = R_inner + block.radial_depth
                clearance = r - block_boundary_r
            else
                block_boundary_r = R_outer - block.radial_depth
                clearance = block_boundary_r - r
            end

            return (
                active = true,
                side = block.side,
                clearance = clearance,
                dθ_signed = dθ_signed,
                half_angle = block.half_angle,
                theta_center = block.theta_center
            )
        end
    end

    return (
        active = false,
        side = "",
        clearance = Inf,
        dθ_signed = Inf,
        half_angle = 0.0,
        theta_center = 0.0
    )
end

function next_known_block_ahead(
    x::Float64,
    y::Float64;
    forbidden_blocks = forbidden_blocks,
    max_lookahead_angle::Float64 = 1.2
)
    θ = atan(y, x)
    θ = θ < 0 ? θ + 2π : θ
    r = hypot(x, y)

    best = nothing
    best_dθ = Inf

    for (j, block) in enumerate(forbidden_blocks)
        dθ_fwd = mod(block.theta_center - θ, 2π)

        if 0.0 <= dθ_fwd <= max_lookahead_angle
            if dθ_fwd < best_dθ
                best_dθ = dθ_fwd

                if block.side == "inner"
                    block_edge = R_inner + block.radial_depth
                    clearance = r - block_edge
                else
                    block_edge = R_outer - block.radial_depth
                    clearance = block_edge - r
                end

                best = (
                    active = true,
                    block_index = j,
                    side = block.side,
                    clearance = clearance,
                    dθ_forward = dθ_fwd,
                    theta_center = block.theta_center,
                    half_angle = block.half_angle
                )
            end
        end
    end

    if best === nothing
        return (
            active = false,
            block_index = 0,
            side = "",
            clearance = Inf,
            dθ_forward = Inf,
            theta_center = 0.0,
            half_angle = 0.0
        )
    end

    return best
end

function block_clearance_to_active_obstacle(x::Float64, y::Float64)
    info = active_block_info(x, y)

    return info.active ? info.clearance : NaN
end

function debug_radial_decision(
    name::String,
    x::Float64,
    y::Float64,
    u_r::Float64;
    forbidden_blocks = forbidden_blocks,
    margin_angle::Float64 = 0.06,
    clearance_buffer::Float64 = 2.0,
    a_r_lim::Float64 = a_r_max
)
    info = active_block_info(
        x, y;
        forbidden_blocks = forbidden_blocks,
        margin_angle = margin_angle
    )

    ar_min, ar_max = radial_action_limits_from_visible_block(
        x, y;
        forbidden_blocks = forbidden_blocks,
        margin_angle = margin_angle,
        clearance_buffer = clearance_buffer,
        a_r_lim = a_r_lim
    )

    println(
        name,
        " | active=", info.active,
        " | side=", info.side,
        " | clearance=", round(info.clearance, digits = 4),
        " | dθ=", isfinite(info.dθ_signed) ? round(info.dθ_signed, digits = 4) : Inf,
        " | limits=(", round(ar_min, digits = 3), ", ", round(ar_max, digits = 3), ")",
        " | chosen_u_r=", round(u_r, digits = 3)
    )
end

function filter_active_constraints(
    μ::Vector{Float64},
    D::Matrix{Float64},
    d::Vector{Float64};
    tol::Float64 = 0.1
)
    m = size(D, 1)

    if m == 0
        return D, d
    end

    active_idx = Int[]

    for i in 1:m
        residual = D[i, :] ⋅ μ - d[i]

        if residual >= -tol
            push!(active_idx, i)
        end
    end

    if isempty(active_idx)
        return zeros(0, size(D, 2)), zeros(0)
    end

    return D[active_idx, :], d[active_idx]
end

function block_pass_memory_state(
    x::Float64,
    y::Float64,
    block;
    angular_margin::Float64 = 0.10,
    radial_enter_margin::Float64 = 0.75,
    radial_release_margin::Float64 = 1.25
)
    r = hypot(x, y)
    θ = atan(y, x)
    θ = θ < 0 ? θ + 2π : θ

    dθ = abs(wrap_angle_pi(θ - block.theta_center))

    near_sector = dθ <= block.half_angle + angular_margin

    if !near_sector
        return (
            near_sector = false,
            wrong_side = false,
            safely_committed = true
        )
    end

    if block.side == "inner"
        block_edge = R_inner + block.radial_depth

        wrong_side = r < block_edge + radial_enter_margin
        safely_committed = r >= block_edge + radial_release_margin

    elseif block.side == "outer"
        block_edge = R_outer - block.radial_depth

        wrong_side = r > block_edge - radial_enter_margin
        safely_committed = r <= block_edge - radial_release_margin

    else
        error("Unknown block side $(block.side)")
    end

    return (
        near_sector = true,
        wrong_side = wrong_side,
        safely_committed = safely_committed
    )
end

function maybe_activate_block_pass_memory!(
    mem::BlockPassMemory,
    x::Float64,
    y::Float64;
    forbidden_blocks = forbidden_blocks,
    detect_margin::Float64 = 0.10,
    radial_enter_margin::Float64 = 0.75,
    radial_release_margin::Float64 = 1.25
)
    if !use_block_memory[]
        mem.active = false
        mem.block_index = 0
        mem.pass_side = :inward
        return nothing
    end

    if mem.active
        return nothing
    end

    θ = atan(y, x)
    θ = θ < 0 ? θ + 2π : θ

    for (j, block) in enumerate(forbidden_blocks)
        dθ = mod(θ - block.theta_center + π, 2π) - π

        if abs(dθ) <= block.half_angle + detect_margin
            st = block_pass_memory_state(
                x, y, block;
                angular_margin = detect_margin,
                radial_enter_margin = radial_enter_margin,
                radial_release_margin = radial_release_margin
            )

            if st.wrong_side
                if block.side == "inner"
                    mem.active = true
                    mem.block_index = j
                    mem.pass_side = :outward
                    return nothing
                elseif block.side == "outer"
                    mem.active = true
                    mem.block_index = j
                    mem.pass_side = :inward
                    return nothing
                end
            end
        end
    end

    return nothing
end

function maybe_release_block_pass_memory!(
    mem::BlockPassMemory,
    x::Float64,
    y::Float64;
    forbidden_blocks = forbidden_blocks,
    angle_release_margin::Float64 = 0.08,
    radial_release_margin::Float64 = 1.25
)
    if !use_block_memory[]
        mem.active = false
        mem.block_index = 0
        mem.pass_side = :inward
        return nothing
    end

    if !mem.active || mem.block_index == 0
        return nothing
    end

    block = forbidden_blocks[mem.block_index]

    st = block_pass_memory_state(
        x, y, block;
        angular_margin = angle_release_margin,
        radial_enter_margin = 0.75,
        radial_release_margin = radial_release_margin
    )

    # Release only once:
    # 1) we have moved beyond the active sector, OR
    # 2) we are safely committed on the correct side and have passed the trailing edge
    θ = atan(y, x)
    θ = θ < 0 ? θ + 2π : θ

    θ_trailing = mod(block.theta_center + block.half_angle, 2π)
    dθ_after_trailing = mod(θ - θ_trailing, 2π)
    passed_trailing = dθ_after_trailing >= angle_release_margin

    if !st.near_sector || (st.safely_committed && passed_trailing)
        mem.active = false
        mem.block_index = 0
        mem.pass_side = :inward
    end

    return nothing
end

function scenario_state_is_feasible(
    scenario::TwoAgentScenario;
    forbidden_blocks = forbidden_blocks
)
    blue_ok = point_is_feasible(
        scenario.blue_true.x,
        scenario.blue_true.y;
        forbidden_blocks = forbidden_blocks
    )

    red_ok = point_is_feasible(
        scenario.red_true.x,
        scenario.red_true.y;
        forbidden_blocks = forbidden_blocks
    )

    return blue_ok && red_ok
end

function radial_action_limits_with_memory(
    x::Float64,
    y::Float64,
    mem::BlockPassMemory;
    a_r_lim::Float64 = a_r_max,
    clearance_buffer::Float64 = 2.0,
    radial_enter_margin::Float64 = 0.75,
    radial_release_margin::Float64 = 1.25
)
    ar_min_base, ar_max_base = radial_action_limits_from_visible_block(
        x, y;
        forbidden_blocks = forbidden_blocks,
        clearance_buffer = clearance_buffer,
        a_r_lim = a_r_lim
    )

    if !use_block_memory[] || !mem.active || mem.block_index == 0
        return ar_min_base, ar_max_base
    end

    block = forbidden_blocks[mem.block_index]

    st = block_pass_memory_state(
        x, y, block;
        angular_margin = angle_release_margin,
        radial_enter_margin = radial_enter_margin,
        radial_release_margin = radial_release_margin
    )

    # Once we are no longer near the sector, do not constrain
    if !st.near_sector
        return ar_min_base, ar_max_base
    end

    # --------------------------------------------------
    # OUTWARD commitment 
    # --------------------------------------------------
    if mem.pass_side == :outward
        if st.wrong_side
            # still too inward -> must be allowed only outward
            ar_min_mem = max(ar_min_base, 0.0)
            ar_max_mem = ar_max_base
            return ar_min_mem, ar_max_mem
        elseif !st.safely_committed
            # in hysteresis band -> no more outward push required,
            # but also do not allow inward motion yet
            ar_min_mem = max(ar_min_base, 0.0)
            ar_max_mem = min(ar_max_base, 0.0)
            return ar_min_mem, ar_max_mem
        else
            # safely committed -> restore full base freedom
            return ar_min_base, ar_max_base
        end
    end

    # --------------------------------------------------
    # INWARD commitment 
    # --------------------------------------------------
    if mem.pass_side == :inward
        if st.wrong_side
            ar_min_mem = ar_min_base
            ar_max_mem = min(ar_max_base, 0.0)
            return ar_min_mem, ar_max_mem
        elseif !st.safely_committed
            ar_min_mem = max(ar_min_base, 0.0)
            ar_max_mem = min(ar_max_base, 0.0)
            return ar_min_mem, ar_max_mem
        else
            return ar_min_base, ar_max_base
        end
    end

    return ar_min_base, ar_max_base
end

# ============================================================
# TRACK PROJECTION / ARC-LENGTH
# ============================================================

function project_to_track(centerline::Vector{Vector{Float64}}, pos::Vector{Float64})
    min_dist = Inf
    s_progress = 0.0
    total = 0.0

    for i in 1:length(centerline)-1
        p1 = centerline[i]
        p2 = centerline[i+1]

        seg = p2 - p1
        seg_len = norm(seg)

        if seg_len < 1e-12
            continue
        end

        τ = clamp(dot(pos - p1, seg) / seg_len^2, 0.0, 1.0)
        proj = p1 + τ * seg
        dist = norm(pos - proj)

        if dist < min_dist
            min_dist = dist
            s_progress = total + τ * seg_len
        end

        total += seg_len
    end

    return s_progress
end

function compute_track_length(centerline::Vector{Vector{Float64}})
    total = 0.0
    for i in 1:length(centerline)-1
        total += norm(centerline[i+1] - centerline[i])
    end
    return total
end

const s_goal = compute_track_length(centerline)

function add_lookahead_sector!(
    p,
    x::Float64,
    y::Float64,
    vx::Float64,
    vy::Float64;
    H::Int,
    dt::Float64,
    R_in::Float64 = R_inner,
    R_out::Float64 = R_outer,
    n_pts::Int = 60,
    color = :gray60,
    alpha::Float64 = 0.10,
    label = false
)
    r = hypot(x, y)
    if r < 1e-9
        return p
    end

    t_hat, _, θ0 = unit_vectors(x, y)
    v_t = max(vx * t_hat[1] + vy * t_hat[2], 0.0)

    Δθ = (v_t / max(r, 1e-6)) * (H * dt)
    θ1 = θ0
    θ2 = θ0 + Δθ

    θs = range(θ1, θ2; length = n_pts)

    xs = vcat(R_in .* cos.(θs), reverse(R_out .* cos.(θs)))
    ys = vcat(R_in .* sin.(θs), reverse(R_out .* sin.(θs)))

    Plots.plot!(
        p, xs, ys;
        seriestype = :shape,
        color = color,
        fillalpha = alpha,
        linealpha = 0.0,
        label = label
    )

    return p
end

# ============================================================
# LIGHT-ZONE HELPERS
# ============================================================

function in_light_zone(x::Float64, y::Float64, light_zones)
    r = hypot(x, y)
    θ = atan(y, x)
    θ = θ < 0 ? θ + 2π : θ

    for zone in light_zones
        dθ = abs(mod(θ - zone.center + π, 2π) - π)

        if dθ <= zone.half_angle
            if zone.side == "inner"
                if R_inner <= r <= R_inner + light_width
                    return true
                end
            elseif zone.side == "outer"
                if R_outer - light_width <= r <= R_outer
                    return true
                end
            end
        end
    end

    return false
end

function light_zone_radial_target(zone;
    R_inner::Float64 = R_inner,
    R_outer::Float64 = R_outer,
    light_width::Float64 = light_width
)
    if zone.side == "inner"
        return R_inner + 0.5 * light_width
    elseif zone.side == "outer"
        return R_outer - 0.5 * light_width
    else
        error("Unknown light-zone side: $(zone.side)")
    end
end

function next_reachable_light_zone_ahead(
    x::Float64,
    y::Float64;
    light_zones = light_zones,
    max_lookahead_angle::Float64 = 1.20
)
    θ_now = atan(y, x)
    θ_now = θ_now < 0 ? θ_now + 2π : θ_now

    best_zone = nothing
    best_dθ = Inf

    for zone in light_zones
        dθ = mod(zone.center - θ_now, 2π)
        if 0.0 <= dθ <= max_lookahead_angle
            if dθ < best_dθ
                best_dθ = dθ
                best_zone = zone
            end
        end
    end

    return best_zone, best_dθ
end

# ============================================================
# FORBIDDEN-BLOCK / FEASIBILITY GEOMETRY
# ============================================================

function point_in_forbidden_block(
    x::Float64,
    y::Float64;
    forbidden_blocks = forbidden_blocks,
    R_inner::Float64 = R_inner,
    R_outer::Float64 = R_outer
)
    r = hypot(x, y)
    θ = atan(y, x)
    θ = θ < 0 ? θ + 2π : θ

    for block in forbidden_blocks
        dθ = abs(mod(θ - block.theta_center + π, 2π) - π)

        if dθ <= block.half_angle
            if block.side == "inner"
                if R_inner <= r <= R_inner + block.radial_depth
                    return true
                end
            elseif block.side == "outer"
                if R_outer - block.radial_depth <= r <= R_outer
                    return true
                end
            end
        end
    end

    return false
end


function point_is_feasible(
    x::Float64,
    y::Float64;
    forbidden_blocks = forbidden_blocks,
    R_inner::Float64 = R_inner,
    R_outer::Float64 = R_outer
)
    r = hypot(x, y)

    if r < R_inner || r > R_outer
        return false
    end

    if point_in_forbidden_block(
        x, y;
        forbidden_blocks = forbidden_blocks,
        R_inner = R_inner,
        R_outer = R_outer
    )
        return false
    end

    return true
end

function segment_crosses_forbidden(
    x1::Float64, y1::Float64,
    x2::Float64, y2::Float64;
    forbidden_blocks = forbidden_blocks,
    n_samples::Int = 25
)
    for α in range(0.0, 1.0; length = n_samples)
        x = (1 - α) * x1 + α * x2
        y = (1 - α) * y1 + α * y2

        if !point_is_feasible(x, y; forbidden_blocks = forbidden_blocks)
            return true
        end
    end

    return false
end

function is_near_block_sector(x::Float64, y::Float64, block; margin::Float64 = 0.08)
    dθ = abs(angular_distance_to_block(x, y, block))
    return dθ <= block.half_angle + margin
end

function free_radial_corridor(
    θ::Float64;
    forbidden_blocks = forbidden_blocks,
    R_inner::Float64 = R_inner,
    R_outer::Float64 = R_outer,
    inner_margin::Float64 = inner_margin_default,
    outer_margin::Float64 = outer_margin_default,
    block_clearance::Float64 = block_clearance_default
)
    θn = mod(θ, 2π)

    r_low  = R_inner + inner_margin
    r_high = R_outer - outer_margin

    for block in forbidden_blocks
        dθ = abs(mod(θn - block.theta_center + π, 2π) - π)

        if dθ <= block.half_angle
            if block.side == "inner"
                r_low = max(r_low, R_inner + block.radial_depth + block_clearance)
            elseif block.side == "outer"
                r_high = min(r_high, R_outer - block.radial_depth - block_clearance)
            end
        end
    end

    if r_low > r_high
        r_mid = 0.5 * (r_low + r_high)
        r_low = r_mid
        r_high = r_mid
    end

    return r_low, r_high
end

function inner_biased_corridor_target(
    r_low::Float64,
    r_high::Float64;
    β_inner_bias::Float64 = 0.30
)
    return r_low + β_inner_bias * (r_high - r_low)
end

# ============================================================
# EXACT GEOMETRIC REPAIR
# ============================================================

function project_radius_to_annulus(
    x::Float64,
    y::Float64;
    R_inner::Float64 = R_inner,
    R_outer::Float64 = R_outer
)
    r = hypot(x, y)

    if r < 1e-12
        return R_inner, 0.0
    elseif r < R_inner
        α = R_inner / r
        return α * x, α * y
    elseif r > R_outer
        α = R_outer / r
        return α * x, α * y
    else
        return x, y
    end
end

function project_out_of_forbidden_block(
    x::Float64,
    y::Float64;
    forbidden_blocks = forbidden_blocks,
    eps_shift::Float64 = 1e-3
)
    r = hypot(x, y)
    θ = atan(y, x)
    θ = θ < 0 ? θ + 2π : θ

    best_x, best_y = x, y
    best_dist2 = Inf
    found = false

    for block in forbidden_blocks
        dθ_signed = mod(θ - block.theta_center + π, 2π) - π
        dθ = abs(dθ_signed)

        if dθ > block.half_angle
            continue
        end

        if block.side == "inner"
            inside_radial = (R_inner <= r <= R_inner + block.radial_depth)
            if !inside_radial
                continue
            end

            r_rad = R_inner + block.radial_depth + eps_shift
            x1 = r_rad * cos(θ)
            y1 = r_rad * sin(θ)
            d1 = (x1 - x)^2 + (y1 - y)^2
            if d1 < best_dist2 && point_is_feasible(x1, y1; forbidden_blocks = forbidden_blocks)
                best_x, best_y = x1, y1
                best_dist2 = d1
                found = true
            end

            θ_left = block.theta_center - block.half_angle - eps_shift
            x2 = r * cos(θ_left)
            y2 = r * sin(θ_left)
            x2, y2 = project_radius_to_annulus(x2, y2)
            d2 = (x2 - x)^2 + (y2 - y)^2
            if d2 < best_dist2 && point_is_feasible(x2, y2; forbidden_blocks = forbidden_blocks)
                best_x, best_y = x2, y2
                best_dist2 = d2
                found = true
            end

            θ_right = block.theta_center + block.half_angle + eps_shift
            x3 = r * cos(θ_right)
            y3 = r * sin(θ_right)
            x3, y3 = project_radius_to_annulus(x3, y3)
            d3 = (x3 - x)^2 + (y3 - y)^2
            if d3 < best_dist2 && point_is_feasible(x3, y3; forbidden_blocks = forbidden_blocks)
                best_x, best_y = x3, y3
                best_dist2 = d3
                found = true
            end

        elseif block.side == "outer"
            inside_radial = (R_outer - block.radial_depth <= r <= R_outer)
            if !inside_radial
                continue
            end

            r_rad = R_outer - block.radial_depth - eps_shift
            x1 = r_rad * cos(θ)
            y1 = r_rad * sin(θ)
            d1 = (x1 - x)^2 + (y1 - y)^2
            if d1 < best_dist2 && point_is_feasible(x1, y1; forbidden_blocks = forbidden_blocks)
                best_x, best_y = x1, y1
                best_dist2 = d1
                found = true
            end

            θ_left = block.theta_center - block.half_angle - eps_shift
            x2 = r * cos(θ_left)
            y2 = r * sin(θ_left)
            x2, y2 = project_radius_to_annulus(x2, y2)
            d2 = (x2 - x)^2 + (y2 - y)^2
            if d2 < best_dist2 && point_is_feasible(x2, y2; forbidden_blocks = forbidden_blocks)
                best_x, best_y = x2, y2
                best_dist2 = d2
                found = true
            end

            θ_right = block.theta_center + block.half_angle + eps_shift
            x3 = r * cos(θ_right)
            y3 = r * sin(θ_right)
            x3, y3 = project_radius_to_annulus(x3, y3)
            d3 = (x3 - x)^2 + (y3 - y)^2
            if d3 < best_dist2 && point_is_feasible(x3, y3; forbidden_blocks = forbidden_blocks)
                best_x, best_y = x3, y3
                best_dist2 = d3
                found = true
            end
        end
    end

    return found ? (best_x, best_y) : (x, y)
end

function repair_position_to_feasible_world(
    x::Float64,
    y::Float64;
    forbidden_blocks = forbidden_blocks,
    max_iters::Int = 5
)
    xr, yr = x, y

    for _ in 1:max_iters
        xr, yr = project_radius_to_annulus(xr, yr)
        xr, yr = project_out_of_forbidden_block(
            xr, yr;
            forbidden_blocks = forbidden_blocks
        )

        if point_is_feasible(xr, yr; forbidden_blocks = forbidden_blocks)
            break
        end
    end

    return xr, yr
end

# ============================================================
# MULTI-ANCHOR LOCAL CONSTRAINT MODEL FOR PCA-CORRECTION
# ============================================================


function add_inner_circle_constraint_row!(
    rows::Vector{Vector{Float64}},
    vals::Vector{Float64},
    θa::Float64,
    radius_edge::Float64
)
    n = [cos(θa), sin(θa)]
    C = [n[1], n[2], 0.0, 0.0]

    # feasible: C⋅x - radius_edge >= 0
    push!(rows, C)
    push!(vals, radius_edge)

    return nothing
end

function add_outer_circle_constraint_row!(
    rows::Vector{Vector{Float64}},
    vals::Vector{Float64},
    θa::Float64,
    radius_edge::Float64
)
    n = [cos(θa), sin(θa)]
    C = [-n[1], -n[2], 0.0, 0.0]

    # feasible: -n⋅x + radius_edge >= 0
    # written as D*x - d >= 0, so d = -radius_edge
    push!(rows, C)
    push!(vals, -radius_edge)

    return nothing
end

function build_local_constraint_model(
    μ::Vector{Float64};
    wall_activation_margin::Float64 = pca_settings[].wall_activation_margin,
    block_angle_margin::Float64 = pca_settings[].block_angle_margin,
    block_radial_margin::Float64 = pca_settings[].block_radial_margin,
    forbidden_blocks = forbidden_blocks,
    n_wall_anchors::Int = N_WALL_ANCHORS,
    n_block_anchors::Int = N_BLOCK_ANCHORS,
    wall_anchor_span::Float64 = WALL_ANCHOR_SPAN,
    block_anchor_span::Float64 = BLOCK_ANCHOR_SPAN
)
    x = μ[1]
    y = μ[2]
    r = hypot(x, y)

    if r < 1e-10
        return zeros(0, 4), Float64[]
    end

    θ = atan(y, x)
    θ = θ < 0 ? θ + 2π : θ

    rows = Vector{Vector{Float64}}()
    vals = Float64[]

    # --------------------------------------------------
    # Inner wall: same activation as before,
    # but several nearby local tangent anchors.
    # --------------------------------------------------
    if r - R_inner <= wall_activation_margin
        θs = range(
            θ - wall_anchor_span,
            θ + wall_anchor_span;
            length = n_wall_anchors
        )

        for θa in θs
            add_inner_circle_constraint_row!(rows, vals, θa, R_inner)
        end
    end

    # --------------------------------------------------
    # Outer wall: same activation as before,
    # but several nearby local tangent anchors.
    # --------------------------------------------------
    if R_outer - r <= wall_activation_margin
        θs = range(
            θ - wall_anchor_span,
            θ + wall_anchor_span;
            length = n_wall_anchors
        )

        for θa in θs
            add_outer_circle_constraint_row!(rows, vals, θa, R_outer)
        end
    end

    # --------------------------------------------------
    # Forbidden blocks:
    # IMPORTANT:
    # activation remains local around the current angle θ,
    # not across the whole block sector.
    # --------------------------------------------------
    for block in forbidden_blocks
        dθ = abs(wrap_angle_pi(θ - block.theta_center))

        if dθ > block.half_angle + block_angle_margin
            continue
        end

        if block.side == "inner"
            block_edge = R_inner + block.radial_depth
            clearance = r - block_edge

            if clearance <= block_radial_margin
                θs = range(
                    θ - block_anchor_span,
                    θ + block_anchor_span;
                    length = n_block_anchors
                )

                for θa in θs
                    add_inner_circle_constraint_row!(rows, vals, θa, block_edge)
                end
            end

        elseif block.side == "outer"
            block_edge = R_outer - block.radial_depth
            clearance = block_edge - r

            if clearance <= block_radial_margin
                θs = range(
                    θ - block_anchor_span,
                    θ + block_anchor_span;
                    length = n_block_anchors
                )

                for θa in θs
                    add_outer_circle_constraint_row!(rows, vals, θa, block_edge)
                end
            end
        end
    end

    if isempty(rows)
        return zeros(0, 4), Float64[]
    end

    D = reduce(vcat, [reshape(row, 1, :) for row in rows])
    d = copy(vals)

    return D, d
end

function positive_finite_pca_value(x::Float64; tol::Float64 = 1e-12)
    return isfinite(x) && x > tol
end

# ============================================================
# INITIALIZATION ON TRACK
# ============================================================

function make_agent_on_track(
    θ0::Float64,
    r0::Float64,
    v_t0::Float64
)
    x0 = r0 * cos(θ0)
    y0 = r0 * sin(θ0)

    t_hat, _, _ = unit_vectors(x0, y0)
    vx0 = v_t0 * t_hat[1]
    vy0 = v_t0 * t_hat[2]

    s0 = project_to_track(centerline, [x0, y0])

    return AgentState(x0, y0, vx0, vy0, s0)
end

# ============================================================
# CORE CURVILINEAR DYNAMICS
# ============================================================

function propagate_mean_curvilinear(
    x::Float64,
    y::Float64,
    vx::Float64,
    vy::Float64,
    a_t::Float64,
    a_r::Float64,
    dt::Float64,
    model::CarModel
)
    r = hypot(x, y)

    if r < 1e-12
        return x, y, 0.0, 0.0
    end

    t_hat, r_hat, θ = unit_vectors(x, y)

    v_t = vx * t_hat[1] + vy * t_hat[2]
    v_r = vx * r_hat[1] + vy * r_hat[2]

    v_t_new = v_t + (a_t - model.c_drag * v_t) * dt
    v_r_new = v_r + (a_r - model.c_slip * v_r) * dt

    v_t_new = max(v_t_new, 0.0)

    r_new = r + v_r_new * dt
    θ_new = θ + (v_t_new / max(r, 1e-6)) * dt

    x_new = r_new * cos(θ_new)
    y_new = r_new * sin(θ_new)

    t_hat_new, r_hat_new, _ = unit_vectors(x_new, y_new)

    vx_new = v_t_new * t_hat_new[1] + v_r_new * r_hat_new[1]
    vy_new = v_t_new * t_hat_new[2] + v_r_new * r_hat_new[2]

    return x_new, y_new, vx_new, vy_new
end

# ============================================================
# TRUE-STATE PROPAGATION
# ============================================================

function propagate_state!(
    agent::AgentState,
    a_t::Float64,
    a_r::Float64,
    dt::Float64,
    model::CarModel
)
    x_old, y_old = agent.x, agent.y
    vx_old, vy_old = agent.vx, agent.vy

    # --------------------------------
    # Truth-side process noise
    # --------------------------------
    a_t_true = a_t + true_accel_noise_t * randn()
    a_r_true = a_r + true_accel_noise_r * randn()

    x_new, y_new, vx_new, vy_new = propagate_mean_curvilinear(
        x_old, y_old, vx_old, vy_old, a_t_true, a_r_true, dt, model
    )

    vx_new += true_vel_noise_xy * randn()
    vy_new += true_vel_noise_xy * randn()

    r_new = hypot(x_new, y_new)

    if r_new < 1e-12
        x_new, y_new = x_old, y_old
        vx_new, vy_new = 0.0, 0.0
        r_new = hypot(x_new, y_new)
    end

    # --------------------------------
    # Hard annulus enforcement
    # --------------------------------
    if r_new < R_inner || r_new > R_outer
        nx, ny = x_new / r_new, y_new / r_new
        r_clip = clamp(r_new, R_inner, R_outer)

        x_new = r_clip * nx
        y_new = r_clip * ny

        t_hat_b, r_hat_b, _ = unit_vectors(x_new, y_new)
        v_t_b = vx_new * t_hat_b[1] + vy_new * t_hat_b[2]
        v_t_b = max(v_t_b, 0.0)
        v_r_b = 0.0

        vx_new = v_t_b * t_hat_b[1] + v_r_b * r_hat_b[1]
        vy_new = v_t_b * t_hat_b[2] + v_r_b * r_hat_b[2]
    end

    # --------------------------------
    # Forbidden-block crossing repair
    # --------------------------------
    crossed_forbidden =
        !point_is_feasible(x_new, y_new; forbidden_blocks = forbidden_blocks) ||
        segment_crosses_forbidden(
            x_old, y_old, x_new, y_new;
            forbidden_blocks = forbidden_blocks,
            n_samples = 30
        )

    if crossed_forbidden
        α_low = 0.0
        α_high = 1.0
        x_feas, y_feas = x_old, y_old

        for _ in 1:20
            α_mid = 0.5 * (α_low + α_high)

            x_mid = (1 - α_mid) * x_old + α_mid * x_new
            y_mid = (1 - α_mid) * y_old + α_mid * y_new

            crossed_mid =
                !point_is_feasible(x_mid, y_mid; forbidden_blocks = forbidden_blocks) ||
                segment_crosses_forbidden(
                    x_old, y_old, x_mid, y_mid;
                    forbidden_blocks = forbidden_blocks,
                    n_samples = 20
                )

            if crossed_mid
                α_high = α_mid
            else
                α_low = α_mid
                x_feas, y_feas = x_mid, y_mid
            end
        end

        x_new, y_new = x_feas, y_feas

        t_hat_b, r_hat_b, _ = unit_vectors(x_new, y_new)
        v_t_b = vx_new * t_hat_b[1] + vy_new * t_hat_b[2]
        v_t_b = max(v_t_b, 0.0)
        v_r_b = 0.0

        vx_new = v_t_b * t_hat_b[1] + v_r_b * r_hat_b[1]
        vy_new = v_t_b * t_hat_b[2] + v_r_b * r_hat_b[2]
    end

    # --------------------------------
    # Final no-backward enforcement
    # --------------------------------
    t_hat_final, _, _ = unit_vectors(x_new, y_new)
    v_t_final = vx_new * t_hat_final[1] + vy_new * t_hat_final[2]

    if v_t_final < 0
        vx_new -= v_t_final * t_hat_final[1]
        vy_new -= v_t_final * t_hat_final[2]
        v_t_final = 0.0
    end

    # --------------------------------
    # Commit
    # --------------------------------
    agent.x = x_new
    agent.y = y_new
    agent.vx = vx_new
    agent.vy = vy_new
    agent.s += v_t_final * dt

    return nothing
end

function propagate_two_agent_truth!(
    scenario::TwoAgentScenario,
    u_blue::Vector{Float64},
    u_red::Vector{Float64},
    model_blue::CarModel,
    model_red::CarModel
)
    propagate_state!(scenario.blue_true, u_blue[1], u_blue[2], dt, model_blue)
    propagate_state!(scenario.red_true,  u_red[1],  u_red[2],  dt, model_red)
    return nothing
end

function propagate_state_planner_feasible(
    agent::AgentState,
    a_t::Float64,
    a_r::Float64,
    dt::Float64,
    model::CarModel
)
    x_old, y_old = agent.x, agent.y
    vx_old, vy_old = agent.vx, agent.vy

    x_new, y_new, vx_new, vy_new = propagate_mean_curvilinear(
        x_old, y_old, vx_old, vy_old, a_t, a_r, dt, model
    )

    r_new = hypot(x_new, y_new)

    if r_new < 1e-12
        x_new, y_new = x_old, y_old
        vx_new, vy_new = 0.0, 0.0
        r_new = hypot(x_new, y_new)
    end

    # deterministic annulus enforcement
    if r_new < R_inner || r_new > R_outer
        nx, ny = x_new / r_new, y_new / r_new
        r_clip = clamp(r_new, R_inner, R_outer)

        x_new = r_clip * nx
        y_new = r_clip * ny

        t_hat_b, r_hat_b, _ = unit_vectors(x_new, y_new)
        v_t_b = vx_new * t_hat_b[1] + vy_new * t_hat_b[2]
        v_t_b = max(v_t_b, 0.0)
        v_r_b = 0.0

        vx_new = v_t_b * t_hat_b[1] + v_r_b * r_hat_b[1]
        vy_new = v_t_b * t_hat_b[2] + v_r_b * r_hat_b[2]
    end

    crossed_forbidden =
        !point_is_feasible(x_new, y_new; forbidden_blocks = forbidden_blocks) ||
        segment_crosses_forbidden(
            x_old, y_old, x_new, y_new;
            forbidden_blocks = forbidden_blocks,
            n_samples = 30
        )

    if crossed_forbidden
        α_low = 0.0
        α_high = 1.0
        x_feas, y_feas = x_old, y_old

        for _ in 1:20
            α_mid = 0.5 * (α_low + α_high)

            x_mid = (1 - α_mid) * x_old + α_mid * x_new
            y_mid = (1 - α_mid) * y_old + α_mid * y_new

            crossed_mid =
                !point_is_feasible(x_mid, y_mid; forbidden_blocks = forbidden_blocks) ||
                segment_crosses_forbidden(
                    x_old, y_old, x_mid, y_mid;
                    forbidden_blocks = forbidden_blocks,
                    n_samples = 20
                )

            if crossed_mid
                α_high = α_mid
            else
                α_low = α_mid
                x_feas, y_feas = x_mid, y_mid
            end
        end

        x_new, y_new = x_feas, y_feas

        t_hat_b, r_hat_b, _ = unit_vectors(x_new, y_new)
        v_t_b = vx_new * t_hat_b[1] + vy_new * t_hat_b[2]
        v_t_b = max(v_t_b, 0.0)
        v_r_b = 0.0

        vx_new = v_t_b * t_hat_b[1] + v_r_b * r_hat_b[1]
        vy_new = v_t_b * t_hat_b[2] + v_r_b * r_hat_b[2]
    end

    t_hat_final, _, _ = unit_vectors(x_new, y_new)
    v_t_final = vx_new * t_hat_final[1] + vy_new * t_hat_final[2]

    if v_t_final < 0
        vx_new -= v_t_final * t_hat_final[1]
        vy_new -= v_t_final * t_hat_final[2]
        v_t_final = 0.0
    end

    return AgentState(
        x_new,
        y_new,
        vx_new,
        vy_new,
        agent.s + v_t_final * dt
    )
end

function propagate_two_agent_truth_unrepaired!(
    scenario::TwoAgentScenario,
    u_blue::Vector{Float64},
    u_red::Vector{Float64},
    model_blue::CarModel,
    model_red::CarModel
)
    scenario.blue_true = propagate_state_planner_feasible(
        scenario.blue_true, u_blue[1], u_blue[2], dt, model_blue
    )

    scenario.red_true = propagate_state_planner_feasible(
        scenario.red_true, u_red[1], u_red[2], dt, model_red
    )

    return nothing
end

# ============================================================
# BELIEF PREDICTION
# ============================================================

function belief_predict!(
    belief::AgentBelief,
    a_t::Float64,
    a_r::Float64,
    dt::Float64,
    model::CarModel
)
    μ = copy(belief.mean)
    x, y, vx, vy = μ

    # --------------------------------
    # Mean propagation
    # --------------------------------
    x_new, y_new, vx_new, vy_new = propagate_mean_curvilinear(
        x, y, vx, vy, a_t, a_r, dt, model
    )

    belief.mean .= [x_new, y_new, vx_new, vy_new]

    # --------------------------------
    # Numerical Jacobian
    # --------------------------------
    h = 1e-5
    A = zeros(4, 4)
    x0 = [x, y, vx, vy]

    for i in 1:4
        dx = zeros(4)
        dx[i] = h

        xp = propagate_mean_curvilinear(
            x0[1] + dx[1], x0[2] + dx[2], x0[3] + dx[3], x0[4] + dx[4],
            a_t, a_r, dt, model
        )

        xm = propagate_mean_curvilinear(
            x0[1] - dx[1], x0[2] - dx[2], x0[3] - dx[3], x0[4] - dx[4],
            a_t, a_r, dt, model
        )

        fp = collect(xp)
        fm = collect(xm)

        A[:, i] = (fp - fm) / (2h)
    end

    # --------------------------------
    # Process noise
    # --------------------------------
    vmag = hypot(vx, vy)
    q_vel = (q_vel_base + q_vel_gain * vmag)^2
    Q = Matrix(Diagonal([q_pos, q_pos, q_vel, q_vel]))

    # --------------------------------
    # Covariance propagation
    # --------------------------------
    P = A * belief.cov * A' + Q
    P = 0.5 .* (P + P')
    P += 1e-9 * I
    belief.cov .= P

    return nothing
end

function belief_predict_mode!(
    belief::AgentBelief,
    a_t::Float64,
    a_r::Float64,
    dt::Float64,
    model::CarModel
)
    belief_predict!(belief, a_t, a_r, dt, model)
    return apply_constraint_estimator_pipeline!(belief)
end

function belief_predict_feasible!(
    belief::AgentBelief,
    a_t::Float64,
    a_r::Float64,
    dt::Float64,
    model::CarModel
)
    return belief_predict_mode!(belief, a_t, a_r, dt, model)
end


function predict_two_agent_beliefs!(
    scenario::TwoAgentScenario,
    u_blue::Vector{Float64},
    u_red::Vector{Float64},
    model_blue::CarModel,
    model_red::CarModel
)
    diag_blue_self = belief_predict_feasible!(
        scenario.blue_self_belief, u_blue[1], u_blue[2], dt, model_blue
    )
    last_pca_pred_blue_self[] = diag_blue_self

    diag_red_self = belief_predict_feasible!(
        scenario.red_self_belief, u_red[1], u_red[2], dt, model_red
    )
    last_pca_pred_red_self[] = diag_red_self

    diag_blue_on_red = belief_predict_feasible!(
        scenario.blue_belief_on_red, u_red[1], u_red[2], dt, model_red
    )
    last_pca_pred_blue_on_red[] = diag_blue_on_red

    diag_red_on_blue = belief_predict_feasible!(
        scenario.red_belief_on_blue, u_blue[1], u_blue[2], dt, model_blue
    )
    last_pca_pred_red_on_blue[] = diag_red_on_blue

    return nothing
end

# ============================================================
# PLANNING-ONLY BELIEF PREDICTION
# ============================================================

function one_step_predict_scenario_for_planning(
    scenario::TwoAgentScenario,
    u_blue::Vector{Float64},
    u_red::Vector{Float64},
    model_blue::CarModel,
    model_red::CarModel
)
    sc = deepcopy(scenario)

    # deterministic nominal truth propagation for planning
    propagate_two_agent_truth_unrepaired!(sc, u_blue, u_red, model_blue, model_red)

    # deterministic nominal belief propagation for planning
    predict_two_agent_beliefs_nominal!(sc, u_blue, u_red, model_blue, model_red)

    # deterministic covariance-only visibility update
    belief_update_nominal_visible_only_mode!(sc.blue_self_belief)
    belief_update_nominal_visible_only_mode!(sc.red_self_belief)
    belief_update_nominal_visible_only_mode!(sc.blue_belief_on_red)
    belief_update_nominal_visible_only_mode!(sc.red_belief_on_blue)

    synchronize_first_order_beliefs!(sc)

    return sc
end

# ============================================================
# MEASUREMENT MODEL
# ============================================================

function measurement_matrix_position()
    return [
        1.0 0.0 0.0 0.0
        0.0 1.0 0.0 0.0
    ]
end

function measurement_noise_light()
    return Matrix(Diagonal([meas_std_light^2, meas_std_light^2]))
end

function sample_position_measurement_if_visible(state::AgentState)
    visible = in_light_zone(state.x, state.y, light_zones)

    if !visible
        return nothing, nothing, false
    end

    R = measurement_noise_light()
    z = [state.x, state.y] .+ randn(2) .* sqrt.(diag(R))

    return z, R, true
end

# ============================================================
# EKF UPDATE
# ============================================================

function belief_update_ekf!(
    belief::AgentBelief,
    z::Vector{Float64},
    R::Matrix{Float64}
)
    H = measurement_matrix_position()

    μ = belief.mean
    P = belief.cov

    S = H * P * H' + R
    K = P * H' * inv(S)

    innovation = z - H * μ
    μ_post = μ + K * innovation

    I4 = Matrix{Float64}(I, 4, 4)
    KH = K * H

    P_post = (I4 - KH) * P * (I4 - KH)' + K * R * K'
    P_post = 0.5 * (P_post + P_post')
    P_post += 1e-9 * I

    belief.mean .= μ_post
    belief.cov  .= P_post

    return nothing
end

function belief_update_nominal_visible_only_mode!(
    belief::AgentBelief;
    R_light::Matrix{Float64} = Matrix(Diagonal([meas_std_light^2, meas_std_light^2]))
)
    got_nominal_visible = belief_update_nominal_visible_only!(belief; R_light = R_light)

    if got_nominal_visible
        diag = apply_constraint_estimator_pipeline!(belief)
    else
        diag = empty_pca_diagnostics()
    end

    return got_nominal_visible, diag
end

# ============================================================
# PCA-AUGMENTED CONSTRAINED POSTERIOR
# ============================================================

function use_gupta_estimator!()
    estimator_mode[] = :gupta
    return nothing
end

function use_gupta_pca_full_estimator!()
    estimator_mode[] = :gupta_pca_full
    return nothing
end

function use_gupta_pca_trunc95_estimator!()
    estimator_mode[] = :gupta_pca_trunc95
    return nothing
end

function use_gupta_pca_full_exact_projection_estimator!()
    estimator_mode[] = :gupta_pca_full_ep
    return nothing
end

function use_gupta_pca_trunc95_exact_projection_estimator!()
    estimator_mode[] = :gupta_pca_trunc95_ep
    return nothing
end

function safe_sym_inv(M::AbstractMatrix{<:Float64}; reg::Float64 = 1e-9)
    Ms = 0.5 .* (Matrix(M) + Matrix(M)')
    n = size(Ms, 1)
    return inv(Symmetric(Ms + reg * Matrix{Float64}(I, n, n)))
end

function project_to_psd(P::Matrix{Float64}; eps::Float64 = 1e-9)
    Ps = 0.5 .* (P + P')
    E = eigen(Symmetric(Ps))
    λ = max.(E.values, eps)
    Ppsd = E.vectors * Diagonal(λ) * E.vectors'
    return 0.5 .* (Ppsd + Ppsd')
end

function gupta_constrained_posterior(
    μ::Vector{Float64},
    P::Matrix{Float64},
    D::Matrix{Float64},
    d::Vector{Float64};
    reg::Float64 = 1e-9
)
    m = size(D, 1)

    if m == 0
        return copy(μ), 0.5 .* (P + P')
    end

    P_sym = 0.5 .* (P + P')
    S = D * P_sym * D'
    condS = cond(S + reg * I(size(S,1)))

    if condS > 1e10
        return copy(μ), 0.5 .* (P + P')   # skip constraint update
    end

    S_inv = safe_sym_inv(S; reg = reg)

    innovation = D * μ - d

    μc = μ - P_sym * D' * (S_inv * innovation)

    Pc = P_sym - P_sym * D' * (S_inv * D * P_sym)
    Pc = 0.5 .* (Pc + Pc')
    Pc += 1e-9 * I

    return μc, Pc
end

function exact_projection_mean_only(
    μ::Vector{Float64},
    P::Matrix{Float64},
    D::Matrix{Float64},
    d::Vector{Float64};
    reg::Float64 = 1e-9
)
    m = size(D, 1)

    if m == 0
        return copy(μ), project_to_psd(P)
    end

    P_sym = 0.5 .* (P + P')
    S = D * P_sym * D'
    S_inv = safe_sym_inv(S; reg = reg)

    μ_ep = μ - P_sym * D' * (S_inv * (D * μ - d))

    # Mean-only exact projection: keep covariance unchanged
    P_ep = project_to_psd(P_sym)

    return μ_ep, P_ep
end

function current_pca_settings_for_mode()
    s = pca_settings[]

    if estimator_mode[] == :gupta_pca_full
        return PCAConstraintSettings(
            1.0,
            nothing,
            s.weighting_mode,
            s.alpha,
            s.eps_eig,
            false,   # soft PCA only
            s.wall_activation_margin,
            s.block_angle_margin,
            s.block_radial_margin
        )

    elseif estimator_mode[] == :gupta_pca_trunc95
        return PCAConstraintSettings(
            0.95,
            nothing,
            s.weighting_mode,
            s.alpha,
            s.eps_eig,
            false,   # soft PCA only
            s.wall_activation_margin,
            s.block_angle_margin,
            s.block_radial_margin
        )

    elseif estimator_mode[] == :gupta_pca_full_ep
        return PCAConstraintSettings(
            1.0,
            nothing,
            s.weighting_mode,
            s.alpha,
            s.eps_eig,
            true,    # soft PCA + exact projection
            s.wall_activation_margin,
            s.block_angle_margin,
            s.block_radial_margin
        )

    elseif estimator_mode[] == :gupta_pca_trunc95_ep
        return PCAConstraintSettings(
            0.95,
            nothing,
            s.weighting_mode,
            s.alpha,
            s.eps_eig,
            true,    # soft PCA + exact projection
            s.wall_activation_margin,
            s.block_angle_margin,
            s.block_radial_margin
        )

    else
        return s
    end
end

function pca_augmented_constrained_posterior(
    μ::Vector{Float64},
    P::Matrix{Float64},
    D::Matrix{Float64},
    d::Vector{Float64},
    settings::PCAConstraintSettings
)
    p = size(D, 1)

    if p == 0
        diag = PCAConstraintDiagnostics(
            false,
            Float64[],
            zeros(0, 0),
            Float64[],
            0,
            zeros(0, 0)
        )
        return copy(μ), 0.5 .* (P + P'), diag
    end

    P_sym = 0.5 .* (P + P')
    μr = D * μ - d
    Σr = D * P_sym * D'
    Σr = 0.5 .* (Σr + Σr')

    E = eigen(Symmetric(Σr))
    λ = collect(E.values)
    U = collect(E.vectors)

    perm = sortperm(λ; rev = true)
    λ = λ[perm]
    U = U[:, perm]

    λ = max.(λ, 0.0)

    q = 0
    if settings.fixed_rank !== nothing
        q = min(settings.fixed_rank, length(λ))
    else
        total_var = sum(λ)
        if total_var > settings.eps_eig
            cum_ratio = cumsum(λ) ./ total_var
            q_found = findfirst(x -> x >= settings.retained_variance, cum_ratio)
            q = q_found === nothing ? length(λ) : q_found
        end
    end

    if q == 0
        diag = PCAConstraintDiagnostics(
            false,
            μr,
            Σr,
            λ,
            0,
            zeros(p, p)
        )
        return copy(μ), 0.5 .* (P + P'), diag
    end

    Uq = U[:, 1:q]
    λq = λ[1:q]

    ω = zeros(q)

    if settings.weighting_mode == :risk
        for i in 1:q
            ω[i] = settings.alpha / (λq[i] + settings.eps_eig)
        end
    elseif settings.weighting_mode == :inverse
        for i in 1:q
            ω[i] = 1.0 / (λq[i] + settings.eps_eig)
        end
    else
        error("Unknown weighting mode $(settings.weighting_mode)")
    end

    Ω = Diagonal(ω)
    M = Uq * Ω * Uq'

    # Eq. (3333)
    P_inv = safe_sym_inv(P_sym; reg = settings.eps_eig)
    Hcorr = P_inv + D' * M * D
    Hcorr = 0.5 .* (Hcorr + Hcorr')

    P_pca = inv(Symmetric(Hcorr))
    P_pca = 0.5 .* (P_pca + P_pca')

    # Eq. (2222)
    μ_pca = μ - P_pca * D' * M * (D * μ - d)

    diag = PCAConstraintDiagnostics(
        true,
        μr,
        Σr,
        λ,
        q,
        M
    )

    return copy(μ_pca), project_to_psd(P_pca), diag
end

function belief_update_with_optional_pca!(
    belief::AgentBelief,
    z::Vector{Float64},
    R::Matrix{Float64}
)
    belief_update_ekf!(belief, z, R)
    return apply_constraint_estimator_pipeline!(belief)
end

function apply_constraint_estimator_pipeline!(
    belief::AgentBelief
)
    μ0 = copy(belief.mean)
    P0 = 0.5 .* (belief.cov + belief.cov')

    # Build local constraint model and keep only active / near-active constraints
    D_full, d_full = build_local_constraint_model(μ0)
    D, d = filter_active_constraints(μ0, D_full, d_full)

    if size(D, 1) == 0
        belief.mean .= μ0
        belief.cov  .= project_to_psd(P0)
        return empty_pca_diagnostics()
    end

    # --------------------------------------------------
    # Classical constrained baseline
    # --------------------------------------------------
    if estimator_mode[] == :gupta
        μg, Pg = gupta_constrained_posterior(μ0, P0, D, d)
        Pg = COV_INFLATION[] * Pg

        belief.mean .= copy(μg)
        belief.cov  .= project_to_psd(Pg)

        return empty_pca_diagnostics()
    end

    # --------------------------------------------------
    # PCA-based soft constrained estimator
    # IMPORTANT: this branch starts from the unconstrained posterior (μ0,P0),
    # not from the Gupta output.
    # --------------------------------------------------
    if estimator_mode[] in (
        :gupta_pca_full,
        :gupta_pca_trunc95,
        :gupta_pca_full_ep,
        :gupta_pca_trunc95_ep
    )
        settings_eff = current_pca_settings_for_mode()

        μp, Pp, diag = pca_augmented_constrained_posterior(
            μ0, P0, D, d, settings_eff
        )

        # Optional exact projection stage (mean-only)
        if settings_eff.exact_projection
            μp, Pp = exact_projection_mean_only(μp, Pp, D, d)
        end

        μp = copy(μp)
        Pp = project_to_psd(Pp)

        Pp = COV_INFLATION[] * Pp

        belief.mean .= μp
        belief.cov  .= project_to_psd(Pp)

        if PCA_PRINT_DEBUG[] && (sim_step[] % PCA_PRINT_EVERY[] == 0)
            println(
                "Estimator pipeline | step=", sim_step[],
                " | mode=", estimator_mode[],
                " | n_constraints=", size(D, 1),
                " | retained_rank=", diag.retained_rank,
                " | exact_projection=", settings_eff.exact_projection
            )
        end

        return diag
    end

    error("Unknown estimator_mode $(estimator_mode[])")
end

# ============================================================
# DETERMINISTIC NOMINAL BELIEF UPDATE FOR PLANNING
# ============================================================

function belief_update_nominal_visible_only!(
    belief::AgentBelief;
    R_light::Matrix{Float64} = Matrix(Diagonal([meas_std_light^2, meas_std_light^2]))
)
    x = belief.mean[1]
    y = belief.mean[2]

    if !in_light_zone(x, y, light_zones)
        return false
    end

    H = measurement_matrix_position()

    S = H * belief.cov * H' + R_light
    K = belief.cov * H' * inv(S)

    # zero innovation: nominal measurement equals predicted mean
    I4 = Matrix{Float64}(I, 4, 4)
    KH = K * H

    P_post = (I4 - KH) * belief.cov * (I4 - KH)' + K * R_light * K'
    P_post = 0.5 * (P_post + P_post')

    belief.cov .= P_post

    return true
end


# ============================================================
# NOMINAL / DETERMINISTIC BELIEF PROPAGATION FOR PLANNING
# ============================================================

function predict_two_agent_beliefs_nominal!(
    scenario::TwoAgentScenario,
    u_blue::Vector{Float64},
    u_red::Vector{Float64},
    model_blue::CarModel,
    model_red::CarModel
)
    belief_predict_mode!(scenario.blue_self_belief,   u_blue[1], u_blue[2], dt, model_blue)
    belief_predict_mode!(scenario.red_self_belief,    u_red[1],  u_red[2],  dt, model_red)

    belief_predict_mode!(scenario.blue_belief_on_red, u_red[1],  u_red[2],  dt, model_red)
    belief_predict_mode!(scenario.red_belief_on_blue, u_blue[1], u_blue[2], dt, model_blue)

    return nothing
end

# ============================================================
# FIRST-ORDER BELIEF SYNCHRONIZATION
# paper-consistent bookkeeping:
# each agent's belief about itself should define the "self" state used in planning,
# while cross-beliefs remain separate estimates of the opponent.
# This helper keeps all belief objects feasible and numerically clean.
# ============================================================

function synchronize_first_order_beliefs!(scenario::TwoAgentScenario)
    scenario.blue_self_belief.cov .= 0.5 .* (
        scenario.blue_self_belief.cov + scenario.blue_self_belief.cov'
    )

    scenario.red_self_belief.cov .= 0.5 .* (
        scenario.red_self_belief.cov + scenario.red_self_belief.cov'
    )

    scenario.blue_belief_on_red.cov .= 0.5 .* (
        scenario.blue_belief_on_red.cov + scenario.blue_belief_on_red.cov'
    )

    scenario.red_belief_on_blue.cov .= 0.5 .* (
        scenario.red_belief_on_blue.cov + scenario.red_belief_on_blue.cov'
    )

    return nothing
end



# ============================================================
# OPTIONAL SELF-LOCALIZATION UPDATE
# ============================================================

function self_localization_update!(
    belief::AgentBelief,
    true_state::AgentState;
    meas_std::Float64 = meas_std_light
)
    R = Matrix(Diagonal([meas_std^2, meas_std^2]))
    z = [true_state.x, true_state.y]

    belief_update_ekf!(belief, z, R)
    apply_constraint_estimator_pipeline!(belief)

    return nothing
end

# ============================================================
# VISIBILITY-GATED SELF / CROSS UPDATES
# ============================================================

function cross_belief_update_if_visible!(
    observer_belief_on_target::AgentBelief,
    true_target::AgentState,
    diag_store::Base.RefValue{PCAConstraintDiagnostics}
)
    z, R, got_measurement = sample_position_measurement_if_visible(true_target)

    if got_measurement
        diag = belief_update_with_optional_pca!(observer_belief_on_target, z, R)
        diag_store[] = diag
    else
        diag_store[] = empty_pca_diagnostics()
    end

    return got_measurement, nothing
end

function self_belief_update_if_visible!(
    belief::AgentBelief,
    true_state::AgentState,
    diag_store::Base.RefValue{PCAConstraintDiagnostics}
)
    z, R, got_measurement = sample_position_measurement_if_visible(true_state)

    if got_measurement
        diag = belief_update_with_optional_pca!(belief, z, R)
        diag_store[] = diag
    else
        diag_store[] = empty_pca_diagnostics()
    end

    return got_measurement, nothing
end

# ============================================================
# TWO-AGENT STAGE 10 BELIEF UPDATE
# ============================================================

function update_two_agent_beliefs_stage10!(
    scenario::TwoAgentScenario
)
    got_blue_self, _ = self_belief_update_if_visible!(
        scenario.blue_self_belief,
        scenario.blue_true,
        last_pca_upd_blue_self
    )

    got_red_self, _ = self_belief_update_if_visible!(
        scenario.red_self_belief,
        scenario.red_true,
        last_pca_upd_red_self
    )

    got_blue_on_red, _ = cross_belief_update_if_visible!(
        scenario.blue_belief_on_red,
        scenario.red_true,
        last_pca_upd_blue_on_red
    )

    got_red_on_blue, _ = cross_belief_update_if_visible!(
        scenario.red_belief_on_blue,
        scenario.blue_true,
        last_pca_upd_red_on_blue
    )

    return got_blue_self, got_red_self, got_blue_on_red, got_red_on_blue
end

# ============================================================
# FULL ONE-STEP STAGE 10 SCENARIO PROPAGATION
# ============================================================

function one_step_predict_scenario(
    scenario::TwoAgentScenario,
    u_blue::Vector{Float64},
    u_red::Vector{Float64},
    model_blue::CarModel,
    model_red::CarModel
)
    sc = deepcopy(scenario)

    propagate_two_agent_truth!(sc, u_blue, u_red, model_blue, model_red)
    predict_two_agent_beliefs!(sc, u_blue, u_red, model_blue, model_red)
    update_two_agent_beliefs_stage10!(sc)

    return sc
end

# ============================================================
# ONE-STEP NOMINAL PLANNING PROPAGATION
# ============================================================

function one_step_nominal_two_player!(
    scenario::TwoAgentScenario,
    u_blue::Vector{Float64},
    u_red::Vector{Float64},
    model_blue::CarModel,
    model_red::CarModel
)
    # deterministic nominal truth propagation
    propagate_two_agent_truth_unrepaired!(scenario, u_blue, u_red, model_blue, model_red)

    # deterministic nominal belief propagation
    predict_two_agent_beliefs_nominal!(scenario, u_blue, u_red, model_blue, model_red)

    # deterministic covariance-only nominal visibility update
    belief_update_nominal_visible_only_mode!(scenario.blue_self_belief)
    belief_update_nominal_visible_only_mode!(scenario.red_self_belief)
    belief_update_nominal_visible_only_mode!(scenario.blue_belief_on_red)
    belief_update_nominal_visible_only_mode!(scenario.red_belief_on_blue)

    synchronize_first_order_beliefs!(scenario)

    return nothing
end

# ============================================================
# STAGE 12 — COMPACT JOINT BELIEF REPRESENTATION
# ============================================================

function vecsym_upper(P::AbstractMatrix{<:Float64})
    n = size(P, 1)
    vals = Float64[]

    for j in 1:n
        for i in 1:j
            push!(vals, P[i, j])
        end
    end

    return vals
end

function matsym_from_upper(v::Vector{Float64}, n::Int)
    P = zeros(n, n)
    k = 1

    for j in 1:n
        for i in 1:j
            P[i, j] = v[k]
            P[j, i] = v[k]
            k += 1
        end
    end

    return P
end

function pack_agent_belief_stage12(belief::AgentBelief)
    return vcat(belief.mean, vecsym_upper(belief.cov))
end

function unpack_agent_belief_stage12(v::Vector{Float64}; n_x::Int = 4)
    nP = div(n_x * (n_x + 1), 2)

    μ = copy(v[1:n_x])
    Pv = copy(v[n_x+1:n_x+nP])

    P = matsym_from_upper(Pv, n_x)
    return AgentBelief(μ, P)
end

function pack_joint_belief_stage12(scenario::TwoAgentScenario)
    return vcat(
        pack_agent_belief_stage12(scenario.blue_self_belief),
        pack_agent_belief_stage12(scenario.red_self_belief),
        pack_agent_belief_stage12(scenario.blue_belief_on_red),
        pack_agent_belief_stage12(scenario.red_belief_on_blue)
    )
end

function unpack_joint_belief_stage12(
    b::Vector{Float64},
    reference_scenario::TwoAgentScenario;
    n_x::Int = 4
)
    n_single = n_x + div(n_x * (n_x + 1), 2)

    i1 = 1:n_single
    i2 = n_single+1:2n_single
    i3 = 2n_single+1:3n_single
    i4 = 3n_single+1:4n_single

    blue_self    = unpack_agent_belief_stage12(copy(b[i1]); n_x = n_x)
    red_self     = unpack_agent_belief_stage12(copy(b[i2]); n_x = n_x)
    blue_on_red  = unpack_agent_belief_stage12(copy(b[i3]); n_x = n_x)
    red_on_blue  = unpack_agent_belief_stage12(copy(b[i4]); n_x = n_x)

    return TwoAgentScenario(
        deepcopy(reference_scenario.blue_true),
        deepcopy(reference_scenario.red_true),
        blue_self,
        red_self,
        blue_on_red,
        red_on_blue,
        deepcopy(reference_scenario.blue_block_memory),
        deepcopy(reference_scenario.red_block_memory)
    )
end

function joint_belief_from_stage11_step(
    step::NominalTwoPlayerStep,
    reference_scenario::TwoAgentScenario
)
    sc = TwoAgentScenario(
        deepcopy(reference_scenario.blue_true),
        deepcopy(reference_scenario.red_true),

        AgentBelief(copy(step.blue_self_mean),    copy(step.blue_self_cov)),
        AgentBelief(copy(step.red_self_mean),     copy(step.red_self_cov)),

        AgentBelief(copy(step.blue_on_red_mean),  copy(step.blue_on_red_cov)),
        AgentBelief(copy(step.red_on_blue_mean),  copy(step.red_on_blue_cov)),

        deepcopy(reference_scenario.blue_block_memory),
        deepcopy(reference_scenario.red_block_memory)
    )

    return pack_joint_belief_stage12(sc), sc
end

# ============================================================
# STAGE 12 — JOINT BELIEF MAP
# ============================================================

function one_step_joint_belief_map_stage12(
    bk::Vector{Float64},
    uB::Vector{Float64},
    uR::Vector{Float64},
    reference_scenario::TwoAgentScenario,
    model_blue::CarModel,
    model_red::CarModel
)
    sc = unpack_joint_belief_stage12(bk, reference_scenario)

    # deterministic nominal prediction
    belief_predict_mode!(sc.blue_self_belief,   uB[1], uB[2], dt, model_blue)
    belief_predict_mode!(sc.red_self_belief,    uR[1], uR[2], dt, model_red)

    belief_predict_mode!(sc.blue_belief_on_red, uR[1], uR[2], dt, model_red)
    belief_predict_mode!(sc.red_belief_on_blue, uB[1], uB[2], dt, model_blue)

    # deterministic covariance-only nominal update
    belief_update_nominal_visible_only_mode!(sc.blue_self_belief)
    belief_update_nominal_visible_only_mode!(sc.red_self_belief)
    belief_update_nominal_visible_only_mode!(sc.blue_belief_on_red)
    belief_update_nominal_visible_only_mode!(sc.red_belief_on_blue)

    synchronize_first_order_beliefs!(sc)

    return pack_joint_belief_stage12(sc)
end

function block_proximity_penalty_smooth(
    mean::Vector{Float64},
    Pxy::Matrix{Float64};
    forbidden_blocks = forbidden_blocks,
    κ_unc::Float64 = 2.0,
    eps_r::Float64 = 1e-3,
    angle_margin::Float64 = 0.05
)
    x = mean[1]
    y = mean[2]

    r = hypot(x, y)
    θ = atan(y, x)
    θ = θ < 0 ? θ + 2π : θ

    σ_unc = sqrt(max(tr(Pxy), 0.0))
    J = 0.0

    for block in forbidden_blocks
        dθ = abs(mod(θ - block.theta_center + π, 2π) - π)

        if dθ <= block.half_angle + angle_margin
            if block.side == "inner"
                d = (r - κ_unc * σ_unc) - (R_inner + block.radial_depth)
            else
                d = (R_outer - block.radial_depth) - (r + κ_unc * σ_unc)
            end

            d = max(d, eps_r)
            J += 1.0 / d^2
        end
    end

    return J
end

# ============================================================
# STAGE 12 — LOCAL BELIEF DYNAMICS LINEARIZATION
# ============================================================

function linearize_two_player_belief_dynamics_fd(
    b0::Vector{Float64},
    uB0::Vector{Float64},
    uR0::Vector{Float64},
    reference_scenario::TwoAgentScenario,
    model_blue::CarModel,
    model_red::CarModel;
    h_b::Float64 = 1e-4,
    h_u::Float64 = 1e-4
)
    g0 = one_step_joint_belief_map_stage12(
        b0, uB0, uR0, reference_scenario, model_blue, model_red
    )

    n_b  = length(b0)
    n_uB = length(uB0)
    n_uR = length(uR0)

    G_b  = zeros(length(g0), n_b)
    G_uB = zeros(length(g0), n_uB)
    G_uR = zeros(length(g0), n_uR)

    for i in 1:n_b
        e = zeros(n_b)
        e[i] = h_b

        gp = one_step_joint_belief_map_stage12(
            b0 + e, uB0, uR0, reference_scenario, model_blue, model_red
        )

        gm = one_step_joint_belief_map_stage12(
            b0 - e, uB0, uR0, reference_scenario, model_blue, model_red
        )

        G_b[:, i] = (gp - gm) / (2h_b)
    end

    for i in 1:n_uB
        e = zeros(n_uB)
        e[i] = h_u

        gp = one_step_joint_belief_map_stage12(
            b0, uB0 + e, uR0, reference_scenario, model_blue, model_red
        )

        gm = one_step_joint_belief_map_stage12(
            b0, uB0 - e, uR0, reference_scenario, model_blue, model_red
        )

        G_uB[:, i] = (gp - gm) / (2h_u)
    end

    for i in 1:n_uR
        e = zeros(n_uR)
        e[i] = h_u

        gp = one_step_joint_belief_map_stage12(
            b0, uB0, uR0 + e, reference_scenario, model_blue, model_red
        )

        gm = one_step_joint_belief_map_stage12(
            b0, uB0, uR0 - e, reference_scenario, model_blue, model_red
        )

        G_uR[:, i] = (gp - gm) / (2h_u)
    end

    return Stage12LocalDynamicsModel(G_b, G_uB, G_uR, g0)
end

# ============================================================
# STAGE COSTS
# ============================================================

function compute_boundary_penalty(
    μ::Vector{Float64};
    R_inner::Float64 = R_inner,
    R_outer::Float64 = R_outer,
    eps_b::Float64 = boundary_penalty_eps
)
    r = hypot(μ[1], μ[2])

    d_inner = max(r - R_inner, eps_b)
    d_outer = max(R_outer - r, eps_b)

    return 1.0 / d_inner^2 + 1.0 / d_outer^2
end

function instantaneous_progress_rate(belief::AgentBelief)
    x, y, vx, vy = belief.mean
    t_hat, _, _ = unit_vectors(x, y)

    v_t = max(vx * t_hat[1] + vy * t_hat[2], 0.0)
    r = max(hypot(x, y), 1e-6)

    return v_t / r, v_t, r
end

function block_proximity_penalty(
    x::Float64,
    y::Float64;
    forbidden_blocks = forbidden_blocks,
    eps_r::Float64 = 1e-3
)
    r = hypot(x, y)
    θ = atan(y, x)
    θ = θ < 0 ? θ + 2π : θ

    J = 0.0

    for block in forbidden_blocks
        dθ = abs(mod(θ - block.theta_center + π, 2π) - π)

        if dθ <= block.half_angle
            if block.side == "inner"
                d = r - (R_inner + block.radial_depth)
            else
                d = (R_outer - block.radial_depth) - r
            end

            d = max(d, eps_r)
            J += 1.0 / d^2
        end
    end

    return J
end

function compute_stage_cost(
    belief::AgentBelief,
    u::Vector{Float64};
    w_prog::Float64 = w_prog,
    w_u::Float64 = w_u,
    w_wall::Float64 = w_wall,
    w_block::Float64 = w_block,
    w_info::Float64 = w_info
)
    x, y, vx, vy = belief.mean
    Pxy = belief.cov[1:2, 1:2]

    prog_rate, v_t, r = instantaneous_progress_rate(belief)

        J_prog  = -w_prog * prog_rate
    J_u     =  w_u * (u[1]^2 + u[2]^2)
    J_wall  =  w_wall * compute_boundary_penalty(belief.mean)

    J_block = w_block * block_proximity_penalty_smooth(
        belief.mean,
        Pxy;
        forbidden_blocks = forbidden_blocks
    )

    J_info  =  w_info * tr(Pxy)

    J_total = J_prog + J_u + J_wall + J_block + J_info

    return (
        J_total = J_total,
        J_prog  = J_prog,
        J_u     = J_u,
        J_wall  = J_wall,
        J_block = J_block,
        J_info  = J_info,
        v_t     = v_t,
        r       = r,
        prog_rate = prog_rate
    )
end

function block_activation_gain(
    x::Float64,
    y::Float64,
    block;
    detect_margin::Float64 = 0.10,
    gain_width::Float64 = 0.08
)
    θ = atan(y, x)
    θ = θ < 0 ? θ + 2π : θ

    dθ = abs(wrap_angle_pi(θ - block.theta_center))
    ξ = (block.half_angle + detect_margin - dθ) / max(gain_width, 1e-6)

    return clamp(ξ, 0.0, 1.0)
end

function next_block_exit_target(
    x::Float64,
    y::Float64;
    forbidden_blocks = forbidden_blocks,
    inner_margin::Float64 = inner_margin_default,
    outer_margin::Float64 = outer_margin_default,
    block_clearance::Float64 = block_clearance_default,
    extra_angle::Float64 = 0.03
)
    info = next_known_block_ahead(
        x, y;
        forbidden_blocks = forbidden_blocks,
        max_lookahead_angle = 1.2
    )

    if !info.active
        return nothing
    end

    block = forbidden_blocks[info.block_index]

    θ_exit = mod(block.theta_center + block.half_angle + extra_angle, 2π)

    r_low, r_high = free_radial_corridor(
        θ_exit;
        forbidden_blocks = forbidden_blocks,
        R_inner = R_inner,
        R_outer = R_outer,
        inner_margin = inner_margin,
        outer_margin = outer_margin,
        block_clearance = block_clearance
    )

    if block.side == "inner"
        # want to be near the lower feasible radius, but safely above it
        r_target = r_low + 0.20 * max(r_high - r_low, 1e-6)
    else
        # if later you add outer blocks
        r_target = r_high - 0.20 * max(r_high - r_low, 1e-6)
    end

    return (
        block_index = info.block_index,
        θ_exit = θ_exit,
        r_low = r_low,
        r_high = r_high,
        r_target = r_target,
        side = block.side
    )
end

function preview_block_passability_cost(
    belief::AgentBelief,
    u::Vector{Float64},
    model::CarModel;
    n_preview::Int = H,
    dt_preview::Float64 = dt,
    w_miss_exit::Float64 = 120.0,
    w_wrong_side::Float64 = 250.0,
    w_corridor::Float64 = 40.0
)
    target = next_block_exit_target(belief.mean[1], belief.mean[2])

    if target === nothing
        return 0.0
    end

    b = deepcopy(belief)
    best_exit_err = Inf
    reached_exit = false
    J = 0.0

    for _ in 1:n_preview
        belief_predict!(b, u[1], u[2], dt_preview, model)

        x = b.mean[1]
        y = b.mean[2]
        r = hypot(x, y)
        θ = atan(y, x)
        θ = θ < 0 ? θ + 2π : θ

        r_low, r_high = free_radial_corridor(
            θ;
            forbidden_blocks = forbidden_blocks,
            R_inner = R_inner,
            R_outer = R_outer,
            inner_margin = inner_margin_default,
            outer_margin = outer_margin_default,
            block_clearance = block_clearance_default
        )

        corridor_width = max(r_high - r_low, 1e-6)

        # penalize being outside the feasible corridor anywhere in preview
        if r < r_low
            J += w_wrong_side * (r_low - r)^2
        elseif r > r_high
            J += w_wrong_side * (r - r_high)^2
        end

        # small shaping inside narrow corridor
        J += w_corridor * ((r - clamp(r, r_low, r_high))^2) / corridor_width^2

        dθ_exit = abs(wrap_angle_pi(θ - target.θ_exit))

        if dθ_exit < 0.05
            reached_exit = true
            best_exit_err = min(best_exit_err, abs(r - target.r_target))
        end
    end

    if reached_exit
        J += w_miss_exit * best_exit_err^2
    else
        J += w_miss_exit * 4.0
    end

    return J
end

# ============================================================
# TWO-AGENT COLLISION COST
# ============================================================

function two_agent_collision_cost(
    belief_a::AgentBelief,
    belief_b::AgentBelief;
    w_c::Float64 = w_coll_default,
    d_safe::Float64 = d_safe_coll_default,
    σ_coll::Float64 = σ_coll_default,
    k_unc::Float64 = k_unc_coll_default,
    eps_d::Float64 = 1e-3
)
    μa = belief_a.mean[1:2]
    μb = belief_b.mean[1:2]

    Σa = belief_a.cov[1:2, 1:2]
    Σb = belief_b.cov[1:2, 1:2]

    d = norm(μb - μa)
    unc_radius = sqrt(max(tr(Σa + Σb), 0.0))

    d_eff = max(d - k_unc * unc_radius, eps_d)

    if d_eff >= d_safe
        return 0.0
    else
        return w_c * (1.0 / d_eff - 1.0 / d_safe)^2
    end
end

function compute_two_agent_stage_costs(
    scenario::TwoAgentScenario,
    u_blue::Vector{Float64},
    u_red::Vector{Float64};
    w_coll::Float64 = w_coll_default
)
    blue_terms = compute_stage_cost(scenario.blue_self_belief, u_blue)
    red_terms  = compute_stage_cost(scenario.red_self_belief,  u_red)

    Jb_coll = two_agent_collision_cost(
        scenario.blue_self_belief,
        scenario.blue_belief_on_red;
        w_c = w_coll
    )

    Jr_coll = two_agent_collision_cost(
        scenario.red_self_belief,
        scenario.red_belief_on_blue;
        w_c = w_coll
    )

    J_blue = blue_terms.J_total + Jb_coll
    J_red  = red_terms.J_total  + Jr_coll

    return (
        J_blue = J_blue,
        J_red  = J_red,
        Jb_coll = Jb_coll,
        Jr_coll = Jr_coll,
        blue_terms = blue_terms,
        red_terms = red_terms
    )
end

# ============================================================
# PREVIEW COSTS
# ============================================================

function preview_block_deadend_cost(
    belief::AgentBelief,
    u::Vector{Float64},
    model::CarModel;
    n_preview::Int = 12,
    dt_preview::Float64 = dt,
    w_preview::Float64 = w_preview_deadend_stage10,
    β_preview::Float64 = 0.20
)
    b = deepcopy(belief)
    J = 0.0

    for k in 1:n_preview
        belief_predict_feasible!(b, u[1], u[2], dt_preview, model)

        x = b.mean[1]
        y = b.mean[2]

        θ = atan(y, x)
        θ = θ < 0 ? θ + 2π : θ

        r_low, r_high = free_radial_corridor(
            θ;
            forbidden_blocks = forbidden_blocks,
            R_inner = R_inner,
            R_outer = R_outer,
            inner_margin = inner_margin_default,
            outer_margin = outer_margin_default,
            block_clearance = block_clearance_default
        )

        corridor_width = max(r_high - r_low, 1e-6)
        r_now = hypot(x, y)
        r_mid = 0.5 * (r_low + r_high)

        wk = exp(-β_preview * (k - 1))

        J += wk * w_preview * ((r_now - r_mid)^2 / corridor_width^2)

        if corridor_width < 1.0
            J += wk * w_preview * 10.0 / corridor_width
        end
    end

    return J
end

function preview_infeasibility_cost(
    belief::AgentBelief,
    u::Vector{Float64},
    model::CarModel;
    n_preview::Int = H,
    dt_preview::Float64 = dt,
    w_hit::Float64 = 200.0,
    w_narrow::Float64 = 25.0
)
    b = deepcopy(belief)
    J = 0.0

    for _ in 1:n_preview
        # planning preview without feasible repair
        belief_predict!(b, u[1], u[2], dt_preview, model)

        x = b.mean[1]
        y = b.mean[2]

        if !point_is_feasible(x, y; forbidden_blocks = forbidden_blocks)
            J += w_hit
        end

        θ = atan(y, x)
        θ = θ < 0 ? θ + 2π : θ

        r_low, r_high = free_radial_corridor(
            θ;
            forbidden_blocks = forbidden_blocks,
            R_inner = R_inner,
            R_outer = R_outer,
            inner_margin = inner_margin_default,
            outer_margin = outer_margin_default,
            block_clearance = block_clearance_default
        )

        corridor_width = max(r_high - r_low, 1e-6)

        if corridor_width < 2.0
            J += w_narrow / corridor_width
        end
    end

    return J
end

# ============================================================
# BASELINE ROLLOUT EVALUATION
# ============================================================

function evaluate_constant_control_rollout(
    scenario::TwoAgentScenario,
    u_blue::Vector{Float64},
    u_red::Vector{Float64},
    model_blue::CarModel,
    model_red::CarModel;
    H::Int = H
)
    sc = deepcopy(scenario)
    Jb = 0.0
    Jr = 0.0

    for _ in 1:H
        xB_old, yB_old = sc.blue_true.x, sc.blue_true.y
        xR_old, yR_old = sc.red_true.x,  sc.red_true.y

        # deterministic nominal planner truth
        propagate_two_agent_truth_unrepaired!(sc, u_blue, u_red, model_blue, model_red)

        blue_bad =
            !point_is_feasible(sc.blue_true.x, sc.blue_true.y; forbidden_blocks = forbidden_blocks) ||
            segment_crosses_forbidden(
                xB_old, yB_old, sc.blue_true.x, sc.blue_true.y;
                forbidden_blocks = forbidden_blocks
            )

        red_bad =
            !point_is_feasible(sc.red_true.x, sc.red_true.y; forbidden_blocks = forbidden_blocks) ||
            segment_crosses_forbidden(
                xR_old, yR_old, sc.red_true.x, sc.red_true.y;
                forbidden_blocks = forbidden_blocks
            )

        if blue_bad || red_bad
            return Inf, Inf
        end

        # deterministic nominal planner beliefs
        predict_two_agent_beliefs_nominal!(sc, u_blue, u_red, model_blue, model_red)

        belief_update_nominal_visible_only!(sc.blue_self_belief)
        belief_update_nominal_visible_only!(sc.red_self_belief)
        belief_update_nominal_visible_only!(sc.blue_belief_on_red)
        belief_update_nominal_visible_only!(sc.red_belief_on_blue)

        synchronize_first_order_beliefs!(sc)

        costs = compute_two_agent_stage_costs(sc, u_blue, u_red)
        Jb += costs.J_blue
        Jr += costs.J_red
    end

    return Jb, Jr
end

# ============================================================
# STAGE 10 BASELINE GREEDY SYMMETRIC PLANNER
# ============================================================

function choose_greedy_control_symmetric(
    scenario::TwoAgentScenario,
    self_side::Symbol,
    u_other_guess::Vector{Float64},
    model_blue::CarModel,
    model_red::CarModel
)
    best_u = [a_t_candidates_stage10[1], a_r_candidates_stage10[1]]
    best_J = Inf

    for a_t in a_t_candidates_stage10
        for a_r in a_r_candidates_stage10
            u_self = [a_t, a_r]

            if self_side == :blue
                u_blue = u_self
                u_red  = u_other_guess

                J_roll, _ = evaluate_constant_control_rollout(
                    scenario, u_blue, u_red, model_blue, model_red
                )

                J_pass = preview_block_passability_cost(
                    scenario.blue_self_belief,
                    u_blue,
                    model_blue;
                    n_preview = H,
                    dt_preview = dt
                )

                J_infeas = preview_infeasibility_cost(
                    scenario.blue_self_belief,
                    u_blue,
                    model_blue;
                    n_preview = H,
                    dt_preview = dt
                )

                J_total = J_roll + J_pass + J_infeas

            elseif self_side == :red
                u_blue = u_other_guess
                u_red  = u_self

                _, J_roll = evaluate_constant_control_rollout(
                    scenario, u_blue, u_red, model_blue, model_red
                )

                J_pass = preview_block_passability_cost(
                    scenario.red_self_belief,
                    u_red,
                    model_red;
                    n_preview = H,
                    dt_preview = dt
                )

                J_infeas = preview_infeasibility_cost(
                    scenario.red_self_belief,
                    u_red,
                    model_red;
                    n_preview = H,
                    dt_preview = dt
                )

                J_total = J_roll + J_pass + J_infeas

            else
                error("self_side must be :blue or :red")
            end

            if J_total < best_J
                best_J = J_total
                best_u = u_self
            end
        end
    end

    return best_u, best_J
end

function radial_action_limits_from_visible_block(
    x::Float64,
    y::Float64;
    forbidden_blocks = forbidden_blocks,
    margin_angle::Float64 = 0.06,
    clearance_buffer::Float64 = 1.0,
    a_r_lim::Float64 = a_r_max
)
    θ = atan(y, x)
    θ = θ < 0 ? θ + 2π : θ

    r_low, r_high = free_radial_corridor(
        θ;
        forbidden_blocks = forbidden_blocks,
        R_inner = R_inner,
        R_outer = R_outer,
        inner_margin = inner_margin_default,
        outer_margin = outer_margin_default,
        block_clearance = block_clearance_default
    )

    r = hypot(x, y)

    d_low  = r - r_low
    d_high = r_high - r

    if d_low <= clearance_buffer
        return (0.0, a_r_lim)
    elseif d_high <= clearance_buffer
        return (-a_r_lim, 0.0)
    else
        return (-a_r_lim, a_r_lim)
    end
end

# ============================================================
# NOMINAL CONTROL-SEQUENCE INITIALIZATION
# ============================================================

function initialize_nominal_sequence(
    u0::Vector{Float64},
    H::Int
)
    return [copy(u0) for _ in 1:H]
end

function initialize_two_player_nominal_sequences(
    H::Int;
    u_blue0::Vector{Float64} = [0.10, 0.0],
    u_red0::Vector{Float64}  = [0.10, 0.0]
)
    u_blue_seq = initialize_nominal_sequence(u_blue0, H)
    u_red_seq  = initialize_nominal_sequence(u_red0, H)
    return u_blue_seq, u_red_seq
end

# ============================================================
# STAGE 11 — TWO-PLAYER NOMINAL SEQUENCE ROLLOUT
# ============================================================

function rollout_two_player_nominal_sequences(
    scenario0::TwoAgentScenario,
    u_blue_seq::Vector{Vector{Float64}},
    u_red_seq::Vector{Vector{Float64}},
    model_blue::CarModel,
    model_red::CarModel
)
    Hloc = length(u_blue_seq)

    if length(u_red_seq) != Hloc
        error("u_blue_seq and u_red_seq must have the same horizon length")
    end

    sc = deepcopy(scenario0)

    steps = NominalTwoPlayerStep[]
    J_blue_stage_hist = Float64[]
    J_red_stage_hist  = Float64[]
    J_blue_coll_hist  = Float64[]
    J_red_coll_hist   = Float64[]

    J_blue_total = 0.0
    J_red_total  = 0.0

    for k in 1:Hloc
        u_blue = u_blue_seq[k]
        u_red  = u_red_seq[k]

        xB_old, yB_old = sc.blue_true.x, sc.blue_true.y
        xR_old, yR_old = sc.red_true.x,  sc.red_true.y

        # deterministic nominal planner truth
        propagate_two_agent_truth_unrepaired!(sc, u_blue, u_red, model_blue, model_red)

        # deterministic nominal planner beliefs
        predict_two_agent_beliefs_nominal!(sc, u_blue, u_red, model_blue, model_red)

        belief_update_nominal_visible_only!(sc.blue_self_belief)
        belief_update_nominal_visible_only!(sc.red_self_belief)
        belief_update_nominal_visible_only!(sc.blue_belief_on_red)
        belief_update_nominal_visible_only!(sc.red_belief_on_blue)

        synchronize_first_order_beliefs!(sc)

        blue_bad =
            !point_is_feasible(sc.blue_true.x, sc.blue_true.y; forbidden_blocks = forbidden_blocks) ||
            segment_crosses_forbidden(
                xB_old, yB_old, sc.blue_true.x, sc.blue_true.y;
                forbidden_blocks = forbidden_blocks
            )

        red_bad =
            !point_is_feasible(sc.red_true.x, sc.red_true.y; forbidden_blocks = forbidden_blocks) ||
            segment_crosses_forbidden(
                xR_old, yR_old, sc.red_true.x, sc.red_true.y;
                forbidden_blocks = forbidden_blocks
            )

        if blue_bad || red_bad
            push!(steps, NominalTwoPlayerStep(
                copy(sc.blue_self_belief.mean),
                copy(sc.red_self_belief.mean),
                copy(sc.blue_belief_on_red.mean),
                copy(sc.red_belief_on_blue.mean),
                copy(sc.blue_self_belief.cov),
                copy(sc.red_self_belief.cov),
                copy(sc.blue_belief_on_red.cov),
                copy(sc.red_belief_on_blue.cov),
                copy(u_blue),
                copy(u_red),
                Inf, Inf, 0.0, 0.0
            ))

            push!(J_blue_stage_hist, Inf)
            push!(J_red_stage_hist,  Inf)
            push!(J_blue_coll_hist,  0.0)
            push!(J_red_coll_hist,   0.0)

            J_blue_total = Inf
            J_red_total  = Inf
            break
        end

        costs = compute_two_agent_stage_costs(sc, u_blue, u_red)

        push!(steps, NominalTwoPlayerStep(
            copy(sc.blue_self_belief.mean),
            copy(sc.red_self_belief.mean),
            copy(sc.blue_belief_on_red.mean),
            copy(sc.red_belief_on_blue.mean),
            copy(sc.blue_self_belief.cov),
            copy(sc.red_self_belief.cov),
            copy(sc.blue_belief_on_red.cov),
            copy(sc.red_belief_on_blue.cov),
            copy(u_blue),
            copy(u_red),
            costs.J_blue,
            costs.J_red,
            costs.Jb_coll,
            costs.Jr_coll
        ))

        push!(J_blue_stage_hist, costs.J_blue)
        push!(J_red_stage_hist,  costs.J_red)
        push!(J_blue_coll_hist,  costs.Jb_coll)
        push!(J_red_coll_hist,   costs.Jr_coll)

        J_blue_total += costs.J_blue
        J_red_total  += costs.J_red
    end

    return NominalTwoPlayerRollout(
        steps,
        J_blue_total,
        J_red_total,
        J_blue_stage_hist,
        J_red_stage_hist,
        J_blue_coll_hist,
        J_red_coll_hist
    )
end

# ============================================================
# STAGE 12 — PER-PLAYER COST FROM COMPACT JOINT STATE
# ============================================================

function stage12_player_cost_from_joint_state(
    b::Vector{Float64},
    uB::Vector{Float64},
    uR::Vector{Float64},
    player::Symbol,
    reference_scenario::TwoAgentScenario
)
    sc = unpack_joint_belief_stage12(b, reference_scenario)

    costs = compute_two_agent_stage_costs(sc, uB, uR)

    J_blue = costs.J_blue
    J_red  = costs.J_red

    if player == :blue
        return J_blue
    elseif player == :red
        return J_red
    else
        error("player must be :blue or :red")
    end
end

# ============================================================
# STAGE 12 — LOCAL STAGE-COST QUADRATICIZATION
# ============================================================

function quadraticize_two_player_stage_cost_fd(
    b0::Vector{Float64},
    uB0::Vector{Float64},
    uR0::Vector{Float64},
    player::Symbol,
    reference_scenario::TwoAgentScenario;
    h::Float64 = 1e-4
)
    n_b  = length(b0)
    n_uB = length(uB0)
    n_uR = length(uR0)

    s0 = vcat(b0, uB0, uR0)
    n_s = length(s0)

    function eval_s(s::Vector{Float64})
        b  = s[1:n_b]
        uB = s[n_b+1:n_b+n_uB]
        uR = s[n_b+n_uB+1:end]

        return stage12_player_cost_from_joint_state(
            b, uB, uR, player, reference_scenario
        )
    end

    c = eval_s(s0)
    q = zeros(n_s)
    Q = zeros(n_s, n_s)

    for i in 1:n_s
        ei = zeros(n_s)
        ei[i] = h

        fp = eval_s(s0 + ei)
        fm = eval_s(s0 - ei)

        q[i] = (fp - fm) / (2h)
    end

    for i in 1:n_s
        ei = zeros(n_s)
        ei[i] = h

        for j in i:n_s
            ej = zeros(n_s)
            ej[j] = h

            fpp = eval_s(s0 + ei + ej)
            fpm = eval_s(s0 + ei - ej)
            fmp = eval_s(s0 - ei + ej)
            fmm = eval_s(s0 - ei - ej)

            Hij = (fpp - fpm - fmp + fmm) / (4h^2)

            Q[i, j] = Hij
            Q[j, i] = Hij
        end
    end

    Q = 0.5 * (Q + Q')

    return Stage12LocalQuadraticCost(c, q, Q, n_b, n_uB, n_uR)
end

# ============================================================
# STAGE 14 — AUGMENT LOCAL GAME WITH VALUE-TO-GO
# ============================================================

function stage14_build_augmented_player_model(
    dyn::Stage12LocalDynamicsModel,
    quad::Stage12LocalQuadraticCost,
    Vnext::Stage14ValueModel
)
    n_b  = quad.n_b
    n_uB = quad.n_uB
    n_uR = quad.n_uR
    n_s  = n_b + n_uB + n_uR

    G = hcat(dyn.G_b, dyn.G_uB, dyn.G_uR)

    q_aug = copy(quad.q)
    Q_aug = copy(quad.Q)
    c_aug = quad.c + Vnext.v

    q_aug += G' * Vnext.v_b
    Q_aug += G' * Vnext.V_bb * G
    Q_aug = 0.5 * (Q_aug + Q_aug')

    return c_aug, q_aug, Q_aug
end

function stage14_safe_cond(M::AbstractMatrix{<:Float64}; eps_cond::Float64 = 1e-12)
    s = svdvals(Matrix(M))
    smax = maximum(s)
    smin = minimum(s)

    if smin < eps_cond
        return Inf
    else
        return smax / smin
    end
end

# ============================================================
# STAGE 14 — TRUE TWO-PLAYER BACKWARD PASS
# ============================================================

function run_stage14_true_backward_pass(
    rollout::NominalTwoPlayerRollout,
    scenario0::TwoAgentScenario,
    model_blue::CarModel,
    model_red::CarModel;
    reg_u::Float64 = 1e-5,
    reg_b::Float64 = 1e-8
)
    H = length(rollout.steps)

    n_b = length(joint_belief_from_stage11_step(rollout.steps[end], scenario0)[1])

    V_blue = Stage14ValueModel(0.0, zeros(n_b), zeros(n_b, n_b))
    V_red  = Stage14ValueModel(0.0, zeros(n_b), zeros(n_b, n_b))

    backward_steps = Vector{Stage14BackwardStep}(undef, H)

    for k in H:-1:1
        step = rollout.steps[k]

        b0, sc_ref = joint_belief_from_stage11_step(step, scenario0)
        uB0 = copy(step.u_blue)
        uR0 = copy(step.u_red)

        dyn = linearize_two_player_belief_dynamics_fd(
            b0, uB0, uR0, sc_ref, model_blue, model_red
        )

        quad_blue = quadraticize_two_player_stage_cost_fd(
            b0, uB0, uR0, :blue, sc_ref
        )

        quad_red = quadraticize_two_player_stage_cost_fd(
            b0, uB0, uR0, :red, sc_ref
        )

        _, qB, QB = stage14_build_augmented_player_model(dyn, quad_blue, V_blue)
        _, qR, QR = stage14_build_augmented_player_model(dyn, quad_red, V_red)

        n_uB = quad_blue.n_uB
        n_uR = quad_blue.n_uR

        i_b  = 1:quad_blue.n_b
        i_uB = quad_blue.n_b + 1 : quad_blue.n_b + n_uB
        i_uR = quad_blue.n_b + n_uB + 1 : quad_blue.n_b + n_uB + n_uR

        Qb_bb = QB[i_b, i_b] + reg_b * I
        Qb_bB = QB[i_b, i_uB]
        Qb_bR = QB[i_b, i_uR]
        Qb_Bb = QB[i_uB, i_b]
        Qb_BB = QB[i_uB, i_uB] + reg_u * I
        Qb_BR = QB[i_uB, i_uR]

        Qr_bb = QR[i_b, i_b] + reg_b * I
        Qr_bB = QR[i_b, i_uB]
        Qr_bR = QR[i_b, i_uR]
        Qr_Rb = QR[i_uR, i_b]
        Qr_RB = QR[i_uR, i_uB]
        Qr_RR = QR[i_uR, i_uR] + reg_u * I

        qb_b = qB[i_b]
        qb_B = qB[i_uB]

        qr_b = qR[i_b]
        qr_R = qR[i_uR]

        A_game = [
            Qb_BB   Qb_BR
            Qr_RB   Qr_RR
        ]

        B_game = [
            Qb_Bb
            Qr_Rb
        ]

        a_game = -[
            qb_B
            qr_R
        ]

        F_game = -B_game

        cond_game = stage14_safe_cond(A_game)

        sol_j = A_game \ a_game
        sol_K = A_game \ F_game

        j_blue = sol_j[1:n_uB]
        j_red  = sol_j[n_uB+1:end]

        K_blue = sol_K[1:n_uB, :]
        K_red  = sol_K[n_uB+1:end, :]

        backward_steps[k] = Stage14BackwardStep(
            j_blue, K_blue,
            j_red,  K_red,
            cond_game
        )

        Mj = vcat(j_blue, j_red)
        MK = vcat(K_blue, K_red)

        Vb_b =
            qb_b +
            Qb_bB * j_blue +
            Qb_bR * j_red +
            K_blue' * qb_B +
            K_red'  * zeros(n_uR)

        Vr_b =
            qr_b +
            Qr_bB * j_blue +
            Qr_bR * j_red +
            K_red' * qr_R +
            K_blue' * zeros(n_uB)

        Vb_bb =
            Qb_bb +
            Qb_bB * K_blue +
            Qb_bR * K_red +
            K_blue' * Qb_Bb +
            K_blue' * Qb_BB * K_blue +
            K_blue' * Qb_BR * K_red

        Vr_bb =
            Qr_bb +
            Qr_bB * K_blue +
            Qr_bR * K_red +
            K_red' * Qr_Rb +
            K_red' * Qr_RB * K_blue +
            K_red' * Qr_RR * K_red

        V_blue = Stage14ValueModel(0.0, Vb_b, 0.5 * (Vb_bb + Vb_bb'))
        V_red  = Stage14ValueModel(0.0, Vr_b, 0.5 * (Vr_bb + Vr_bb'))
    end

    return Stage14BackwardPassResult(backward_steps, V_blue, V_red)
end

# ============================================================
# STAGE 13 — LOCAL STATIC GAME BLOCKS
# ============================================================

function extract_stage13_game_blocks(
    quad_blue::Stage12LocalQuadraticCost,
    quad_red::Stage12LocalQuadraticCost
)
    n_b  = quad_blue.n_b
    n_uB = quad_blue.n_uB
    n_uR = quad_blue.n_uR

    i_uB = n_b + 1 : n_b + n_uB
    i_uR = n_b + n_uB + 1 : n_b + n_uB + n_uR

    Q_BB = quad_blue.Q[i_uB, i_uB]
    Q_BR = quad_blue.Q[i_uB, i_uR]
    q_B  = quad_blue.q[i_uB]

    Q_RB = quad_red.Q[i_uR, i_uB]
    Q_RR = quad_red.Q[i_uR, i_uR]
    q_R  = quad_red.q[i_uR]

    return Q_BB, Q_BR, q_B, Q_RB, Q_RR, q_R
end

# ============================================================
# STAGE 13 — STATIC TWO-PLAYER NASH STEP
# ============================================================

function solve_stage13_static_game(
    quad_blue::Stage12LocalQuadraticCost,
    quad_red::Stage12LocalQuadraticCost;
    reg::Float64 = 1e-6
)
    Q_BB, Q_BR, q_B, Q_RB, Q_RR, q_R =
        extract_stage13_game_blocks(quad_blue, quad_red)

    n_uB = length(q_B)
    n_uR = length(q_R)

    Q_BB_reg = 0.5 * (Q_BB + Q_BB') + reg * I
    Q_RR_reg = 0.5 * (Q_RR + Q_RR') + reg * I

    A = [
        Q_BB_reg  Q_BR
        Q_RB      Q_RR_reg
    ]

    rhs = -vcat(q_B, q_R)

    du = A \ rhs

    du_blue = du[1:n_uB]
    du_red  = du[n_uB+1:n_uB+n_uR]

    return du_blue, du_red
end

function clamp_control_vector(
    u::Vector{Float64},
    x::Float64,
    y::Float64;
    a_t_lim::Float64 = a_t_max,
    a_r_lim::Float64 = a_r_max
)
    ar_min, ar_max = radial_action_limits_from_visible_block(
        x, y;
        forbidden_blocks = forbidden_blocks,
        a_r_lim = a_r_lim
    )

    return [
        clamp(u[1], -a_t_lim, a_t_lim),
        clamp(u[2], ar_min, ar_max)
    ]
end

# ============================================================
# PLOTTING HELPERS
# ============================================================

function apply_paperstyle_axes!(p; title_str::String = "")
    Plots.plot!(
        p;
        size = (1100, 950),
        dpi = 300,
        aspect_ratio = 1,
        grid = true,
        framestyle = :box,
        legend = :inside,
        legend_position = (0.50, 0.28),
        background_color = :white,
        foreground_color_legend = :white,
        title = title_str
    )
    return p
end

function plot_track()
    θ = range(0, 2π, length = 400)

    inner_x = R_inner .* cos.(θ)
    inner_y = R_inner .* sin.(θ)

    outer_x = R_outer .* cos.(θ)
    outer_y = R_outer .* sin.(θ)

    p = Plots.plot(
        inner_x, inner_y;
        label = "Inner",
        linewidth = 2.0,
        color = :deepskyblue,
        aspect_ratio = 1
    )

    Plots.plot!(
        p,
        outer_x, outer_y;
        label = "Outer",
        linewidth = 2.0,
        color = :coral
    )

    drew_light_label = false
    for zone in light_zones
        θs = range(zone.center - zone.half_angle,
                   zone.center + zone.half_angle,
                   length = 80)

        if zone.side == "inner"
            r1 = R_inner
            r2 = R_inner + light_width
        else
            r1 = R_outer - light_width
            r2 = R_outer
        end

        xs = vcat(r1 .* cos.(θs), reverse(r2 .* cos.(θs)))
        ys = vcat(r1 .* sin.(θs), reverse(r2 .* sin.(θs)))

        Plots.plot!(
            p, xs, ys;
            seriestype = :shape,
            color = :gold,
            fillalpha = 0.35,
            linealpha = 0.10,
            label = drew_light_label ? false : "Light zones"
        )
        drew_light_label = true
    end

    drew_block_label = false
    for block in forbidden_blocks
        θs = range(block.theta_center - block.half_angle,
                   block.theta_center + block.half_angle,
                   length = 80)

        if block.side == "inner"
            r1 = R_inner
            r2 = R_inner + block.radial_depth
        else
            r1 = R_outer - block.radial_depth
            r2 = R_outer
        end

        xs = vcat(r1 .* cos.(θs), reverse(r2 .* cos.(θs)))
        ys = vcat(r1 .* sin.(θs), reverse(r2 .* sin.(θs)))

        Plots.plot!(
            p, xs, ys;
            seriestype = :shape,
            color = :gray70,
            fillalpha = 0.45,
            linealpha = 0.12,
            label = drew_block_label ? false : "Forbidden blocks"
        )
        drew_block_label = true
    end

    apply_paperstyle_axes!(p)
    return p
end

function draw_clipped_covariance_ellipse!(
    p,
    x::Float64,
    y::Float64,
    P::AbstractMatrix{<:Float64};
    n_sigma::Float64 = 2.0,
    n_pts::Int = 180,
    color = :auto,
    linewidth::Float64 = 1.2,
    alpha::Float64 = 0.55,
    forbidden_blocks = forbidden_blocks
)
    vals, vecs = eigen(Symmetric(P))
    vals = max.(vals, 1e-10)

    θ = range(0, 2π, length = n_pts)
    circle = [cos.(θ)'; sin.(θ)']
    shape = n_sigma .* vecs * Diagonal(sqrt.(vals)) * circle

    xe = x .+ shape[1, :]
    ye = y .+ shape[2, :]

    feasible = [point_is_feasible(xe[k], ye[k]; forbidden_blocks = forbidden_blocks) for k in eachindex(xe)]

    start_idx = nothing
    for k in 1:length(xe)
        if feasible[k] && start_idx === nothing
            start_idx = k
        elseif (!feasible[k] || k == length(xe)) && start_idx !== nothing
            end_idx = feasible[k] && k == length(xe) ? k : k - 1

            if end_idx >= start_idx
                Plots.plot!(
                    p,
                    xe[start_idx:end_idx],
                    ye[start_idx:end_idx];
                    color = color,
                    linewidth = linewidth,
                    alpha = alpha,
                    label = false
                )
            end

            start_idx = nothing
        end
    end

    if feasible[1] && feasible[end]
        first_bad = findfirst(.!feasible)
        if first_bad !== nothing
            tail_start = findlast(.!feasible) + 1
            head_end = first_bad - 1

            if tail_start <= length(xe) && head_end >= 1
                xwrap = vcat(xe[tail_start:end], xe[1:head_end])
                ywrap = vcat(ye[tail_start:end], ye[1:head_end])

                Plots.plot!(
                    p,
                    xwrap,
                    ywrap;
                    color = color,
                    linewidth = linewidth,
                    alpha = alpha,
                    label = false
                )
            end
        end
    end

    return p
end

function add_agent_connections!(
    p,
    traj_a::Vector{Tuple{Float64,Float64}},
    traj_b::Vector{Tuple{Float64,Float64}};
    stride::Int = 5
)
    K = min(length(traj_a), length(traj_b))

    for k in 1:stride:K
        xa, ya = traj_a[k]
        xb, yb = traj_b[k]

        Plots.plot!(
            p,
            [xa, xb],
            [ya, yb];
            linestyle = :dash,
            linewidth = 0.8,
            color = :black,
            alpha = 0.30,
            label = false
        )
    end

    return p
end

# ============================================================
# ACTIVE PLOTS
# ============================================================

function plot_stage10_two_agent(log::TwoAgentLog; ellipse_stride::Int = 10)
    p = plot_track()

    Plots.plot!(p, first.(log.blue_true), last.(log.blue_true);
    linewidth = 2.4, color = :blue, label = "blue true")

    Plots.plot!(p, first.(log.red_true), last.(log.red_true);
    linewidth = 2.4, color = :red, label = "red true")

    Plots.plot!(p, first.(log.blue_self_mean), last.(log.blue_self_mean);
    linestyle = :dash, linewidth = 2.0, color = :blue, label = "blue self belief")

    Plots.plot!(p, first.(log.red_self_mean), last.(log.red_self_mean);
    linestyle = :dash, linewidth = 2.0, color = :red, label = "red self belief")

    Plots.plot!(p, first.(log.blue_on_red_mean), last.(log.blue_on_red_mean);
    linestyle = :dash, linewidth = 2.0, color = :purple, label = "blue belief on red")

    Plots.plot!(p, first.(log.red_on_blue_mean), last.(log.red_on_blue_mean);
        linestyle = :dash, linewidth = 2.0, color = :forestgreen, label = "red belief on blue")

    add_agent_connections!(p, log.blue_true, log.red_true; stride = 5)

    for k in 1:ellipse_stride:length(log.blue_self_cov)
        xb, yb = log.blue_self_mean[k]
        draw_clipped_covariance_ellipse!(p, xb, yb, log.blue_self_cov[k];
            n_sigma = 2.0, color = :blue)

        xr, yr = log.red_self_mean[k]
        draw_clipped_covariance_ellipse!(p, xr, yr, log.red_self_cov[k];
            n_sigma = 2.0, color = :red)

        xbr, ybr = log.blue_on_red_mean[k]
        draw_clipped_covariance_ellipse!(p, xbr, ybr, log.blue_on_red_cov[k];
            n_sigma = 2.0, color = :purple)

        xrb, yrb = log.red_on_blue_mean[k]
        draw_clipped_covariance_ellipse!(p, xrb, yrb, log.red_on_blue_cov[k];
            n_sigma = 2.0, color = :forestgreen)
    end

    apply_paperstyle_axes!(p; title_str = "Stage 10: Two-Agent Belief Bookkeeping")
    Plots.savefig(p, "stage10_two_agent.pdf")
    return p
end

# ============================================================
# STAGE 11 PLOTS — NOMINAL BELIEF TRAJECTORIES
# ============================================================

function plot_stage11_nominal_belief_rollout(
    rollout::NominalTwoPlayerRollout;
    ellipse_stride::Int = 2
)
    p = plot_track()

    xb = [step.blue_self_mean[1] for step in rollout.steps]
    yb = [step.blue_self_mean[2] for step in rollout.steps]

    xr = [step.red_self_mean[1] for step in rollout.steps]
    yr = [step.red_self_mean[2] for step in rollout.steps]

    xbr = [step.blue_on_red_mean[1] for step in rollout.steps]
    ybr = [step.blue_on_red_mean[2] for step in rollout.steps]

    xrb = [step.red_on_blue_mean[1] for step in rollout.steps]
    yrb = [step.red_on_blue_mean[2] for step in rollout.steps]

    Plots.plot!(p, xb, yb; linewidth = 2.4, color = :blue, label = "blue nominal self")
    Plots.plot!(p, xr, yr; linewidth = 2.4, color = :red, label = "red nominal self")

    Plots.plot!(p, xbr, ybr; linestyle = :dash, linewidth = 2.0, color = :purple, label = "blue nominal belief on red")
    Plots.plot!(p, xrb, yrb; linestyle = :dash, linewidth = 2.0, color = :forestgreen, label = "red nominal belief on blue")

    for k in 1:ellipse_stride:length(rollout.steps)
        step = rollout.steps[k]

        draw_clipped_covariance_ellipse!(p,
            step.blue_self_mean[1], step.blue_self_mean[2], step.blue_self_cov[1:2, 1:2];
            n_sigma = 2.0, color = :blue)

        draw_clipped_covariance_ellipse!(p,
            step.red_self_mean[1], step.red_self_mean[2], step.red_self_cov[1:2, 1:2];
            n_sigma = 2.0, color = :red)

        draw_clipped_covariance_ellipse!(p,
            step.blue_on_red_mean[1], step.blue_on_red_mean[2], step.blue_on_red_cov[1:2, 1:2];
            n_sigma = 2.0, color = :purple)

        draw_clipped_covariance_ellipse!(p,
            step.red_on_blue_mean[1], step.red_on_blue_mean[2], step.red_on_blue_cov[1:2, 1:2];
            n_sigma = 2.0, color = :forestgreen)
    end

    apply_paperstyle_axes!(p; title_str = "Stage 11: Nominal Two-Player Belief Rollout")
    Plots.savefig(p, "stage11_nominal_belief_rollout.pdf")
    return p
end

function plot_stage10_truth_only(log::TwoAgentLog)
    p = plot_track()

    bx = first.(log.blue_true)
    by = last.(log.blue_true)
    rx = first.(log.red_true)
    ry = last.(log.red_true)

    Plots.plot!(p, bx, by; linewidth = 2.4, color = :blue, label = "blue true")
    Plots.plot!(p, rx, ry; linewidth = 2.4, color = :red, label = "red true")

    Plots.scatter!(p, bx[1:10:end], by[1:10:end];
        markersize = 3.5,
        markerstrokewidth = 1.0,
        markercolor = :white,
        markerstrokecolor = :blue,
        label = "blue true samples")

    Plots.scatter!(p, rx[1:10:end], ry[1:10:end];
        markersize = 3.5,
        markerstrokewidth = 1.0,
        markercolor = :white,
        markerstrokecolor = :red,
        label = "red true samples")

    add_agent_connections!(p, log.blue_true, log.red_true; stride = 5)

    apply_paperstyle_axes!(p; title_str = "Stage 10 Truth-Only Validation")
    Plots.savefig(p, "stage10_truth_only.pdf")
    return p
end

function plot_stage10_traces(log::TwoAgentLog)
    p = Plots.plot(log.t, log.tr_blue_self;
        linewidth = 2.0,
        color = :blue,
        label = "tr blue self",
        xlabel = "time [s]",
        ylabel = "trace",
        title = "Stage 10 covariance traces")

        Plots.plot!(p, log.t, log.tr_red_self;
        linewidth = 2.0,
        color = :red,
        label = "tr red self")

        Plots.plot!(p, log.t, log.tr_blue_on_red;
        linewidth = 2.0,
        color = :forestgreen,
        label = "tr blue on red")

        Plots.plot!(p, log.t, log.tr_red_on_blue;
        linewidth = 2.0,
        color = :purple,
        label = "tr red on blue")

        Plots.plot!(
        p;
        legend = :topleft,
        size = (900, 500),
        dpi = 300,
        framestyle = :box,
        grid = true,
        background_color = :white
    )

    Plots.savefig(p, "stage10_covariance_traces.pdf")
    return p
end

function plot_stage10_collision_vs_distance(log::TwoAgentLog)
    p = Plots.plot(log.t, log.distance_true;
        linewidth = 2.2,
        color = :blue,
        label = "true distance",
        xlabel = "time [s]",
        ylabel = "value",
        title = "Stage 10: Distance vs Belief-Based Collision Risk")

        Plots.plot!(p, log.t, log.collision_cost_blue;
        linewidth = 2.0,
        linestyle = :dash,
        color = :orangered3,
        label = "blue collision cost")

        Plots.plot!(p, log.t, log.collision_cost_red;
        linewidth = 2.0,
        linestyle = :dash,
        color = :forestgreen,
        label = "red collision cost")

        Plots.plot!(
        p;
        legend = :topleft,
        size = (900, 500),
        dpi = 300,
        framestyle = :box,
        grid = true,
        background_color = :white
    )

    Plots.savefig(p, "stage10_collision_vs_distance.pdf")
    return p
end

function plot_stage10_controls(log::TwoAgentLog)
    blue_at = [u[1] for u in log.u_blue_hist]
    blue_ar = [u[2] for u in log.u_blue_hist]
    red_at  = [u[1] for u in log.u_red_hist]
    red_ar  = [u[2] for u in log.u_red_hist]

    p = Plots.plot(log.t, blue_at;
        linewidth = 2.0,
        color = :blue,
        label = "blue a_t",
        xlabel = "time [s]",
        ylabel = "control",
        title = "Stage 10 chosen controls")

        Plots.plot!(p, log.t, blue_ar;
        linewidth = 2.0,
        linestyle = :dash,
        color = :orangered3,
        label = "blue a_r")

        Plots.plot!(p, log.t, red_at;
        linewidth = 2.0,
        color = :forestgreen,
        label = "red a_t")

        Plots.plot!(p, log.t, red_ar;
        linewidth = 2.0,
        linestyle = :dash,
        color = :purple,
        label = "red a_r")

        Plots.plot!(
        p;
        legend = :topright,
        size = (900, 500),
        dpi = 300,
        framestyle = :box,
        grid = true,
        background_color = :white
    )

    Plots.savefig(p, "stage10_controls.pdf")
    return p
end

# ============================================================
# STAGE 11 PLOTS — NOMINAL COSTS
# ============================================================

function plot_stage11_nominal_costs(rollout::NominalTwoPlayerRollout)
    k_hist = collect(1:length(rollout.steps))

    p = Plots.plot(
        k_hist, rollout.J_blue_stage_hist;
        linewidth = 2.0,
        color = :blue,
        label = "blue stage cost",
        xlabel = "horizon step",
        ylabel = "cost",
        title = "Stage 11: Nominal Per-Step Costs"
    )

    Plots.plot!(p, k_hist, rollout.J_red_stage_hist;
        linewidth = 2.0,
        color = :red,
        label = "red stage cost")

    Plots.plot!(p, k_hist, rollout.J_blue_coll_hist;
        linewidth = 2.0,
        linestyle = :dash,
        color = :purple,
        label = "blue collision term")

    Plots.plot!(p, k_hist, rollout.J_red_coll_hist;
        linewidth = 2.0,
        linestyle = :dash,
        color = :forestgreen,
        label = "red collision term")

    Plots.plot!(
        p;
        legend = :topright,
        size = (900, 500),
        dpi = 300,
        framestyle = :box,
        grid = true,
        background_color = :white
    )

    Plots.savefig(p, "stage11_nominal_costs.pdf")
    return p
end

function print_stage11_nominal_cost_summary(rollout::NominalTwoPlayerRollout)
    println("Nominal blue total cost = ", round(rollout.J_blue_total, digits = 4))
    println("Nominal red total cost  = ", round(rollout.J_red_total, digits = 4))
end

# ============================================================
# DIAGNOSTIC HELPERS
# ============================================================

function min_true_block_clearance(traj::Vector{Tuple{Float64,Float64}})
    min_clear = Inf

    for (x, y) in traj
        r = hypot(x, y)
        θ = atan(y, x)
        θ = θ < 0 ? θ + 2π : θ

        for block in forbidden_blocks
            dθ = abs(mod(θ - block.theta_center + π, 2π) - π)

            if dθ <= block.half_angle
                if block.side == "inner"
                    clear = r - (R_inner + block.radial_depth)
                    min_clear = min(min_clear, clear)
                elseif block.side == "outer"
                    clear = (R_outer - block.radial_depth) - r
                    min_clear = min(min_clear, clear)
                end
            end
        end
    end

    return min_clear
end

function make_empty_two_agent_log()
    return TwoAgentLog(
        Float64[],

        # true states
        Vector{Any}(), Vector{Any}(),

        # belief means
        Vector{Any}(), Vector{Any}(),
        Vector{Any}(), Vector{Any}(),

        # covariances
        Vector{Any}(), Vector{Any}(),
        Vector{Any}(), Vector{Any}(),

        # traces
        Float64[], Float64[], Float64[], Float64[],

        # collision cost & distance
        Float64[], Float64[], Float64[],

        # controls
        Vector{Vector{Float64}}(), Vector{Vector{Float64}}(),

        # visibility + measurements
        Bool[], Bool[],
        Vector{Any}(), Vector{Any}(),

        # PCA prediction (self + cross)
        Bool[], Bool[], Bool[], Bool[],
        Int[],  Int[],  Int[],  Int[],
        Float64[], Float64[], Float64[], Float64[],

        # PCA update (self + cross)
        Bool[], Bool[], Bool[], Bool[],
        Int[],  Int[],  Int[],  Int[],
        Float64[], Float64[], Float64[], Float64[],

        # block + wall clearance
        Float64[], Float64[],
        Float64[], Float64[],
        Float64[], Float64[],

        # block logic
        Bool[], Bool[],   # critical
        Bool[], Bool[],   # wrong side
        Bool[], Bool[],   # safe side

        # NEW — near sector
        Bool[], Bool[],

        # errors
        Float64[], Float64[], Float64[], Float64[],

        # NEW — control diagnostics
        Float64[], Float64[],

        # future risk + events
        Float64[], Float64[],
        Int[], Int[],

        # NEW — extended states (for CSV reconstruction)
        Float64[], Float64[],   # blue_true_x, blue_true_y
        Float64[], Float64[],   # red_true_x, red_true_y

        Float64[], Float64[],   # blue_true_vx, blue_true_vy
        Float64[], Float64[],   # red_true_vx, red_true_vy

        Float64[], Float64[],   # blue_self_x, blue_self_y
        Float64[], Float64[],   # red_self_x, red_self_y

        Float64[], Float64[],   # blue_on_red_x, blue_on_red_y
        Float64[], Float64[],   # red_on_blue_x, red_on_blue_y

        Float64[],              # distance_true_hist

        Float64[], Float64[],   # meas flags (numeric)
        Float64[], Float64[],   # visibility flags (numeric)

        # nominal rollout
        Float64[], Float64[],
        Float64[], Float64[],

        # covariance components
        Float64[], Float64[], Float64[],
        Float64[], Float64[], Float64[],
        Float64[], Float64[], Float64[],
        Float64[], Float64[], Float64[]
    )
end

function wall_clearances(x::Float64, y::Float64)
    r = hypot(x, y)
    return (
        inner_clear = r - R_inner,
        outer_clear = R_outer - r
    )
end

function nearest_relevant_block_state(
    x::Float64,
    y::Float64;
    forbidden_blocks = forbidden_blocks,
    angular_margin::Float64 = 0.10,
    radial_enter_margin::Float64 = 0.75,
    radial_release_margin::Float64 = 1.25
)
    θ = atan(y, x)
    θ = θ < 0 ? θ + 2π : θ

    best_idx = 0
    best_dθ = Inf

    for (j, block) in enumerate(forbidden_blocks)
        dθ = abs(wrap_angle_pi(θ - block.theta_center))
        if dθ <= block.half_angle + angular_margin
            if dθ < best_dθ
                best_dθ = dθ
                best_idx = j
            end
        end
    end

    if best_idx == 0
        return (
            found = false,
            block_index = 0,
            side = "",
            clearance = Inf,
            wrong_side = false,
            safely_committed = false,
            near_sector = false
        )
    end

    block = forbidden_blocks[best_idx]
    st = block_pass_memory_state(
        x, y, block;
        angular_margin = angular_margin,
        radial_enter_margin = radial_enter_margin,
        radial_release_margin = radial_release_margin
    )

    if block.side == "inner"
        clear = hypot(x, y) - (R_inner + block.radial_depth)
    else
        clear = (R_outer - block.radial_depth) - hypot(x, y)
    end

    return (
        found = true,
        block_index = best_idx,
        side = block.side,
        clearance = clear,
        wrong_side = st.wrong_side,
        safely_committed = st.safely_committed,
        near_sector = st.near_sector
    )
end

function is_critical_constraint_moment(
    x::Float64,
    y::Float64;
    forbidden_blocks = forbidden_blocks,
    wall_margin::Float64 = critical_wall_margin,
    block_margin::Float64 = critical_block_margin
)
    w = wall_clearances(x, y)
    b = nearest_relevant_block_state(x, y; forbidden_blocks = forbidden_blocks)

    near_wall = (w.inner_clear <= wall_margin) || (w.outer_clear <= wall_margin)
    near_block = b.found && (b.clearance <= block_margin)

    return near_wall || near_block
end

function maybe_next_event_id(prev_flag::Bool, cur_flag::Bool, current_id::Int)
    if (!prev_flag) && cur_flag
        return current_id + 1
    else
        return current_id
    end
end

function compute_future_min_clearance_series(
    clear_hist::Vector{Float64};
    M::Int = future_risk_horizon_steps
)
    n = length(clear_hist)
    out = fill(Inf, n)

    for k in 1:n
        j2 = min(n, k + M)
        if k < j2
            out[k] = minimum(clear_hist[k+1:j2])
        else
            out[k] = clear_hist[k]
        end
    end

    return out
end

function safe_corr(x::Vector{Float64}, y::Vector{Float64})
    if length(x) != length(y) || length(x) < 2
        return NaN
    end
    if Statistics.std(x) < 1e-12 || Statistics.std(y) < 1e-12
        return NaN
    end
    return Statistics.cor(x, y)
end

function rank_transform(v::Vector{Float64})
    p = sortperm(v)
    r = similar(v, Float64)
    for (i, idx) in enumerate(p)
        r[idx] = i
    end
    return r
end

function safe_spearman(x::Vector{Float64}, y::Vector{Float64})
    if length(x) != length(y) || length(x) < 2
        return NaN
    end
    return safe_corr(rank_transform(x), rank_transform(y))
end

function block_anticipation_score(
    clear_hist::Vector{Float64},
    pca_active_hist::Vector{Bool};
    risk_threshold::Float64 = risky_clearance_threshold,
    lookback_steps::Int = anticipation_lookback_steps
)
    n = min(length(clear_hist), length(pca_active_hist))
    if n == 0
        return (score_pct = NaN, n_events = 0, n_anticipated = 0)
    end

    event_indices = Int[]
    prev_risky = false

    for k in 1:n
        cur_risky = isfinite(clear_hist[k]) && (clear_hist[k] <= risk_threshold)
        if cur_risky && !prev_risky
            push!(event_indices, k)
        end
        prev_risky = cur_risky
    end

    n_events = length(event_indices)
    if n_events == 0
        return (score_pct = NaN, n_events = 0, n_anticipated = 0)
    end

    n_anticipated = 0
    for k in event_indices
        k1 = max(1, k - lookback_steps)
        if any(pca_active_hist[k1:k])
            n_anticipated += 1
        end
    end

    return (
        score_pct = 100.0 * n_anticipated / n_events,
        n_events = n_events,
        n_anticipated = n_anticipated
    )
end

function predictive_relevance_summary(
    lambda_hist::Vector{Float64},
    active_hist::Vector{Bool},
    future_clear_hist::Vector{Float64},
    critical_hist::Vector{Bool}
)
    idx = findall(critical_hist)
    if isempty(idx)
        return (
            pearson_lambda_future_clear = NaN,
            spearman_lambda_future_clear = NaN,
            pearson_active_future_clear = NaN,
            n_points = 0
        )
    end

    xλ = [lambda_hist[i] for i in idx if isfinite(future_clear_hist[i])]
    y  = [future_clear_hist[i] for i in idx if isfinite(future_clear_hist[i])]
    xa = [active_hist[i] ? 1.0 : 0.0 for i in idx if isfinite(future_clear_hist[i])]

    return (
        pearson_lambda_future_clear = safe_corr(xλ, y),
        spearman_lambda_future_clear = safe_spearman(xλ, y),
        pearson_active_future_clear = safe_corr(xa, y),
        n_points = length(y)
    )
end

function peak_radial_action(log::TwoAgentLog)
    blue = isempty(log.u_blue_hist) ? NaN : maximum(abs(u[2]) for u in log.u_blue_hist)
    red  = isempty(log.u_red_hist)  ? NaN : maximum(abs(u[2]) for u in log.u_red_hist)
    return (blue = blue, red = red)
end

function percentage_reduction(baseline_val::Float64, pca_val::Float64)
    if !isfinite(baseline_val) || abs(baseline_val) < 1e-12
        return NaN
    end
    return 100.0 * (baseline_val - pca_val) / baseline_val
end

function print_pca_summary(log::TwoAgentLog)
    n = length(log.t)

    if n == 0
        println("PCA summary: empty log")
        return nothing
    end

    function frac_true(v::Vector{Bool})
        return isempty(v) ? 0.0 : sum(v) / length(v)
    end

    function mean_or_zero(v)
        return isempty(v) ? 0.0 : sum(v) / length(v)
    end

    println("\n=== PCA summary ===")
    println("Estimator mode = ", estimator_mode[])

    # ========================
    # PREDICTION
    # ========================
    println("\n--- PCA (Prediction) ---")

    println("Blue self   | active = ", round(frac_true(log.pca_pred_blue_self_active), digits=3),
            " | rank = ", round(mean_or_zero(log.pca_pred_blue_self_rank), digits=3),
            " | λmax = ", round(mean_or_zero(log.pca_pred_blue_self_lambda_max), digits=6))

    println("Red self    | active = ", round(frac_true(log.pca_pred_red_self_active), digits=3),
            " | rank = ", round(mean_or_zero(log.pca_pred_red_self_rank), digits=3),
            " | λmax = ", round(mean_or_zero(log.pca_pred_red_self_lambda_max), digits=6))

    println("Blue on red | active = ", round(frac_true(log.pca_pred_blue_on_red_active), digits=3),
            " | rank = ", round(mean_or_zero(log.pca_pred_blue_on_red_rank), digits=3),
            " | λmax = ", round(mean_or_zero(log.pca_pred_blue_on_red_lambda_max), digits=6))

    println("Red on blue | active = ", round(frac_true(log.pca_pred_red_on_blue_active), digits=3),
            " | rank = ", round(mean_or_zero(log.pca_pred_red_on_blue_rank), digits=3),
            " | λmax = ", round(mean_or_zero(log.pca_pred_red_on_blue_lambda_max), digits=6))

    # ========================
    # UPDATE
    # ========================
    println("\n--- PCA (Update) ---")

    println("Blue self   | active = ", round(frac_true(log.pca_upd_blue_self_active), digits=3),
            " | rank = ", round(mean_or_zero(log.pca_upd_blue_self_rank), digits=3),
            " | λmax = ", round(mean_or_zero(log.pca_upd_blue_self_lambda_max), digits=6))

    println("Red self    | active = ", round(frac_true(log.pca_upd_red_self_active), digits=3),
            " | rank = ", round(mean_or_zero(log.pca_upd_red_self_rank), digits=3),
            " | λmax = ", round(mean_or_zero(log.pca_upd_red_self_lambda_max), digits=6))

    println("Blue on red | active = ", round(frac_true(log.pca_upd_blue_on_red_active), digits=3),
            " | rank = ", round(mean_or_zero(log.pca_upd_blue_on_red_rank), digits=3),
            " | λmax = ", round(mean_or_zero(log.pca_upd_blue_on_red_lambda_max), digits=6))

    println("Red on blue | active = ", round(frac_true(log.pca_upd_red_on_blue_active), digits=3),
            " | rank = ", round(mean_or_zero(log.pca_upd_red_on_blue_rank), digits=3),
            " | λmax = ", round(mean_or_zero(log.pca_upd_red_on_blue_lambda_max), digits=6))

    return nothing
end

# ============================================================
# BASELINE PLANNER DISPATCH
# ============================================================

function choose_controls!(
    ::GreedyPlanner,
    scenario::TwoAgentScenario,
    model_blue::CarModel,
    model_red::CarModel
)
    u_red_guess = [0.1, 0.0]

    u_blue, _ = choose_greedy_control_symmetric(
        scenario, :blue, u_red_guess, model_blue, model_red
    )

    u_blue_guess = u_blue

    u_red, _ = choose_greedy_control_symmetric(
        scenario, :red, u_blue_guess, model_blue, model_red
    )

    return u_blue, u_red
end

# ============================================================
# MAIN STAGE 10 RUNNER
# ============================================================

function run_stage10_baseline(; N_test::Int = N_test_default)
    println("=== Stage 10: Clean Two-Agent Greedy Baseline ===")

    model_blue = CarModel(c_drag_default, c_slip_default)
    model_red  = CarModel(c_drag_default, c_slip_default)

    blue_true = make_agent_on_track(theta0_default, r0_default, v_t0_default)
    red_true  = make_agent_on_track(theta0_default + 0.25, r0_default + 1.0, v_t0_default)

    scenario = create_two_agent_scenario(blue_true, red_true)
    planner = GreedyPlanner()
    log = make_empty_two_agent_log()

    prog = Progress(N_test, desc = "Stage 10 simulation")

    for k in 1:N_test
        t = (k - 1) * dt

        u_blue, u_red = choose_controls!(planner, scenario, model_blue, model_red)

        push!(log.u_blue_hist, copy(u_blue))
        push!(log.u_red_hist, copy(u_red))
        push!(log.t, t)

        push!(log.blue_true, (scenario.blue_true.x, scenario.blue_true.y))
        push!(log.red_true,  (scenario.red_true.x,  scenario.red_true.y))

        push!(log.blue_self_mean, (scenario.blue_self_belief.mean[1], scenario.blue_self_belief.mean[2]))
        push!(log.red_self_mean,  (scenario.red_self_belief.mean[1],  scenario.red_self_belief.mean[2]))

        push!(log.blue_on_red_mean, (scenario.blue_belief_on_red.mean[1], scenario.blue_belief_on_red.mean[2]))
        push!(log.red_on_blue_mean, (scenario.red_belief_on_blue.mean[1], scenario.red_belief_on_blue.mean[2]))

        push!(log.blue_self_cov, copy(scenario.blue_self_belief.cov[1:2, 1:2]))
        push!(log.red_self_cov,  copy(scenario.red_self_belief.cov[1:2, 1:2]))

        push!(log.blue_on_red_cov, copy(scenario.blue_belief_on_red.cov[1:2, 1:2]))
        push!(log.red_on_blue_cov, copy(scenario.red_belief_on_blue.cov[1:2, 1:2]))

        push!(log.tr_blue_self, tr(scenario.blue_self_belief.cov[1:2, 1:2]))
        push!(log.tr_red_self,  tr(scenario.red_self_belief.cov[1:2, 1:2]))
        push!(log.tr_blue_on_red, tr(scenario.blue_belief_on_red.cov[1:2, 1:2]))
        push!(log.tr_red_on_blue, tr(scenario.red_belief_on_blue.cov[1:2, 1:2]))

        dtrue = hypot(
            scenario.blue_true.x - scenario.red_true.x,
            scenario.blue_true.y - scenario.red_true.y
        )
        push!(log.distance_true, dtrue)

        push!(log.collision_cost_blue,
            two_agent_collision_cost(
                scenario.blue_self_belief,
                scenario.blue_belief_on_red;
                w_c = w_coll_default,
                d_safe = d_safe_coll_default,
                σ_coll = σ_coll_default,
                k_unc = k_unc_coll_default
            )
        )

        push!(log.collision_cost_red,
            two_agent_collision_cost(
                scenario.red_self_belief,
                scenario.red_belief_on_blue;
                w_c = w_coll_default,
                d_safe = d_safe_coll_default,
                σ_coll = σ_coll_default,
                k_unc = k_unc_coll_default
            )
        )

        propagate_two_agent_truth!(scenario, u_blue, u_red, model_blue, model_red)
        predict_two_agent_beliefs!(scenario, u_blue, u_red, model_blue, model_red)
        got_blue_self, got_red_self, got_blue_on_red, got_red_on_blue =
            update_two_agent_beliefs_stage10!(scenario)

        push!(log.blue_self_meas_hist, got_blue_self)
        push!(log.red_self_meas_hist, got_red_self)
        push!(log.blue_saw_red_hist, got_blue_on_red)
        push!(log.red_saw_blue_hist, got_red_on_blue)

        push!(log.pca_pred_blue_self_active,   last_pca_pred_blue_self[].active)
        push!(log.pca_pred_red_self_active,    last_pca_pred_red_self[].active)
        push!(log.pca_pred_blue_on_red_active, last_pca_pred_blue_on_red[].active)
        push!(log.pca_pred_red_on_blue_active, last_pca_pred_red_on_blue[].active)

        push!(log.pca_pred_blue_self_rank,   last_pca_pred_blue_self[].retained_rank)
        push!(log.pca_pred_red_self_rank,    last_pca_pred_red_self[].retained_rank)
        push!(log.pca_pred_blue_on_red_rank, last_pca_pred_blue_on_red[].retained_rank)
        push!(log.pca_pred_red_on_blue_rank, last_pca_pred_red_on_blue[].retained_rank)

        push!(log.pca_pred_blue_self_lambda_max,
            isempty(last_pca_pred_blue_self[].eigvals) ? 0.0 : last_pca_pred_blue_self[].eigvals[1]
        )
        push!(log.pca_pred_red_self_lambda_max,
            isempty(last_pca_pred_red_self[].eigvals) ? 0.0 : last_pca_pred_red_self[].eigvals[1]
        )
        push!(log.pca_pred_blue_on_red_lambda_max,
            isempty(last_pca_pred_blue_on_red[].eigvals) ? 0.0 : last_pca_pred_blue_on_red[].eigvals[1]
        )
        push!(log.pca_pred_red_on_blue_lambda_max,
            isempty(last_pca_pred_red_on_blue[].eigvals) ? 0.0 : last_pca_pred_red_on_blue[].eigvals[1]
        )

        push!(log.pca_upd_blue_self_active,   last_pca_upd_blue_self[].active)
        push!(log.pca_upd_red_self_active,    last_pca_upd_red_self[].active)
        push!(log.pca_upd_blue_on_red_active, last_pca_upd_blue_on_red[].active)
        push!(log.pca_upd_red_on_blue_active, last_pca_upd_red_on_blue[].active)

        push!(log.pca_upd_blue_self_rank,   last_pca_upd_blue_self[].retained_rank)
        push!(log.pca_upd_red_self_rank,    last_pca_upd_red_self[].retained_rank)
        push!(log.pca_upd_blue_on_red_rank, last_pca_upd_blue_on_red[].retained_rank)
        push!(log.pca_upd_red_on_blue_rank, last_pca_upd_red_on_blue[].retained_rank)

        push!(log.pca_upd_blue_self_lambda_max,
            isempty(last_pca_upd_blue_self[].eigvals) ? 0.0 : last_pca_upd_blue_self[].eigvals[1]
        )
        push!(log.pca_upd_red_self_lambda_max,
            isempty(last_pca_upd_red_self[].eigvals) ? 0.0 : last_pca_upd_red_self[].eigvals[1]
        )
        push!(log.pca_upd_blue_on_red_lambda_max,
            isempty(last_pca_upd_blue_on_red[].eigvals) ? 0.0 : last_pca_upd_blue_on_red[].eigvals[1]
        )
        push!(log.pca_upd_red_on_blue_lambda_max,
            isempty(last_pca_upd_red_on_blue[].eigvals) ? 0.0 : last_pca_upd_red_on_blue[].eigvals[1]
        )

        next!(prog)
    end

    println("Blue self belief error = ",
        round(norm(scenario.blue_self_belief.mean[1:2] - [scenario.blue_true.x, scenario.blue_true.y]), digits = 4))
    println("Red self belief error = ",
        round(norm(scenario.red_self_belief.mean[1:2] - [scenario.red_true.x, scenario.red_true.y]), digits = 4))
    println("Blue belief on red error = ",
        round(norm(scenario.blue_belief_on_red.mean[1:2] - [scenario.red_true.x, scenario.red_true.y]), digits = 4))
    println("Red belief on blue error = ",
        round(norm(scenario.red_belief_on_blue.mean[1:2] - [scenario.blue_true.x, scenario.blue_true.y]), digits = 4))

    p1 = plot_stage10_two_agent(log; ellipse_stride = 10)
    display(p1)

    p_truth = plot_stage10_truth_only(log)
    display(p_truth)

    p2 = plot_stage10_traces(log)
    display(p2)

    p3 = plot_stage10_collision_vs_distance(log)
    display(p3)

    p4 = plot_stage10_controls(log)
    display(p4)

    println("Final true distance = ", round(log.distance_true[end], digits = 4))
    println("Final blue collision cost = ", round(log.collision_cost_blue[end], digits = 4))
    println("Final red collision cost  = ", round(log.collision_cost_red[end], digits = 4))
    println("Blue min block clearance = ", round(min_true_block_clearance(log.blue_true), digits = 4))
    println("Red min block clearance  = ", round(min_true_block_clearance(log.red_true), digits = 4))
    println("=== Done ===")

    return log
end

function make_stage11_reference_problem()
    model_blue = CarModel(c_drag_default, c_slip_default)
    model_red  = CarModel(c_drag_default, c_slip_default)

    blue_true = make_agent_on_track(theta0_default, r0_default, v_t0_default)
    red_true  = make_agent_on_track(theta0_default + 0.25, r0_default + 1.0, v_t0_default)

    scenario0 = create_two_agent_scenario(blue_true, red_true)

    return scenario0, model_blue, model_red
end

# ============================================================
# STAGE 11 DEMO RUNNER
# ============================================================

function run_stage11_nominal_rollout(; H::Int = H)
    println("=== Stage 11: Two-Player Nominal Sequence Rollout ===")

    model_blue = CarModel(c_drag_default, c_slip_default)
    model_red  = CarModel(c_drag_default, c_slip_default)

    blue_true = make_agent_on_track(theta0_default, r0_default, v_t0_default)
    red_true  = make_agent_on_track(theta0_default + 0.25, r0_default + 1.0, v_t0_default)

    scenario0 = create_two_agent_scenario(blue_true, red_true)

    u_blue_seq, u_red_seq = initialize_two_player_nominal_sequences(
        H;
        u_blue0 = [0.10, 0.0],
        u_red0  = [0.10, 0.0]
    )

    rollout = rollout_two_player_nominal_sequences(
        scenario0,
        u_blue_seq,
        u_red_seq,
        model_blue,
        model_red
    )

    print_stage11_nominal_cost_summary(rollout)

    p1 = plot_stage11_nominal_belief_rollout(rollout; ellipse_stride = 2)
    display(p1)

    p2 = plot_stage11_nominal_costs(rollout)
    display(p2)

    println("\n=== Stage 11 quick diagnostic ===")
    for k in 1:min(10, length(rollout.J_blue_stage_hist))
        println(
            "k=", k,
            " | blue J=", round(rollout.J_blue_stage_hist[k], digits=3),
            " | red J=", round(rollout.J_red_stage_hist[k], digits=3),
            " | blue coll=", round(rollout.J_blue_coll_hist[k], digits=3),
            " | red coll=", round(rollout.J_red_coll_hist[k], digits=3)
        )
    end

    println("Nominal blue total cost = ", round(rollout.J_blue_total, digits=4))
    println("Nominal red total cost  = ", round(rollout.J_red_total, digits=4))
    println("=== Done ===")

    return rollout
end

# ============================================================
# STAGE 12 — DEBUG / VERIFICATION
# ============================================================

function safe_cond(M::AbstractMatrix{<:Float64}; eps_cond::Float64 = 1e-12)
    s = svdvals(Matrix(M))
    smax = maximum(s)
    smin = minimum(s)

    if smin < eps_cond
        return Inf
    else
        return smax / smin
    end
end

function block_clearance_to_inner_obstacle(
    x::Float64,
    y::Float64;
    forbidden_blocks = forbidden_blocks,
    R_inner::Float64 = R_inner
)
    r = hypot(x, y)
    θ = atan(y, x)
    θ = θ < 0 ? θ + 2π : θ

    best_clearance = Inf

    for block in forbidden_blocks
        if block.side != "inner"
            continue
        end

        dθ = mod(θ - block.theta_center + π, 2π) - π

        if abs(dθ) <= block.half_angle
            clearance = r - (R_inner + block.radial_depth)
            best_clearance = min(best_clearance, clearance)
        end
    end

    return best_clearance
end

function block_is_still_dangerous(
    x::Float64,
    y::Float64,
    block;
    R_inner::Float64 = R_inner,
    R_outer::Float64 = R_outer,
    angle_release_margin::Float64 = 0.08,
    radial_release_margin::Float64 = 0.40
)
    r = hypot(x, y)
    θ = atan(y, x)
    θ = θ < 0 ? θ + 2π : θ

    # trailing edge of the block in the counterclockwise direction
    θ_trailing = mod(block.theta_center + block.half_angle, 2π)

    # forward angular distance from the trailing edge to the current agent angle
    # if this is positive and larger than margin, the agent has passed the block
    dθ_after_trailing = mod(θ - θ_trailing, 2π)

    passed_angularly = dθ_after_trailing >= angle_release_margin

    if block.side == "inner"
        block_edge = R_inner + block.radial_depth
        radially_safe = r >= block_edge + radial_release_margin
    elseif block.side == "outer"
        block_edge = R_outer - block.radial_depth
        radially_safe = r <= block_edge - radial_release_margin
    else
        error("Unknown block side $(block.side)")
    end

    still_in_sector = abs(wrap_angle_pi(θ - block.theta_center)) <= (block.half_angle + angle_release_margin)

    # dangerous if:
    # 1) still inside/near the block sector, or
    # 2) already slightly past it but not yet on the safe radial side
    still_dangerous = still_in_sector || (!passed_angularly ? true : !radially_safe)

    return still_dangerous
end

function print_stage12_local_model_debug(
    k::Int,
    dyn::Stage12LocalDynamicsModel,
    quad_blue::Stage12LocalQuadraticCost,
    quad_red::Stage12LocalQuadraticCost
)
    println("\n=== Stage 12 local-model debug | k = ", k, " ===")

    println("Dynamics model:")
    println("  size(G_b)  = ", size(dyn.G_b))
    println("  size(G_uB) = ", size(dyn.G_uB))
    println("  size(G_uR) = ", size(dyn.G_uR))
    println("  cond(G_b)  = ", round(safe_cond(dyn.G_b), digits = 4))
    println("  cond(G_uB) = ", round(safe_cond(dyn.G_uB), digits = 4))
    println("  cond(G_uR) = ", round(safe_cond(dyn.G_uR), digits = 4))

    println("Blue quadratic cost:")
    println("  size(q) = ", size(quad_blue.q))
    println("  size(Q) = ", size(quad_blue.Q))
    println("  cond(Q) = ", round(safe_cond(quad_blue.Q), digits = 4))

    println("Red quadratic cost:")
    println("  size(q) = ", size(quad_red.q))
    println("  size(Q) = ", size(quad_red.Q))
    println("  cond(Q) = ", round(safe_cond(quad_red.Q), digits = 4))
end

function run_stage12_local_linearization_debug(
    rollout::NominalTwoPlayerRollout,
    scenario0::TwoAgentScenario,
    model_blue::CarModel,
    model_red::CarModel;
    step_indices::Vector{Int} = [1, 3, 5]
)
    println("=== Stage 12: Two-Player Local Linearization / Quadraticization ===")

    Kmax = length(rollout.steps)

    for k in step_indices
        if !(1 <= k <= Kmax)
            println("Skipping k=", k, " (outside rollout horizon)")
            continue
        end

        step = rollout.steps[k]

        b0, sc_ref = joint_belief_from_stage11_step(step, scenario0)
        uB0 = copy(step.u_blue)
        uR0 = copy(step.u_red)

        dyn = linearize_two_player_belief_dynamics_fd(
            b0, uB0, uR0, sc_ref, model_blue, model_red
        )

        quad_blue = quadraticize_two_player_stage_cost_fd(
            b0, uB0, uR0, :blue, sc_ref
        )

        quad_red = quadraticize_two_player_stage_cost_fd(
            b0, uB0, uR0, :red, sc_ref
        )

        print_stage12_local_model_debug(k, dyn, quad_blue, quad_red)
    end

    println("=== Done ===")
end

# ============================================================
# STAGE 13 — IMPROVED TWO-PLAYER FORWARD ROLLOUT
# ============================================================

function rollout_stage13_one_step_game_improvement(
    scenario0::TwoAgentScenario,
    nominal_rollout::NominalTwoPlayerRollout,
    model_blue::CarModel,
    model_red::CarModel;
    α_step::Float64 = 0.20
)
    H = length(nominal_rollout.steps)

    u_blue_new_seq = Vector{Vector{Float64}}(undef, H)
    u_red_new_seq  = Vector{Vector{Float64}}(undef, H)

    step_results = Stage13StepResult[]

    for k in 1:H
        step = nominal_rollout.steps[k]

        b0, sc_ref = joint_belief_from_stage11_step(step, scenario0)

        uB_nom = copy(step.u_blue)
        uR_nom = copy(step.u_red)

        quad_blue = quadraticize_two_player_stage_cost_fd(
            b0, uB_nom, uR_nom, :blue, sc_ref
        )

        quad_red = quadraticize_two_player_stage_cost_fd(
            b0, uB_nom, uR_nom, :red, sc_ref
        )

        du_blue, du_red = solve_stage13_static_game(quad_blue, quad_red)

        uB_new = clamp_control_vector(
            uB_nom + α_step * du_blue,
            step.blue_self_mean[1],
            step.blue_self_mean[2]
        )

        uR_new = clamp_control_vector(
            uR_nom + α_step * du_red,
            step.red_self_mean[1],
            step.red_self_mean[2]
        )

        blue_cost_nom = step.J_blue_stage
        red_cost_nom  = step.J_red_stage

        blue_cost_quad_pred = quad_blue.c + dot(quad_blue.q, vcat(zeros(quad_blue.n_b), uB_new - uB_nom, uR_new - uR_nom))
        red_cost_quad_pred  = quad_red.c  + dot(quad_red.q,  vcat(zeros(quad_red.n_b),  uB_new - uB_nom, uR_new - uR_nom))

        push!(step_results, Stage13StepResult(
            copy(uB_nom),
            copy(uR_nom),
            copy(du_blue),
            copy(du_red),
            copy(uB_new),
            copy(uR_new),
            blue_cost_nom,
            red_cost_nom,
            blue_cost_quad_pred,
            red_cost_quad_pred
        ))

        u_blue_new_seq[k] = uB_new
        u_red_new_seq[k]  = uR_new
    end

    improved_rollout = rollout_two_player_nominal_sequences(
        scenario0,
        u_blue_new_seq,
        u_red_new_seq,
        model_blue,
        model_red
    )

    return Stage13ImprovedRollout(nominal_rollout, improved_rollout, step_results)
end

# ============================================================
# STAGE 13 — PLOTS
# ============================================================

function plot_stage13_improved_vs_nominal(stage13::Stage13ImprovedRollout)
    p = plot_track()

    plot!(p,
        [step.blue_self_mean[1] for step in stage13.base_rollout.steps],
        [step.blue_self_mean[2] for step in stage13.base_rollout.steps];
        color = :blue,
        linestyle = :dash,
        linewidth = 2.0,
        label = "blue nominal"
    )

        Plots.plot!(p,
        [step.red_self_mean[1] for step in stage13.base_rollout.steps],
        [step.red_self_mean[2] for step in stage13.base_rollout.steps];
        color = :red,
        linestyle = :dash,
        linewidth = 2.0,
        label = "red nominal"
    )

    Plots.plot!(p,
        [step.blue_self_mean[1] for step in stage13.improved_rollout.steps],
        [step.blue_self_mean[2] for step in stage13.improved_rollout.steps];
        color = :blue,
        linewidth = 2.5,
        label = "blue improved"
    )

    Plots.plot!(p,
        [step.red_self_mean[1] for step in stage13.improved_rollout.steps],
        [step.red_self_mean[2] for step in stage13.improved_rollout.steps];
        color = :red,
        linewidth = 2.5,
        label = "red improved"
    )

    apply_paperstyle_axes!(p; title_str = "Stage 13: Nominal vs Locally Improved Rollout")
    return p
end

function plot_stage13_control_updates(stage13::Stage13ImprovedRollout)
    K = length(stage13.step_results)

    blue_du_t = [sr.du_blue[1] for sr in stage13.step_results]
    blue_du_r = [sr.du_blue[2] for sr in stage13.step_results]
    red_du_t  = [sr.du_red[1]  for sr in stage13.step_results]
    red_du_r  = [sr.du_red[2]  for sr in stage13.step_results]

    p = Plots.plot(1:K, blue_du_t;
        color = :blue,
        linewidth = 2.0,
        label = "blue Δu_t",
        xlabel = "horizon step",
        ylabel = "control update",
        title = "Stage 13: Local Nash Control Updates"
    )

    Plots.plot!(p, 1:K, blue_du_r; color = :orangered3, linestyle = :dash, linewidth = 2.0, label = "blue Δu_r")
    Plots.plot!(p, 1:K, red_du_t;  color = :forestgreen, linewidth = 2.0, label = "red Δu_t")
    Plots.plot!(p, 1:K, red_du_r;  color = :purple, linestyle = :dash, linewidth = 2.0, label = "red Δu_r")

    return p
end

# ============================================================
# STAGE 13 — RUNNER
# ============================================================

function run_stage13_one_step_game_improvement(; H::Int = H, α_step::Float64 = 0.20)
    println("=== Stage 13: One-Step Local Game Improvement ===")

    scenario0, model_blue, model_red = make_stage11_reference_problem()

    u_blue_seq, u_red_seq = initialize_two_player_nominal_sequences(
        H;
        u_blue0 = [0.10, 0.0],
        u_red0  = [0.10, 0.0]
    )

    nominal_rollout = rollout_two_player_nominal_sequences(
        scenario0,
        u_blue_seq,
        u_red_seq,
        model_blue,
        model_red
    )

    stage13 = rollout_stage13_one_step_game_improvement(
        scenario0,
        nominal_rollout,
        model_blue,
        model_red;
        α_step = α_step
    )

    println("Nominal blue total cost   = ", round(stage13.base_rollout.J_blue_total, digits = 4))
    println("Nominal red total cost    = ", round(stage13.base_rollout.J_red_total, digits = 4))
    println("Improved blue total cost  = ", round(stage13.improved_rollout.J_blue_total, digits = 4))
    println("Improved red total cost   = ", round(stage13.improved_rollout.J_red_total, digits = 4))

    p1 = plot_stage13_improved_vs_nominal(stage13)
    display(p1)

    p2 = plot_stage13_control_updates(stage13)
    display(p2)

    println("=== Done ===")
    return stage13
end

# ============================================================
# STAGE 14 — PLOTS + RUNNER
# ============================================================

function plot_stage14_correction_magnitudes(bp::Stage14BackwardPassResult)
    H = length(bp.steps)

    jb = [norm(bp.steps[k].j_blue) for k in 1:H]
    jr = [norm(bp.steps[k].j_red)  for k in 1:H]

    kb = [norm(bp.steps[k].K_blue) for k in 1:H]
    kr = [norm(bp.steps[k].K_red)  for k in 1:H]

    p = Plots.plot(1:H, jb;
        linewidth = 2.0,
        color = :blue,
        label = "||j_blue||",
        xlabel = "horizon step",
        ylabel = "magnitude",
        title = "Stage 14: Backward-Pass Correction Magnitudes"
    )

    Plots.plot!(p, 1:H, jr; linewidth = 2.0, color = :red, label = "||j_red||")
    Plots.plot!(p, 1:H, kb; linewidth = 2.0, linestyle = :dash, color = :blue, label = "||K_blue||")
    Plots.plot!(p, 1:H, kr; linewidth = 2.0, linestyle = :dash, color = :red, label = "||K_red||")

    return p
end

function plot_stage14_game_conditioning(bp::Stage14BackwardPassResult)
    H = length(bp.steps)
    vals = [bp.steps[k].cond_game_matrix for k in 1:H]

    p = Plots.plot(1:H, vals;
        linewidth = 2.0,
        color = :black,
        label = "cond(game matrix)",
        xlabel = "horizon step",
        ylabel = "condition number",
        title = "Stage 14: Local Game Matrix Conditioning"
    )

    return p
end

function run_stage14_true_backward_pass_from_rollout(
    rollout::NominalTwoPlayerRollout,
    scenario0::TwoAgentScenario,
    model_blue::CarModel,
    model_red::CarModel
)
    H = length(rollout.steps)

    n_b = length(joint_belief_from_stage11_step(rollout.steps[end], scenario0)[1])

    V_blue = Stage14ValueModel(0.0, zeros(n_b), zeros(n_b, n_b))
    V_red  = Stage14ValueModel(0.0, zeros(n_b), zeros(n_b, n_b))

    backward_steps = Vector{Stage14BackwardStep}(undef, H)

    for k in H:-1:1
        step = rollout.steps[k]

        b0, sc_ref = joint_belief_from_stage11_step(step, scenario0)
        uB0 = copy(step.u_blue)
        uR0 = copy(step.u_red)

        dyn = linearize_two_player_belief_dynamics_fd(
            b0, uB0, uR0, sc_ref, model_blue, model_red
        )

        quad_blue = quadraticize_two_player_stage_cost_fd(
            b0, uB0, uR0, :blue, sc_ref
        )

        quad_red = quadraticize_two_player_stage_cost_fd(
            b0, uB0, uR0, :red, sc_ref
        )

        _, qB, QB = stage14_build_augmented_player_model(dyn, quad_blue, V_blue)
        _, qR, QR = stage14_build_augmented_player_model(dyn, quad_red, V_red)

        n_uB = quad_blue.n_uB
        n_uR = quad_blue.n_uR

        i_b  = 1:quad_blue.n_b
        i_uB = quad_blue.n_b + 1 : quad_blue.n_b + n_uB
        i_uR = quad_blue.n_b + n_uB + 1 : quad_blue.n_b + n_uB + n_uR

        Qb_bb = QB[i_b, i_b] + 1e-8 * I
        Qb_bB = QB[i_b, i_uB]
        Qb_bR = QB[i_b, i_uR]
        Qb_Bb = QB[i_uB, i_b]
        Qb_BB = QB[i_uB, i_uB] + 1e-5 * I
        Qb_BR = QB[i_uB, i_uR]

        Qr_bb = QR[i_b, i_b] + 1e-8 * I
        Qr_bB = QR[i_b, i_uB]
        Qr_bR = QR[i_b, i_uR]
        Qr_Rb = QR[i_uR, i_b]
        Qr_RB = QR[i_uR, i_uB]
        Qr_RR = QR[i_uR, i_uR] + 1e-5 * I

        qb_b = qB[i_b]
        qb_B = qB[i_uB]

        qr_b = qR[i_b]
        qr_R = qR[i_uR]

        A_game = [
            Qb_BB   Qb_BR
            Qr_RB   Qr_RR
        ]

        B_game = [
            Qb_Bb
            Qr_Rb
        ]

        a_game = -[
            qb_B
            qr_R
        ]

        F_game = -B_game

        cond_game = stage14_safe_cond(A_game)

        sol_j = A_game \ a_game
        sol_K = A_game \ F_game

        j_blue = sol_j[1:n_uB]
        j_red  = sol_j[n_uB+1:end]

        K_blue = sol_K[1:n_uB, :]
        K_red  = sol_K[n_uB+1:end, :]

        backward_steps[k] = Stage14BackwardStep(
            j_blue, K_blue,
            j_red,  K_red,
            cond_game
        )

        Vb_b =
            qb_b +
            Qb_bB * j_blue +
            Qb_bR * j_red +
            K_blue' * qb_B

        Vr_b =
            qr_b +
            Qr_bB * j_blue +
            Qr_bR * j_red +
            K_red' * qr_R

        Vb_bb =
            Qb_bb +
            Qb_bB * K_blue +
            Qb_bR * K_red +
            K_blue' * Qb_Bb +
            K_blue' * Qb_BB * K_blue +
            K_blue' * Qb_BR * K_red

        Vr_bb =
            Qr_bb +
            Qr_bB * K_blue +
            Qr_bR * K_red +
            K_red' * Qr_Rb +
            K_red' * Qr_RB * K_blue +
            K_red' * Qr_RR * K_red

        V_blue = Stage14ValueModel(0.0, Vb_b, 0.5 * (Vb_bb + Vb_bb'))
        V_red  = Stage14ValueModel(0.0, Vr_b, 0.5 * (Vr_bb + Vr_bb'))
    end

    return Stage14BackwardPassResult(backward_steps, V_blue, V_red)
end

function run_stage14_true_backward_pass(; H::Int = H)
    println("=== Stage 14: True Two-Player Backward Pass ===")

    scenario0, model_blue, model_red = make_stage11_reference_problem()

    u_blue_seq, u_red_seq = initialize_two_player_nominal_sequences(
        H;
        u_blue0 = [0.10, 0.0],
        u_red0  = [0.10, 0.0]
    )

    rollout = rollout_two_player_nominal_sequences(
        scenario0,
        u_blue_seq,
        u_red_seq,
        model_blue,
        model_red
    )

    bp = run_stage14_true_backward_pass_from_rollout(
        rollout,
        scenario0,
        model_blue,
        model_red
    )

    p1 = plot_stage14_correction_magnitudes(bp)
    display(p1)

    p2 = plot_stage14_game_conditioning(bp)
    display(p2)

    println("=== Done ===")
    return bp
end

function local_control_box(
    x::Float64,
    y::Float64,
    mem::BlockPassMemory;
    a_t_lim::Float64 = a_t_max,
    a_r_lim::Float64 = a_r_max
)
    ar_min, ar_max = radial_action_limits_with_memory(
        x, y, mem;
        a_r_lim = a_r_lim
    )

    return -a_t_lim, a_t_lim, ar_min, ar_max
end

function apply_boxed_control_update(
    u_nom::Vector{Float64},
    du::Vector{Float64},
    x::Float64,
    y::Float64,
    mem::BlockPassMemory;
    α_line::Float64,
    a_t_lim::Float64 = a_t_max,
    a_r_lim::Float64 = a_r_max
)
    at_min, at_max, ar_min, ar_max = local_control_box(
        x, y, mem;
        a_t_lim = a_t_lim,
        a_r_lim = a_r_lim
    )

    u_try_t = u_nom[1] + α_line * du[1]
    u_try_r = u_nom[2] + α_line * du[2]

    u_new_t = clamp(u_try_t, at_min, at_max)
    u_new_r = clamp(u_try_r, ar_min, ar_max)

    return [u_new_t, u_new_r]
end

# ============================================================
# STAGE 15 — FORWARD AFFINE POLICY APPLICATION
# ============================================================

function apply_stage15_affine_policy_rollout(
    scenario0::TwoAgentScenario,
    nominal_rollout::NominalTwoPlayerRollout,
    bp::Stage14BackwardPassResult,
    model_blue::CarModel,
    model_red::CarModel;
    α_line::Float64 = 0.35
)
    H = length(nominal_rollout.steps)

    scenario = deepcopy(scenario0)

    u_blue_new_seq = Vector{Vector{Float64}}(undef, H)
    u_red_new_seq  = Vector{Vector{Float64}}(undef, H)

    b_ref0, _ = joint_belief_from_stage11_step(nominal_rollout.steps[1], scenario0)
    b_cur = copy(b_ref0)

    prog_fwd = Progress(H, desc = "Stage 15 forward policy rollout")

    for k in 1:H
        step_nom = nominal_rollout.steps[k]
        bp_k = bp.steps[k]

        b_nom, _ = joint_belief_from_stage11_step(step_nom, scenario0)
        δb = b_cur - b_nom

        uB_nom = copy(step_nom.u_blue)
        uR_nom = copy(step_nom.u_red)

        δuB = bp_k.j_blue + bp_k.K_blue * δb
        δuR = bp_k.j_red  + bp_k.K_red  * δb

        uB_new = apply_boxed_control_update(
            uB_nom,
            δuB,
            scenario.blue_self_belief.mean[1],
            scenario.blue_self_belief.mean[2],
            scenario.blue_block_memory;
            α_line = α_line
        )

        uR_new = apply_boxed_control_update(
            uR_nom,
            δuR,
            scenario.red_self_belief.mean[1],
            scenario.red_self_belief.mean[2],
            scenario.red_block_memory;
            α_line = α_line
        )

        # keep the radial component from the memory-aware box
        # only saturate tangential acceleration here
        uB_new = [
            clamp(uB_new[1], -a_t_max, a_t_max),
            uB_new[2]
        ]

        uR_new = [
            clamp(uR_new[1], -a_t_max, a_t_max),
            uR_new[2]
        ]

        u_blue_new_seq[k] = uB_new
        u_red_new_seq[k]  = uR_new

        one_step_nominal_two_player!(scenario, uB_new, uR_new, model_blue, model_red)

        b_cur = pack_joint_belief_stage12(scenario)

        next!(prog_fwd; showvalues = [
            (:k, k),
            (:uB_t, round(uB_new[1], digits = 3)),
            (:uB_r, round(uB_new[2], digits = 3)),
            (:uR_t, round(uR_new[1], digits = 3)),
            (:uR_r, round(uR_new[2], digits = 3))
        ])
    end

    new_rollout = rollout_two_player_nominal_sequences(
        scenario0,
        u_blue_new_seq,
        u_red_new_seq,
        model_blue,
        model_red
    )

    return new_rollout
end

# ============================================================
# STAGE 15 — OUTER LOOP SOLVER
# ============================================================

function run_stage15_outer_loop(
    scenario0::TwoAgentScenario,
    model_blue::CarModel,
    model_red::CarModel;
    H::Int = H,
    N_outer::Int = 5,
    α_line::Float64 = 0.35
)
    u_blue_seq, u_red_seq = initialize_two_player_nominal_sequences(
        H;
        u_blue0 = [0.10, 0.0],
        u_red0  = [0.10, 0.0]
    )

    rollout = rollout_two_player_nominal_sequences(
        scenario0,
        u_blue_seq,
        u_red_seq,
        model_blue,
        model_red
    )

    iter_results = Stage15IterationResult[]

    prog_outer = Progress(N_outer, desc = "Stage 15 outer loop")

    for outer in 1:N_outer
        bp = run_stage14_true_backward_pass_from_rollout(
            rollout,
            scenario0,
            model_blue,
            model_red
        )

        push!(iter_results, Stage15IterationResult(
            rollout,
            bp,
            rollout.J_blue_total,
            rollout.J_red_total
        ))

        rollout = apply_stage15_affine_policy_rollout(
            scenario0,
            rollout,
            bp,
            model_blue,
            model_red;
            α_line = α_line
        )

        next!(prog_outer; showvalues = [
            (:iter, outer),
            (:J_blue, round(rollout.J_blue_total, digits = 4)),
            (:J_red, round(rollout.J_red_total, digits = 4))
        ])
    end

    final_bp = run_stage14_true_backward_pass_from_rollout(
        rollout,
        scenario0,
        model_blue,
        model_red
    )

    return Stage15OuterLoopResult(iter_results, rollout, final_bp)
end

# ============================================================
# STAGE 15 — PLOTS + RUNNER
# ============================================================

function plot_stage15_convergence(stage15::Stage15OuterLoopResult)
    nI = length(stage15.iterations)

    Jb = [it.J_blue_total for it in stage15.iterations]
    Jr = [it.J_red_total  for it in stage15.iterations]

    p = Plots.plot(1:nI, Jb;
        linewidth = 2.0,
        color = :blue,
        label = "blue total cost",
        xlabel = "outer iteration",
        ylabel = "cost",
        title = "Stage 15: Outer-Loop Cost Convergence"
    )

    Plots.plot!(p, 1:nI, Jr;
        linewidth = 2.0,
        color = :red,
        label = "red total cost"
    )

    return p
end

function plot_stage15_before_after(stage15::Stage15OuterLoopResult)
    p = plot_track()

    first_rollout = stage15.iterations[1].rollout
    final_rollout = stage15.final_rollout

    Plots.plot!(p,
        [st.blue_self_mean[1] for st in first_rollout.steps],
        [st.blue_self_mean[2] for st in first_rollout.steps];
        color = :blue,
        linestyle = :dash,
        linewidth = 2.0,
        label = "blue initial nominal"
    )

    Plots.plot!(p,
        [st.red_self_mean[1] for st in first_rollout.steps],
        [st.red_self_mean[2] for st in first_rollout.steps];
        color = :red,
        linestyle = :dash,
        linewidth = 2.0,
        label = "red initial nominal"
    )

    Plots.plot!(p,
        [st.blue_self_mean[1] for st in final_rollout.steps],
        [st.blue_self_mean[2] for st in final_rollout.steps];
        color = :blue,
        linewidth = 2.5,
        label = "blue final rollout"
    )

    Plots.plot!(p,
        [st.red_self_mean[1] for st in final_rollout.steps],
        [st.red_self_mean[2] for st in final_rollout.steps];
        color = :red,
        linewidth = 2.5,
        label = "red final rollout"
    )

    apply_paperstyle_axes!(p; title_str = "Stage 15: Before/After Outer-Loop Solver")
    return p
end

function plot_stage15_control_sequences(stage15::Stage15OuterLoopResult)
    first_rollout = stage15.iterations[1].rollout
    final_rollout = stage15.final_rollout
    H = length(final_rollout.steps)

    blue_r_0 = [st.u_blue[2] for st in first_rollout.steps]
    blue_r_f = [st.u_blue[2] for st in final_rollout.steps]

    red_r_0  = [st.u_red[2] for st in first_rollout.steps]
    red_r_f  = [st.u_red[2] for st in final_rollout.steps]

    p = Plots.plot(1:H, blue_r_0;
        linewidth = 2.0,
        linestyle = :dash,
        color = :blue,
        label = "blue initial a_r",
        xlabel = "horizon step",
        ylabel = "radial control",
        title = "Stage 15: Before/After Control Sequences"
    )

    Plots.plot!(p, 1:H, blue_r_f; linewidth = 2.2, color = :blue, label = "blue final a_r")
    Plots.plot!(p, 1:H, red_r_0;  linewidth = 2.0, linestyle = :dash, color = :red, label = "red initial a_r")
    Plots.plot!(p, 1:H, red_r_f;  linewidth = 2.2, color = :red, label = "red final a_r")

    return p
end

function run_stage15_outer_loop_solver(; H::Int = H, N_outer::Int = 5, α_line::Float64 = 0.35)
    println("=== Stage 15: Forward Policy Update + Outer Loop ===")

    scenario0, model_blue, model_red = make_stage11_reference_problem()

    stage15 = run_stage15_outer_loop(
        scenario0,
        model_blue,
        model_red;
        H = H,
        N_outer = N_outer,
        α_line = α_line
    )

    p1 = plot_stage15_convergence(stage15)
    display(p1)

    p2 = plot_stage15_before_after(stage15)
    display(p2)

    p3 = plot_stage15_control_sequences(stage15)
    display(p3)

    println("Initial blue cost = ", round(stage15.iterations[1].J_blue_total, digits = 4))
    println("Initial red cost  = ", round(stage15.iterations[1].J_red_total, digits = 4))
    println("Final blue cost   = ", round(stage15.final_rollout.J_blue_total, digits = 4))
    println("Final red cost    = ", round(stage15.final_rollout.J_red_total, digits = 4))
    println("=== Done ===")

    return stage15
end

function steps_until_passed_block(
    x::Float64,
    y::Float64,
    vx::Float64,
    vy::Float64,
    mem::BlockPassMemory;
    H_cap::Int = 64
)
    if !use_block_memory[]
        return 0
    end

    if !mem.active || mem.block_index == 0
        return 0
    end

    block = forbidden_blocks[mem.block_index]

    θ = atan(y, x)
    θ = θ < 0 ? θ + 2π : θ

    r = hypot(x, y)
    t_hat, _, _ = unit_vectors(x, y)
    v_t = max(vx * t_hat[1] + vy * t_hat[2], 0.0)

    θ_release = mod(block.theta_center + block.half_angle + angle_release_margin, 2π)
    dθ_release = mod(θ_release - θ, 2π)

    ω = v_t / max(r, 1e-6)

    if ω < 1e-8
        return H_cap
    end

    n_steps = ceil(Int, dθ_release / (ω * dt))
    return clamp(n_steps + extra_tail_steps, 0, H_cap)
end

# ============================================================
# STAGE 16 — FIRST-CONTROL EXTRACTION
# ============================================================

function choose_effective_horizon(
    scenario::TwoAgentScenario;
    H_base::Int = H_base,
    H_mid::Int = H_mid,
    H_far::Int = H_far,
    H_max::Int = H_max,
    dt::Float64 = dt,
    max_lookahead_angle::Float64 = 1.2
)
    # ------------------------------------------------------------
    # Local helper:
    # map angular distance to a basic horizon before memory logic
    # ------------------------------------------------------------
    function horizon_from_dtheta(dθ::Float64)
        if !isfinite(dθ)
            return H_base
        elseif dθ <= 0.25
            return H_max
        elseif dθ <= 0.50
            return H_far
        elseif dθ <= 0.80
            return H_mid
        else
            return H_base
        end
    end

    # ------------------------------------------------------------
    # BLUE
    # ------------------------------------------------------------
    blue_info = next_known_block_ahead(
        scenario.blue_true.x,
        scenario.blue_true.y;
        forbidden_blocks = forbidden_blocks,
        max_lookahead_angle = max_lookahead_angle
    )

    dθ_blue = blue_info.active ? blue_info.dθ_forward : Inf
    H_blue_adaptive = horizon_from_dtheta(dθ_blue)

    H_blue_mem = 0
    if use_block_memory[] && scenario.blue_block_memory.active
        H_blue_mem = steps_until_passed_block(
            scenario.blue_true.x,
            scenario.blue_true.y,
            scenario.blue_true.vx,
            scenario.blue_true.vy,
            scenario.blue_block_memory;
            H_cap = H_max
        )
    end

    H_blue = clamp(max(H_blue_adaptive, H_blue_mem), H_base, H_max)

    # ------------------------------------------------------------
    # RED
    # ------------------------------------------------------------
    red_info = next_known_block_ahead(
        scenario.red_true.x,
        scenario.red_true.y;
        forbidden_blocks = forbidden_blocks,
        max_lookahead_angle = max_lookahead_angle
    )

    dθ_red = red_info.active ? red_info.dθ_forward : Inf
    H_red_adaptive = horizon_from_dtheta(dθ_red)

    H_red_mem = 0
    if use_block_memory[] && scenario.red_block_memory.active
        H_red_mem = steps_until_passed_block(
            scenario.red_true.x,
            scenario.red_true.y,
            scenario.red_true.vx,
            scenario.red_true.vy,
            scenario.red_block_memory;
            H_cap = H_max
        )
    end

    H_red = clamp(max(H_red_adaptive, H_red_mem), H_base, H_max)

    # ------------------------------------------------------------
    # Final shared horizon for the two-player solve
    # ------------------------------------------------------------
    H_eff = max(H_blue, H_red)

    info = (
        dθ_blue   = dθ_blue,
        idx_blue  = blue_info.active ? blue_info.block_index : 0,
        dθ_red    = dθ_red,
        idx_red   = red_info.active ? red_info.block_index : 0,
        H_adaptive = max(H_blue_adaptive, H_red_adaptive),
        H_blue_mem = H_blue_mem,
        H_red_mem  = H_red_mem
    )

    return H_eff, info
end

function solve_stage16_first_controls(
    scenario::TwoAgentScenario,
    model_blue::CarModel,
    model_red::CarModel;
    H::Int = H,
    N_outer::Int = 5,
    α_line::Float64 = 0.35,
    use_adaptive_horizon::Bool = true,
    H_base::Int = H_base,
    H_mid::Int = H_mid,
    H_far::Int = H_far,
    H_max::Int = H_max,
    verbose_horizon::Bool = false
)
    planning_scenario = deepcopy(scenario)
    synchronize_first_order_beliefs!(planning_scenario)

    H_eff = H
    horizon_info = nothing

    if use_adaptive_horizon
        H_eff, horizon_info = choose_effective_horizon(
            planning_scenario;
            H_base = H_base,
            H_mid = H_mid,
            H_far = H_far,
            H_max = H_max,
            dt = dt
        )
    end

    stage15 = run_stage15_outer_loop(
        planning_scenario,
        model_blue,
        model_red;
        H = H_eff,
        N_outer = N_outer,
        α_line = α_line
    )

    if verbose_horizon && horizon_info !== nothing
        println(
            "Horizon choice | H_eff=", H_eff,
            " | dθ_blue=", isfinite(horizon_info.dθ_blue) ? round(horizon_info.dθ_blue, digits = 3) : Inf,
            " | next_blue_block=", horizon_info.idx_blue,
            " | dθ_red=", isfinite(horizon_info.dθ_red) ? round(horizon_info.dθ_red, digits = 3) : Inf,
            " | next_red_block=", horizon_info.idx_red,
            " | H_adaptive=", horizon_info.H_adaptive,
            " | H_blue_mem=", horizon_info.H_blue_mem,
            " | H_red_mem=", horizon_info.H_red_mem
        )
    end

    u_blue = copy(stage15.final_rollout.steps[1].u_blue)
    u_red  = copy(stage15.final_rollout.steps[1].u_red)

    return u_blue, u_red, stage15
end

function plot_stage16_two_agent_simulation(log::TwoAgentLog; ellipse_stride::Int = 10)
    p = plot_track()

    # true trajectories
    Plots.plot!(p, first.(log.blue_true), last.(log.blue_true);
        linewidth = 2.4, color = :blue, label = "blue true")

    Plots.plot!(p, first.(log.red_true), last.(log.red_true);
        linewidth = 2.4, color = :red, label = "red true")

    # self beliefs
    Plots.plot!(p, first.(log.blue_self_mean), last.(log.blue_self_mean);
        linestyle = :dash, linewidth = 2.0, color = :blue, label = "blue self belief")

    Plots.plot!(p, first.(log.red_self_mean), last.(log.red_self_mean);
        linestyle = :dash, linewidth = 2.0, color = :red, label = "red self belief")

    # cross beliefs
    Plots.plot!(p, first.(log.blue_on_red_mean), last.(log.blue_on_red_mean);
        linestyle = :dash, linewidth = 2.0, color = :purple, label = "blue belief on red")

    Plots.plot!(p, first.(log.red_on_blue_mean), last.(log.red_on_blue_mean);
        linestyle = :dash, linewidth = 2.0, color = :forestgreen, label = "red belief on blue")

    # optional visual connection between true agents
    add_agent_connections!(p, log.blue_true, log.red_true; stride = 5)

    # ellipses for all four beliefs
    for k in 1:ellipse_stride:length(log.blue_self_cov)
        xb, yb = log.blue_self_mean[k]
        draw_clipped_covariance_ellipse!(p,
            xb, yb, log.blue_self_cov[k];
            n_sigma = 2.0, color = :blue)

        xr, yr = log.red_self_mean[k]
        draw_clipped_covariance_ellipse!(p,
            xr, yr, log.red_self_cov[k];
            n_sigma = 2.0, color = :red)

        xbr, ybr = log.blue_on_red_mean[k]
        draw_clipped_covariance_ellipse!(p,
            xbr, ybr, log.blue_on_red_cov[k];
            n_sigma = 2.0, color = :purple)

        xrb, yrb = log.red_on_blue_mean[k]
        draw_clipped_covariance_ellipse!(p,
            xrb, yrb, log.red_on_blue_cov[k];
            n_sigma = 2.0, color = :forestgreen)
    end

    apply_paperstyle_axes!(p; title_str = "")

    # force larger legend font after all plotting/styling calls
    Plots.plot!(p;
        legendfontsize = 16
    )

    return p
end

function apply_scatter_publication_style!(
    p;
    title_str::String,
    xlabel_str::String,
    ylabel_str::String,
    legend_pos = :topright
)
    Plots.plot!(
        p;
        title = title_str,
        xlabel = xlabel_str,
        ylabel = ylabel_str,
        legend = legend_pos,
        size = (1200, 700),
        dpi = 300,
        framestyle = :box,
        grid = true,
        background_color = :white,
        foreground_color_axis = :black,
        foreground_color_border = :black,
        foreground_color_text = :black,
        xguidefontsize = 16,
        yguidefontsize = 16,
        xtickfontsize = 12,
        ytickfontsize = 12,
        legendfontsize = 12,
        titlefontsize = 20
    )
    return p
end

function scatter_with_colorbar_style(x, y, c; title_str="", xlabel_str="", ylabel_str="", cbar_title="")
    ρ = safe_corr(x, y)

    p = Plots.scatter(
        x, y;
        zcolor = c,
        markersize = 9,
        markerstrokewidth = 1.2,
        markerstrokecolor = :black,
        alpha = 0.75,
        colorbar = true,
        colorbar_title = cbar_title,
        legend = false
    )

    apply_scatter_publication_style!(
        p;
        title_str = "$(title_str)  (ρ = $(round(ρ, digits=3)))",
        xlabel_str = xlabel_str,
        ylabel_str = ylabel_str,
        legend_pos = :none
    )

    return p
end

function corr_text(x::Vector{Float64}, y::Vector{Float64})
    ρ = safe_corr(x, y)
    return isnan(ρ) ? "NaN" : string(round(ρ, digits=3))
end

function plot_stage16_collision_vs_distance(log::TwoAgentLog)
    p = Plots.plot(log.t, log.distance_true;
        linewidth = 2.2,
        color = :blue,
        label = "true distance",
        xlabel = "time [s]",
        ylabel = "value",
        title = "Stage 16: Distance vs Collision Risk")

    Plots.plot!(p, log.t, log.collision_cost_blue;
        linewidth = 2.0,
        linestyle = :dash,
        color = :orangered3,
        label = "blue collision cost")

    Plots.plot!(p, log.t, log.collision_cost_red;
        linewidth = 2.0,
        linestyle = :dash,
        color = :forestgreen,
        label = "red collision cost")

    Plots.plot!(p;
        legend = :topleft,
        size = (900, 500),
        dpi = 300,
        framestyle = :box,
        grid = true,
        background_color = :white
    )

    return p
end

function plot_stage16_controls(log::TwoAgentLog)
    blue_at = [u[1] for u in log.u_blue_hist]
    blue_ar = [u[2] for u in log.u_blue_hist]
    red_at  = [u[1] for u in log.u_red_hist]
    red_ar  = [u[2] for u in log.u_red_hist]

    p = Plots.plot(log.t, blue_at;
        linewidth = 2.0,
        color = :blue,
        label = "blue a_t",
        xlabel = "time [s]",
        ylabel = "control",
        title = "Stage 16 Controls")

    Plots.plot!(p, log.t, blue_ar;
        linewidth = 2.0,
        linestyle = :dash,
        color = :orangered3,
        label = "blue a_r")

    Plots.plot!(p, log.t, red_at;
        linewidth = 2.0,
        color = :forestgreen,
        label = "red a_t")

    Plots.plot!(p, log.t, red_ar;
        linewidth = 2.0,
        linestyle = :dash,
        color = :purple,
        label = "red a_r")

    Plots.plot!(p;
        legend = :topright,
        size = (900, 500),
        dpi = 300,
        framestyle = :box,
        grid = true,
        background_color = :white
    )

    return p
end

function plot_stage16_covariance_traces(log::TwoAgentLog)
    p = Plots.plot(log.t, log.tr_blue_self;
        linewidth = 2.0,
        color = :blue,
        label = "tr blue self",
        xlabel = "time [s]",
        ylabel = "trace",
        title = "Stage 16 Covariance Traces")

    Plots.plot!(p, log.t, log.tr_red_self;
        linewidth = 2.0, color = :red, label = "tr red self")

    Plots.plot!(p, log.t, log.tr_blue_on_red;
        linewidth = 2.0, color = :forestgreen, label = "tr blue on red")

    Plots.plot!(p, log.t, log.tr_red_on_blue;
        linewidth = 2.0, color = :purple, label = "tr red on blue")

    Plots.plot!(p;
        legend = :topleft,
        size = (900, 500),
        dpi = 300,
        framestyle = :box,
        grid = true,
        background_color = :white
    )

    return p
end

function plot_stage16_truth_only(log::TwoAgentLog)
    p = plot_track()

    bx = first.(log.blue_true)
    by = last.(log.blue_true)
    rx = first.(log.red_true)
    ry = last.(log.red_true)

    Plots.plot!(p, bx, by;
        linewidth = 2.4, color = :blue, label = "blue true")

    Plots.plot!(p, rx, ry;
        linewidth = 2.4, color = :red, label = "red true")

    Plots.scatter!(p, bx[1:10:end], by[1:10:end];
        markersize = 3.5,
        markerstrokewidth = 1.0,
        markercolor = :white,
        markerstrokecolor = :blue,
        label = "blue samples")

    Plots.scatter!(p, rx[1:10:end], ry[1:10:end];
        markersize = 3.5,
        markerstrokewidth = 1.0,
        markercolor = :white,
        markerstrokecolor = :red,
        label = "red samples")

    add_agent_connections!(p, log.blue_true, log.red_true; stride = 5)

    apply_paperstyle_axes!(p; title_str = "Stage 16 Truth-Only Validation")
    return p
end

function plot_stage16_belief_errors(log::TwoAgentLog)
    n = length(log.t)

    e_blue_self = zeros(n)
    e_red_self = zeros(n)
    e_blue_on_red = zeros(n)
    e_red_on_blue = zeros(n)

    for k in 1:n
        xb, yb = log.blue_true[k]
        xr, yr = log.red_true[k]

        μbs = log.blue_self_mean[k]
        μrs = log.red_self_mean[k]
        μbr = log.blue_on_red_mean[k]
        μrb = log.red_on_blue_mean[k]

        e_blue_self[k] = hypot(μbs[1] - xb, μbs[2] - yb)
        e_red_self[k] = hypot(μrs[1] - xr, μrs[2] - yr)
        e_blue_on_red[k] = hypot(μbr[1] - xr, μbr[2] - yr)
        e_red_on_blue[k] = hypot(μrb[1] - xb, μrb[2] - yb)
    end

    p = Plots.plot(log.t, e_blue_self;
        linewidth = 2.0,
        color = :blue,
        label = "blue self error",
        xlabel = "time [s]",
        ylabel = "position error",
        title = "Stage 16 Belief Errors")

    Plots.plot!(p, log.t, e_red_self;
        linewidth = 2.0, color = :red, label = "red self error")

    Plots.plot!(p, log.t, e_blue_on_red;
        linewidth = 2.0, color = :forestgreen, label = "blue on red error")

    Plots.plot!(p, log.t, e_red_on_blue;
        linewidth = 2.0, color = :purple, label = "red on blue error")

    Plots.plot!(p;
        legend = :topright,
        size = (900, 500),
        dpi = 300,
        framestyle = :box,
        grid = true,
        background_color = :white
    )

    return p
end

function plot_stage16_vs_nominal(
    log::TwoAgentLog,
    nominal_rollout::NominalTwoPlayerRollout
)
    p = plot_track()

    # Stage 16 true trajectories
    Plots.plot!(p, first.(log.blue_true), last.(log.blue_true);
        linewidth = 2.6, color = :blue, label = "blue closed-loop true")

    Plots.plot!(p, first.(log.red_true), last.(log.red_true);
        linewidth = 2.6, color = :red, label = "red closed-loop true")

    # Stage 11 nominal self-belief trajectories
    Plots.plot!(p,
        [st.blue_self_mean[1] for st in nominal_rollout.steps],
        [st.blue_self_mean[2] for st in nominal_rollout.steps];
        linewidth = 2.0, linestyle = :dash, color = :blue, label = "blue nominal")

    Plots.plot!(p,
        [st.red_self_mean[1] for st in nominal_rollout.steps],
        [st.red_self_mean[2] for st in nominal_rollout.steps];
        linewidth = 2.0, linestyle = :dash, color = :red, label = "red nominal")

    apply_paperstyle_axes!(p; title_str = "Stage 16 vs Stage 11 Nominal")
    return p
end

function plot_stage16_visibility_flags(log::TwoAgentLog)
    t = log.t

    y_blue_saw_red  = [b ? 1.0 : 0.0 for b in log.blue_saw_red_hist]
    y_red_saw_blue  = [b ? 1.0 : 0.0 for b in log.red_saw_blue_hist]
    y_blue_self_meas = [b ? 1.0 : 0.0 for b in log.blue_self_meas_hist]
    y_red_self_meas  = [b ? 1.0 : 0.0 for b in log.red_self_meas_hist]

    p = Plots.plot(t, y_blue_saw_red;
        linewidth = 2.0,
        label = "blue saw red",
        xlabel = "time [s]",
        ylabel = "flag",
        title = "Stage 16 Visibility / Measurement Flags")

    Plots.plot!(p, t, y_red_saw_blue;
        linewidth = 2.0,
        label = "red saw blue")

    Plots.plot!(p, t, y_blue_self_meas;
        linewidth = 2.0,
        linestyle = :dash,
        label = "blue self meas")

    Plots.plot!(p, t, y_red_self_meas;
        linewidth = 2.0,
        linestyle = :dash,
        label = "red self meas")

    Plots.plot!(p;
        legend = :topright,
        size = (900, 500),
        dpi = 300,
        framestyle = :box,
        grid = true,
        background_color = :white,
        ylims = (-0.1, 1.1)
    )

    return p
end

function plot_stage16_pca_rank(log::TwoAgentLog)
    p = Plots.plot(log.t, log.pca_upd_blue_self_rank;
        linewidth = 2.0,
        color = :blue,
        label = "blue self (upd)",
        xlabel = "time [s]",
        ylabel = "retained PCA rank",
        title = "Stage 16 PCA Rank (Update)"
    )

    Plots.plot!(p, log.t, log.pca_upd_red_self_rank;
        linewidth = 2.0,
        color = :red,
        label = "red self (upd)")

    Plots.plot!(p, log.t, log.pca_upd_blue_on_red_rank;
        linewidth = 2.0,
        linestyle = :dash,
        color = :forestgreen,
        label = "blue on red (upd)")

    Plots.plot!(p, log.t, log.pca_upd_red_on_blue_rank;
        linewidth = 2.0,
        linestyle = :dash,
        color = :purple,
        label = "red on blue (upd)")

    Plots.plot!(p;
        legend = :topright,
        size = (900, 500),
        dpi = 300,
        framestyle = :box,
        grid = true,
        background_color = :white
    )

    return p
end

function plot_stage16_pca_lambda_max(log::TwoAgentLog)
    p = Plots.plot(log.t, log.pca_upd_blue_self_lambda_max;
        linewidth = 2.0,
        color = :blue,
        label = "blue self λmax (upd)",
        xlabel = "time [s]",
        ylabel = "largest eigenvalue",
        title = "Stage 16 PCA λmax (Update)"
    )

    Plots.plot!(p, log.t, log.pca_upd_red_self_lambda_max;
        linewidth = 2.0,
        color = :red,
        label = "red self λmax (upd)")

    Plots.plot!(p, log.t, log.pca_upd_blue_on_red_lambda_max;
        linewidth = 2.0,
        linestyle = :dash,
        color = :forestgreen,
        label = "blue on red λmax (upd)")

    Plots.plot!(p, log.t, log.pca_upd_red_on_blue_lambda_max;
        linewidth = 2.0,
        linestyle = :dash,
        color = :purple,
        label = "red on blue λmax (upd)")

    Plots.plot!(p;
        legend = :topright,
        size = (900, 500),
        dpi = 300,
        framestyle = :box,
        grid = true,
        background_color = :white
    )

    return p
end

function plot_stage16_block_response(log::TwoAgentLog)
    t = log.t

    blue_ar = [u[2] for u in log.u_blue_hist]
    red_ar  = [u[2] for u in log.u_red_hist]

    p = Plots.plot(
        t, log.blue_block_clear_hist;
        linewidth = 2.4,
        color = :blue,
        label = "blue block clearance",
        xlabel = "time [s]",
        ylabel = "value",
        title = "Stage 16: Block Response (clearance, radial action, PCA)"
    )

    Plots.plot!(p, t, log.red_block_clear_hist;
        linewidth = 2.4,
        color = :red,
        label = "red block clearance")

    Plots.plot!(p, t, blue_ar;
        linewidth = 2.0,
        linestyle = :dash,
        color = :blue,
        alpha = 0.9,
        label = "blue a_r")

    Plots.plot!(p, t, red_ar;
        linewidth = 2.0,
        linestyle = :dash,
        color = :red,
        alpha = 0.9,
        label = "red a_r")

    Plots.plot!(p, t, log.pca_pred_blue_self_lambda_max;
        linewidth = 1.8,
        linestyle = :dot,
        color = :navy,
        label = "blue PCA λmax pred")

    Plots.plot!(p, t, log.pca_pred_red_self_lambda_max;
        linewidth = 1.8,
        linestyle = :dot,
        color = :darkred,
        label = "red PCA λmax pred")

    Plots.hline!(p, [risky_clearance_threshold];
        linestyle = :dashdot,
        linewidth = 1.5,
        color = :black,
        label = "risky clearance threshold")

    Plots.plot!(p;
        legend = :topright,
        size = (1050, 560),
        dpi = 300,
        framestyle = :box,
        grid = true,
        background_color = :white
    )

    return p
end

function plot_stage16_pca_pred_vs_upd(log::TwoAgentLog)
    p1 = Plots.plot(log.t, log.pca_pred_blue_self_lambda_max;
        linewidth = 2.0, color = :blue, label = "blue self λmax pred",
        xlabel = "time [s]", ylabel = "λmax",
        title = "Stage 16 PCA: Prediction vs Update"
    )

    Plots.plot!(p1, log.t, log.pca_upd_blue_self_lambda_max;
        linewidth = 2.0, linestyle = :dash, color = :blue, label = "blue self λmax upd")

    Plots.plot!(p1, log.t, log.pca_pred_red_self_lambda_max;
        linewidth = 2.0, color = :red, label = "red self λmax pred")

    Plots.plot!(p1, log.t, log.pca_upd_red_self_lambda_max;
        linewidth = 2.0, linestyle = :dash, color = :red, label = "red self λmax upd")

    Plots.plot!(p1, log.t, log.pca_pred_blue_on_red_lambda_max;
        linewidth = 1.8, color = :forestgreen, label = "blue on red λmax pred")

    Plots.plot!(p1, log.t, log.pca_upd_blue_on_red_lambda_max;
        linewidth = 1.8, linestyle = :dash, color = :forestgreen, label = "blue on red λmax upd")

    Plots.plot!(p1, log.t, log.pca_pred_red_on_blue_lambda_max;
        linewidth = 1.8, color = :purple, label = "red on blue λmax pred")

    Plots.plot!(p1, log.t, log.pca_upd_red_on_blue_lambda_max;
        linewidth = 1.8, linestyle = :dash, color = :purple, label = "red on blue λmax upd")

    Plots.plot!(p1;
        legend = :topright,
        size = (1100, 560),
        dpi = 300,
        framestyle = :box,
        grid = true,
        background_color = :white
    )

    return p1
end

function plot_stage16_pca_rank_pred_vs_upd(log::TwoAgentLog)
    p = Plots.plot(log.t, log.pca_pred_blue_self_rank;
        linewidth = 2.0, color = :blue, label = "blue self rank pred",
        xlabel = "time [s]", ylabel = "rank",
        title = "Stage 16 PCA Rank: Prediction vs Update"
    )

    Plots.plot!(p, log.t, log.pca_upd_blue_self_rank;
        linewidth = 2.0, linestyle = :dash, color = :blue, label = "blue self rank upd")

    Plots.plot!(p, log.t, log.pca_pred_red_self_rank;
        linewidth = 2.0, color = :red, label = "red self rank pred")

    Plots.plot!(p, log.t, log.pca_upd_red_self_rank;
        linewidth = 2.0, linestyle = :dash, color = :red, label = "red self rank upd")

    Plots.plot!(p, log.t, log.pca_pred_blue_on_red_rank;
        linewidth = 1.8, color = :forestgreen, label = "blue on red rank pred")

    Plots.plot!(p, log.t, log.pca_upd_blue_on_red_rank;
        linewidth = 1.8, linestyle = :dash, color = :forestgreen, label = "blue on red rank upd")

    Plots.plot!(p, log.t, log.pca_pred_red_on_blue_rank;
        linewidth = 1.8, color = :purple, label = "red on blue rank pred")

    Plots.plot!(p, log.t, log.pca_upd_red_on_blue_rank;
        linewidth = 1.8, linestyle = :dash, color = :purple, label = "red on blue rank upd")

    Plots.plot!(p;
        legend = :topright,
        size = (1100, 560),
        dpi = 300,
        framestyle = :box,
        grid = true,
        background_color = :white
    )

    return p
end

function plot_future_risk_vs_pca(log::TwoAgentLog)
    idxB = [
        i for i in eachindex(log.t)
        if log.blue_critical_moment_hist[i] &&
           isfinite(log.future_min_clear_blue[i]) &&
           positive_finite_pca_value(log.pca_pred_blue_self_lambda_max[i])
    ]

    idxR = [
        i for i in eachindex(log.t)
        if log.red_critical_moment_hist[i] &&
           isfinite(log.future_min_clear_red[i]) &&
           positive_finite_pca_value(log.pca_pred_red_self_lambda_max[i])
    ]

    xB = [log.pca_pred_blue_self_lambda_max[i] for i in idxB]
    yB = [log.future_min_clear_blue[i] for i in idxB]

    xR = [log.pca_pred_red_self_lambda_max[i] for i in idxR]
    yR = [log.future_min_clear_red[i] for i in idxR]

    cB = Float64[]
    for i in idxB
        xb, yb = log.blue_true[i]
        xr, yr = log.red_true[i]
        sb = project_to_track(centerline, [xb, yb])
        sr = project_to_track(centerline, [xr, yr])
        push!(cB, sr - sb)
    end

    cR = Float64[]
    for i in idxR
        xb, yb = log.blue_true[i]
        xr, yr = log.red_true[i]
        sb = project_to_track(centerline, [xb, yb])
        sr = project_to_track(centerline, [xr, yr])
        push!(cR, sb - sr)
    end

    x_all = vcat(xB, xR)
    y_all = vcat(yB, yR)
    ρtxt = corr_text(x_all, y_all)

    fig, ax = PyPlot.subplots(figsize=(9, 7))
    sc = nothing

    if !isempty(xB)
        sc = ax.scatter(
            xB, yB;
            c = cB,
            cmap = "coolwarm",
            edgecolors = "k",
            alpha = 0.7,
            s = 90
        )
    end

    if !isempty(xR)
        sc = ax.scatter(
            xR, yR;
            c = cR,
            cmap = "coolwarm",
            edgecolors = "k",
            alpha = 0.7,
            s = 90
        )
    end

    ax.set_xlabel("PCA λmax now (prediction)")
    ax.set_ylabel("Minimum clearance over next M steps")
    ax.set_title("Predictive Relevance: PCA vs Future Risk  (ρ = $ρtxt)")

    if sc !== nothing
        fig.colorbar(sc, ax=ax, label="Relative progress")
    end

    ax.grid(true)
    fig.tight_layout()
    fig.savefig("stage16_future_risk_vs_pca.pdf", dpi=300, bbox_inches="tight")

    return fig
end

function plot_decision_vs_pca_critical(log::TwoAgentLog)
    idxB = [
        i for i in eachindex(log.t)
        if log.blue_critical_moment_hist[i] &&
           positive_finite_pca_value(log.pca_pred_blue_self_lambda_max[i])
    ]

    idxR = [
        i for i in eachindex(log.t)
        if log.red_critical_moment_hist[i] &&
           positive_finite_pca_value(log.pca_pred_red_self_lambda_max[i])
    ]

    xB = [log.pca_pred_blue_self_lambda_max[i] for i in idxB]
    yB = [abs(log.u_blue_hist[i][2]) for i in idxB]

    xR = [log.pca_pred_red_self_lambda_max[i] for i in idxR]
    yR = [abs(log.u_red_hist[i][2]) for i in idxR]

    cB = Float64[]
    for i in idxB
        xb, yb = log.blue_true[i]
        xr, yr = log.red_true[i]
        sb = project_to_track(centerline, [xb, yb])
        sr = project_to_track(centerline, [xr, yr])
        push!(cB, sr - sb)
    end

    cR = Float64[]
    for i in idxR
        xb, yb = log.blue_true[i]
        xr, yr = log.red_true[i]
        sb = project_to_track(centerline, [xb, yb])
        sr = project_to_track(centerline, [xr, yr])
        push!(cR, sb - sr)
    end

    x_all = vcat(xB, xR)
    y_all = vcat(yB, yR)
    ρtxt = corr_text(x_all, y_all)

    fig, ax = PyPlot.subplots(figsize=(9, 7))
    sc = nothing

    if !isempty(xB)
        sc = ax.scatter(
            xB, yB;
            c = cB,
            cmap = "coolwarm",
            edgecolors = "k",
            alpha = 0.7,
            s = 90
        )
    end

    if !isempty(xR)
        sc = ax.scatter(
            xR, yR;
            c = cR,
            cmap = "coolwarm",
            edgecolors = "k",
            alpha = 0.7,
            s = 90
        )
    end

    ax.set_xlabel("PCA λmax now (prediction)")
    ax.set_ylabel("|a_r| now")
    ax.set_title("Decision-Making vs PCA Magnitude at Critical Moments  (ρ = $ρtxt)")

    if sc !== nothing
        fig.colorbar(sc, ax=ax, label="Relative progress")
    end

    ax.grid(true)
    fig.tight_layout()
    fig.savefig("stage16_decision_vs_pca_critical.pdf", dpi=300, bbox_inches="tight")

    return fig
end

function plot_compare_block_response(log_base::TwoAgentLog, log_pca::TwoAgentLog)
    t_base = log_base.t
    t_pca  = log_pca.t

    blue_ar_base = [u[2] for u in log_base.u_blue_hist]
    red_ar_base  = [u[2] for u in log_base.u_red_hist]

    blue_ar_pca = [u[2] for u in log_pca.u_blue_hist]
    red_ar_pca  = [u[2] for u in log_pca.u_red_hist]

    p = Plots.plot(
        t_base, log_base.blue_block_clear_hist;
        linewidth = 2.2,
        linestyle = :dash,
        color = :blue,
        label = "blue clearance baseline",
        xlabel = "time [s]",
        ylabel = "value",
        title = "Baseline vs PCA: Block Response"
    )

    Plots.plot!(p, t_pca, log_pca.blue_block_clear_hist;
        linewidth = 2.6,
        color = :blue,
        label = "blue clearance PCA"
    )

    Plots.plot!(p, t_base, log_base.red_block_clear_hist;
        linewidth = 2.2,
        linestyle = :dash,
        color = :red,
        label = "red clearance baseline"
    )

    Plots.plot!(p, t_pca, log_pca.red_block_clear_hist;
        linewidth = 2.6,
        color = :red,
        label = "red clearance PCA"
    )

    Plots.plot!(p, t_base, blue_ar_base;
        linewidth = 1.8,
        linestyle = :dot,
        color = :blue,
        alpha = 0.9,
        label = "blue a_r baseline"
    )

    Plots.plot!(p, t_pca, blue_ar_pca;
        linewidth = 2.0,
        linestyle = :dashdot,
        color = :navy,
        alpha = 0.95,
        label = "blue a_r PCA"
    )

    Plots.plot!(p, t_base, red_ar_base;
        linewidth = 1.8,
        linestyle = :dot,
        color = :red,
        alpha = 0.9,
        label = "red a_r baseline"
    )

    Plots.plot!(p, t_pca, red_ar_pca;
        linewidth = 2.0,
        linestyle = :dashdot,
        color = :darkred,
        alpha = 0.95,
        label = "red a_r PCA"
    )

    Plots.hline!(p, [risky_clearance_threshold];
        linestyle = :dashdot,
        linewidth = 1.5,
        color = :black,
        label = "risky clearance threshold"
    )

    Plots.plot!(p;
        legend = :topright,
        size = (1150, 620),
        dpi = 300,
        framestyle = :box,
        grid = true,
        background_color = :white
    )

    return p
end

function plot_stage16_truth_with_lookahead(
    log::TwoAgentLog;
    H::Int,
    dt::Float64 = dt,
    every_seconds::Float64 = 0.5
)
    p = plot_track()

    bx = first.(log.blue_true)
    by = last.(log.blue_true)
    rx = first.(log.red_true)
    ry = last.(log.red_true)

    Plots.plot!(p, bx, by; linewidth = 2.4, color = :blue, label = "blue true")
    Plots.plot!(p, rx, ry; linewidth = 2.4, color = :red,  label = "red true")

    stride = max(1, round(Int, every_seconds / dt))

    drew_blue_sector = false
    drew_red_sector  = false

    for k in 1:stride:length(log.t)
        xb, yb = log.blue_true[k]
        xr, yr = log.red_true[k]

        ub = log.u_blue_hist[k]
        ur = log.u_red_hist[k]

        # reconstruct approximate current velocities from adjacent points if possible
        if k < length(log.t)
            xb2, yb2 = log.blue_true[k+1]
            xr2, yr2 = log.red_true[k+1]

            vxb = (xb2 - xb) / dt
            vyb = (yb2 - yb) / dt

            vxr = (xr2 - xr) / dt
            vyr = (yr2 - yr) / dt
        elseif k > 1
            xb1, yb1 = log.blue_true[k-1]
            xr1, yr1 = log.red_true[k-1]

            vxb = (xb - xb1) / dt
            vyb = (yb - yb1) / dt

            vxr = (xr - xr1) / dt
            vyr = (yr - yr1) / dt
        else
            vxb = 0.0; vyb = 0.0
            vxr = 0.0; vyr = 0.0
        end

        add_lookahead_sector!(
            p, xb, yb, vxb, vyb;
            H = H,
            dt = dt,
            color = :blue,
            alpha = 0.06,
            label = drew_blue_sector ? false : "blue lookahead area"
        )
        drew_blue_sector = true

        add_lookahead_sector!(
            p, xr, yr, vxr, vyr;
            H = H,
            dt = dt,
            color = :red,
            alpha = 0.06,
            label = drew_red_sector ? false : "red lookahead area"
        )
        drew_red_sector = true
    end

    apply_paperstyle_axes!(p; title_str = "Stage 16 Truth + Effective Lookahead Area")
    return p
end

function plot_single_lookahead_snapshot(log, k; H, dt)
    p = plot_track()

    xb, yb = log.blue_true[k]
    xr, yr = log.red_true[k]

    # velocities
    if k < length(log.t)
        xb2, yb2 = log.blue_true[k+1]
        xr2, yr2 = log.red_true[k+1]

        vxb = (xb2 - xb)/dt
        vyb = (yb2 - yb)/dt
        vxr = (xr2 - xr)/dt
        vyr = (yr2 - yr)/dt
    else
        vxb = 0.0; vyb = 0.0
        vxr = 0.0; vyr = 0.0
    end

    # trajectories (faded)
    Plots.plot!(p, first.(log.blue_true), last.(log.blue_true);
          color=:blue, alpha=0.2, label=false)
    Plots.plot!(p, first.(log.red_true), last.(log.red_true);
          color=:red, alpha=0.2, label=false)

    # current positions
    Plots.scatter!(p, [xb], [yb]; color=:blue, label="blue now")
    Plots.scatter!(p, [xr], [yr]; color=:red, label="red now")

    # lookahead
    add_lookahead_sector!(p, xb, yb, vxb, vyb; H=H, dt=dt, color=:blue, alpha=0.2)
    add_lookahead_sector!(p, xr, yr, vxr, vyr; H=H, dt=dt, color=:red, alpha=0.2)

    return p
end

function ensure_output_dir()
    if !isdir(DIAG_OUTPUT_DIR)
        mkdir(DIAG_OUTPUT_DIR)
    end
end

function write_table_csv(path::String, header::Vector{String}, rows::AbstractVector{<:AbstractVector})
    open(path, "w") do io
        println(io, join(header, ","))
        for row in rows
            vals = map(v -> begin
                if v isa Bool
                    v ? "1" : "0"
                elseif v isa AbstractFloat
                    isfinite(v) ? string(v) : ""
                else
                    string(v)
                end
            end, row)
            println(io, join(vals, ","))
        end
    end
end

function export_log_timeseries_csv(log::TwoAgentLog, tag::String)
    ensure_output_dir()

    header = [
        "t",

        "blue_true_x","blue_true_y",
        "red_true_x","red_true_y",

        "blue_true_vx","blue_true_vy",
        "red_true_vx","red_true_vy",

        "blue_self_x","blue_self_y",
        "red_self_x","red_self_y",

        "blue_on_red_x","blue_on_red_y",
        "red_on_blue_x","red_on_blue_y",

        "uB_t","uB_r","uR_t","uR_r",

        "tr_blue_self","tr_red_self","tr_blue_on_red","tr_red_on_blue",
        "distance_true",

        "blue_self_meas","red_self_meas","blue_saw_red","red_saw_blue",

        "err_blue_self_pos","err_red_self_pos","err_blue_on_red_pos","err_red_on_blue_pos",

        "blue_block_clear","red_block_clear",
        "blue_inner_wall_clear","red_inner_wall_clear",
        "blue_outer_wall_clear","red_outer_wall_clear",

        "blue_critical","red_critical",
        "blue_wrong_side","red_wrong_side",
        "blue_safe_side","red_safe_side",

        "pca_pred_blue_self_active","pca_pred_red_self_active",
        "pca_pred_blue_self_rank","pca_pred_red_self_rank",
        "pca_pred_blue_self_lambda","pca_pred_red_self_lambda",

        "pca_upd_blue_self_active","pca_upd_red_self_active",
        "pca_upd_blue_self_rank","pca_upd_red_self_rank",
        "pca_upd_blue_self_lambda","pca_upd_red_self_lambda",

        "future_min_clear_blue","future_min_clear_red",
        "blue_block_event_id","red_block_event_id",

        "nom_blue_self_x","nom_blue_self_y",
        "nom_red_self_x","nom_red_self_y",

        "blue_self_P11","blue_self_P12","blue_self_P22",
        "red_self_P11","red_self_P12","red_self_P22",
        "blue_on_red_P11","blue_on_red_P12","blue_on_red_P22",
        "red_on_blue_P11","red_on_blue_P12","red_on_blue_P22"
    ]

    rows = Vector{Vector}(undef, length(log.t))
    for i in eachindex(log.t)
        rows[i] = Any[
            log.t[i],

            log.blue_true_x[i], log.blue_true_y[i],
            log.red_true_x[i],  log.red_true_y[i],

            log.blue_true_vx[i], log.blue_true_vy[i],
            log.red_true_vx[i],  log.red_true_vy[i],

            log.blue_self_x[i], log.blue_self_y[i],
            log.red_self_x[i],  log.red_self_y[i],

            log.blue_on_red_x[i], log.blue_on_red_y[i],
            log.red_on_blue_x[i], log.red_on_blue_y[i],

            log.u_blue_hist[i][1], log.u_blue_hist[i][2],
            log.u_red_hist[i][1],  log.u_red_hist[i][2],

            log.tr_blue_self[i], log.tr_red_self[i],
            log.tr_blue_on_red[i], log.tr_red_on_blue[i],
            log.distance_true_hist[i],

            log.blue_self_meas_hist_num[i], log.red_self_meas_hist_num[i],
            log.blue_saw_red_hist_num[i],   log.red_saw_blue_hist_num[i],

            log.err_blue_self_pos[i], log.err_red_self_pos[i],
            log.err_blue_on_red_pos[i], log.err_red_on_blue_pos[i],

            log.blue_block_clear_hist[i], log.red_block_clear_hist[i],
            log.blue_inner_wall_clear_hist[i], log.red_inner_wall_clear_hist[i],
            log.blue_outer_wall_clear_hist[i], log.red_outer_wall_clear_hist[i],

            log.blue_critical_moment_hist[i], log.red_critical_moment_hist[i],
            log.blue_block_wrong_side_hist[i], log.red_block_wrong_side_hist[i],
            log.blue_block_safe_side_hist[i], log.red_block_safe_side_hist[i],

            log.pca_pred_blue_self_active[i], log.pca_pred_red_self_active[i],
            log.pca_pred_blue_self_rank[i], log.pca_pred_red_self_rank[i],
            log.pca_pred_blue_self_lambda_max[i], log.pca_pred_red_self_lambda_max[i],

            log.pca_upd_blue_self_active[i], log.pca_upd_red_self_active[i],
            log.pca_upd_blue_self_rank[i], log.pca_upd_red_self_rank[i],
            log.pca_upd_blue_self_lambda_max[i], log.pca_upd_red_self_lambda_max[i],

            log.future_min_clear_blue[i], log.future_min_clear_red[i],
            log.blue_block_event_id[i], log.red_block_event_id[i],

            log.nom_blue_self_x[i], log.nom_blue_self_y[i],
            log.nom_red_self_x[i],  log.nom_red_self_y[i],

            log.blue_self_P11[i], log.blue_self_P12[i], log.blue_self_P22[i],
            log.red_self_P11[i],  log.red_self_P12[i],  log.red_self_P22[i],
            log.blue_on_red_P11[i], log.blue_on_red_P12[i], log.blue_on_red_P22[i],
            log.red_on_blue_P11[i], log.red_on_blue_P12[i], log.red_on_blue_P22[i]
        ]
    end

    write_table_csv(joinpath(DIAG_OUTPUT_DIR, "timeseries_" * tag * ".csv"), header, rows)
end

function export_summary_csv(
    tag::String;
    blue_anticipation,
    red_anticipation,
    blue_predictive,
    red_predictive,
    peakB::Float64,
    peakR::Float64
)
    ensure_output_dir()

    header = ["metric", "value"]
    rows = [
        ["blue_block_anticipation_pct", blue_anticipation.score_pct],
        ["blue_block_events", blue_anticipation.n_events],
        ["blue_block_anticipated", blue_anticipation.n_anticipated],

        ["red_block_anticipation_pct", red_anticipation.score_pct],
        ["red_block_events", red_anticipation.n_events],
        ["red_block_anticipated", red_anticipation.n_anticipated],

        ["blue_predictive_pearson_lambda_future_clear", blue_predictive.pearson_lambda_future_clear],
        ["blue_predictive_spearman_lambda_future_clear", blue_predictive.spearman_lambda_future_clear],
        ["blue_predictive_pearson_active_future_clear", blue_predictive.pearson_active_future_clear],
        ["blue_predictive_n_points", blue_predictive.n_points],

        ["red_predictive_pearson_lambda_future_clear", red_predictive.pearson_lambda_future_clear],
        ["red_predictive_spearman_lambda_future_clear", red_predictive.spearman_lambda_future_clear],
        ["red_predictive_pearson_active_future_clear", red_predictive.pearson_active_future_clear],
        ["red_predictive_n_points", red_predictive.n_points],

        ["blue_peak_abs_ar", peakB],
        ["red_peak_abs_ar", peakR]
    ]

    write_table_csv(joinpath(DIAG_OUTPUT_DIR, "summary_" * tag * ".csv"), header, rows)
end

function finalize_log_diagnostics!(log::TwoAgentLog)
    n = length(log.t)
    if n == 0
        return nothing
    end

    log.future_min_clear_blue .= compute_future_min_clearance_series(
        log.blue_block_clear_hist; M = future_risk_horizon_steps
    )
    log.future_min_clear_red .= compute_future_min_clearance_series(
        log.red_block_clear_hist; M = future_risk_horizon_steps
    )

    return nothing
end

# ============================================================
# STAGE 16 — CLOSED-LOOP RECEDING-HORIZON SIMULATION
# ============================================================

function run_stage16_all_estimator_modes(; kwargs...)
    ensure_output_dir()

    println("\n==============================")
    println("Running GUPTA mode")
    println("==============================\n")
    use_gupta_estimator!()
    log_gupta = run_stage16_closed_loop_game_simulation(; kwargs...)

    println("\n======================================")
    println("Running GUPTA + FULL PCA mode")
    println("======================================\n")
    use_gupta_pca_full_estimator!()
    log_gupta_pca_full = run_stage16_closed_loop_game_simulation(; kwargs...)

    println("\n==========================================")
    println("Running GUPTA + TRUNCATED PCA (95%) mode")
    println("==========================================\n")
    use_gupta_pca_trunc95_estimator!()
    log_gupta_pca_trunc95 = run_stage16_closed_loop_game_simulation(; kwargs...)

    println("\n==============================================")
    println("Running GUPTA + FULL PCA + EXACT PROJECTION")
    println("==============================================\n")
    use_gupta_pca_full_exact_projection_estimator!()
    log_gupta_pca_full_ep = run_stage16_closed_loop_game_simulation(; kwargs...)

    println("\n======================================================")
    println("Running GUPTA + TRUNCATED PCA (95%) + EXACT PROJECTION")
    println("======================================================\n")
    use_gupta_pca_trunc95_exact_projection_estimator!()
    log_gupta_pca_trunc95_ep = run_stage16_closed_loop_game_simulation(; kwargs...)

    return Dict(
        :gupta => log_gupta,
        :gupta_pca_full => log_gupta_pca_full,
        :gupta_pca_trunc95 => log_gupta_pca_trunc95,
        :gupta_pca_full_ep => log_gupta_pca_full_ep,
        :gupta_pca_trunc95_ep => log_gupta_pca_trunc95_ep
    )
end

function run_stage16_closed_loop_game_simulation(;
    N_sim::Int = 120,
    H::Int = H,
    N_outer::Int = 1,
    α_line::Float64 = 0.35,
    theta0_blue::Float64 = theta0_default,
    theta0_red::Float64 = theta0_default + 0.25,
    r0_blue::Float64 = r0_default,
    r0_red::Float64 = r0_default + 1.0,
    v_t0_blue::Float64 = v_t0_default,
    v_t0_red::Float64 = v_t0_default
)
    println("=== Stage 16: Closed-Loop Receding-Horizon Game Simulation ===")

    model_blue = CarModel(c_drag_default, c_slip_default)
    model_red  = CarModel(c_drag_default, c_slip_default)

    blue_true = make_agent_on_track(theta0_blue, r0_blue, v_t0_blue)
    red_true  = make_agent_on_track(theta0_red, r0_red, v_t0_red)

    scenario = create_two_agent_scenario(blue_true, red_true)

    u_blue_nom, u_red_nom = initialize_two_player_nominal_sequences(
        H;
        u_blue0 = [0.10, 0.0],
        u_red0  = [0.10, 0.0]
    )

    nominal_rollout = rollout_two_player_nominal_sequences(
        deepcopy(scenario),
        u_blue_nom,
        u_red_nom,
        model_blue,
        model_red
    )

    log = make_empty_two_agent_log()

    prog = Progress(N_sim; desc = "Stage 16 simulation", dt = 0.1)

    # ------------------------------------------------------------
    # event / smoothness bookkeeping
    # ------------------------------------------------------------
    blue_event_id = 0
    red_event_id = 0

    prev_blue_critical = false
    prev_red_critical = false

    prev_u_blue_r = 0.0
    prev_u_red_r = 0.0

    for k in 1:N_sim
        sim_step[] = k
        t = (k - 1) * dt

        last_pca_pred_blue_self[]   = empty_pca_diagnostics()
        last_pca_pred_red_self[]    = empty_pca_diagnostics()
        last_pca_pred_blue_on_red[] = empty_pca_diagnostics()
        last_pca_pred_red_on_blue[] = empty_pca_diagnostics()

        last_pca_upd_blue_self[]   = empty_pca_diagnostics()
        last_pca_upd_red_self[]    = empty_pca_diagnostics()
        last_pca_upd_blue_on_red[] = empty_pca_diagnostics()
        last_pca_upd_red_on_blue[] = empty_pca_diagnostics()

        # ----------------------------------------------------
        # 1) First-time block observation -> latch pass intent
        # ----------------------------------------------------
        maybe_activate_block_pass_memory!(
            scenario.blue_block_memory,
            scenario.blue_true.x,
            scenario.blue_true.y;
            forbidden_blocks = forbidden_blocks
        )

        maybe_activate_block_pass_memory!(
            scenario.red_block_memory,
            scenario.red_true.x,
            scenario.red_true.y;
            forbidden_blocks = forbidden_blocks
        )

        if !use_block_memory[]
            scenario.blue_block_memory.active = false
            scenario.blue_block_memory.block_index = 0
            scenario.blue_block_memory.pass_side = :inward

            scenario.red_block_memory.active = false
            scenario.red_block_memory.block_index = 0
            scenario.red_block_memory.pass_side = :inward
        end

        # ----------------------------------------------------
        # 2) Solve first receding-horizon controls
        # ----------------------------------------------------
        u_blue, u_red, _ = solve_stage16_first_controls(
            scenario,
            model_blue,
            model_red;
            H = H,
            N_outer = N_outer,
            α_line = α_line,
            use_adaptive_horizon = true,
            H_base = H_base,
            H_mid  = H_mid,
            H_far  = H_far,
            H_max  = H_max,
            verbose_horizon = (k % 20 == 0)
        )

        # ----------------------------------------------------
        # 3) Propagate true states and beliefs
        # ----------------------------------------------------
        propagate_two_agent_truth!(scenario, u_blue, u_red, model_blue, model_red)
        predict_two_agent_beliefs!(scenario, u_blue, u_red, model_blue, model_red)

        got_blue_self, got_red_self, got_blue_on_red, got_red_on_blue =
            update_two_agent_beliefs_stage10!(scenario)

        # ----------------------------------------------------
        # 4) Readable block / wall / critical diagnostics
        # ----------------------------------------------------
        blue_block_state = nearest_relevant_block_state(
            scenario.blue_true.x, scenario.blue_true.y;
            forbidden_blocks = forbidden_blocks
        )

        red_block_state = nearest_relevant_block_state(
            scenario.red_true.x, scenario.red_true.y;
            forbidden_blocks = forbidden_blocks
        )

        blue_wall = wall_clearances(scenario.blue_true.x, scenario.blue_true.y)
        red_wall  = wall_clearances(scenario.red_true.x, scenario.red_true.y)

        blue_critical_now = is_critical_constraint_moment(
            scenario.blue_true.x, scenario.blue_true.y;
            forbidden_blocks = forbidden_blocks
        )

        red_critical_now = is_critical_constraint_moment(
            scenario.red_true.x, scenario.red_true.y;
            forbidden_blocks = forbidden_blocks
        )

        blue_event_id = maybe_next_event_id(prev_blue_critical, blue_critical_now, blue_event_id)
        red_event_id  = maybe_next_event_id(prev_red_critical,  red_critical_now,  red_event_id)

        prev_blue_critical = blue_critical_now
        prev_red_critical  = red_critical_now

        # ----------------------------------------------------
        # 5) Estimation error diagnostics
        # ----------------------------------------------------
        err_blue_self = hypot(
            scenario.blue_self_belief.mean[1] - scenario.blue_true.x,
            scenario.blue_self_belief.mean[2] - scenario.blue_true.y
        )

        err_red_self = hypot(
            scenario.red_self_belief.mean[1] - scenario.red_true.x,
            scenario.red_self_belief.mean[2] - scenario.red_true.y
        )

        err_blue_on_red = hypot(
            scenario.blue_belief_on_red.mean[1] - scenario.red_true.x,
            scenario.blue_belief_on_red.mean[2] - scenario.red_true.y
        )

        err_red_on_blue = hypot(
            scenario.red_belief_on_blue.mean[1] - scenario.blue_true.x,
            scenario.red_belief_on_blue.mean[2] - scenario.blue_true.y
        )

        # ----------------------------------------------------
        # 6) Control smoothness diagnostics
        # ----------------------------------------------------
        du_blue_r = compute_du_diagnostics ? (u_blue[2] - prev_u_blue_r) : 0.0
        du_red_r  = compute_du_diagnostics ? (u_red[2]  - prev_u_red_r)  : 0.0

        prev_u_blue_r = u_blue[2]
        prev_u_red_r  = u_red[2]

        # ----------------------------------------------------
        # LOG (post-step, aligned)
        # ----------------------------------------------------
        push!(log.u_blue_hist, copy(u_blue))
        push!(log.u_red_hist,  copy(u_red))
        push!(log.t, t)

        push!(log.blue_true, (scenario.blue_true.x, scenario.blue_true.y))
        push!(log.red_true,  (scenario.red_true.x,  scenario.red_true.y))

        push!(log.blue_true_x, scenario.blue_true.x)
        push!(log.blue_true_y, scenario.blue_true.y)
        push!(log.red_true_x, scenario.red_true.x)
        push!(log.red_true_y, scenario.red_true.y)

        push!(log.blue_true_vx, scenario.blue_true.vx)
        push!(log.blue_true_vy, scenario.blue_true.vy)
        push!(log.red_true_vx, scenario.red_true.vx)
        push!(log.red_true_vy, scenario.red_true.vy)

        push!(log.blue_self_mean, (
            scenario.blue_self_belief.mean[1],
            scenario.blue_self_belief.mean[2]
        ))
        push!(log.red_self_mean, (
            scenario.red_self_belief.mean[1],
            scenario.red_self_belief.mean[2]
        ))

        push!(log.blue_on_red_mean, (
            scenario.blue_belief_on_red.mean[1],
            scenario.blue_belief_on_red.mean[2]
        ))
        push!(log.red_on_blue_mean, (
            scenario.red_belief_on_blue.mean[1],
            scenario.red_belief_on_blue.mean[2]
        ))

        push!(log.blue_self_x, scenario.blue_self_belief.mean[1])
        push!(log.blue_self_y, scenario.blue_self_belief.mean[2])
        push!(log.red_self_x, scenario.red_self_belief.mean[1])
        push!(log.red_self_y, scenario.red_self_belief.mean[2])

        push!(log.blue_on_red_x, scenario.blue_belief_on_red.mean[1])
        push!(log.blue_on_red_y, scenario.blue_belief_on_red.mean[2])
        push!(log.red_on_blue_x, scenario.red_belief_on_blue.mean[1])
        push!(log.red_on_blue_y, scenario.red_belief_on_blue.mean[2])

        push!(log.blue_self_cov, copy(scenario.blue_self_belief.cov[1:2, 1:2]))
        push!(log.red_self_cov,  copy(scenario.red_self_belief.cov[1:2, 1:2]))

        push!(log.blue_on_red_cov, copy(scenario.blue_belief_on_red.cov[1:2, 1:2]))
        push!(log.red_on_blue_cov, copy(scenario.red_belief_on_blue.cov[1:2, 1:2]))

        Pb  = scenario.blue_self_belief.cov[1:2, 1:2]
        Pr  = scenario.red_self_belief.cov[1:2, 1:2]
        Pbr = scenario.blue_belief_on_red.cov[1:2, 1:2]
        Prb = scenario.red_belief_on_blue.cov[1:2, 1:2]

        push!(log.blue_self_P11, Pb[1,1]);  push!(log.blue_self_P12, Pb[1,2]);  push!(log.blue_self_P22, Pb[2,2])
        push!(log.red_self_P11,  Pr[1,1]);  push!(log.red_self_P12,  Pr[1,2]);  push!(log.red_self_P22,  Pr[2,2])

        push!(log.blue_on_red_P11, Pbr[1,1]); push!(log.blue_on_red_P12, Pbr[1,2]); push!(log.blue_on_red_P22, Pbr[2,2])
        push!(log.red_on_blue_P11, Prb[1,1]); push!(log.red_on_blue_P12, Prb[1,2]); push!(log.red_on_blue_P22, Prb[2,2])

        push!(log.tr_blue_self, tr(scenario.blue_self_belief.cov[1:2, 1:2]))
        push!(log.tr_red_self,  tr(scenario.red_self_belief.cov[1:2, 1:2]))
        push!(log.tr_blue_on_red, tr(scenario.blue_belief_on_red.cov[1:2, 1:2]))
        push!(log.tr_red_on_blue, tr(scenario.red_belief_on_blue.cov[1:2, 1:2]))

        dtrue = hypot(
            scenario.blue_true.x - scenario.red_true.x,
            scenario.blue_true.y - scenario.red_true.y
        )
        push!(log.distance_true, dtrue)

        push!(log.distance_true_hist, dtrue)

        push!(log.collision_cost_blue,
            two_agent_collision_cost(
                scenario.blue_self_belief,
                scenario.blue_belief_on_red;
                w_c = w_coll_default,
                d_safe = d_safe_coll_default,
                σ_coll = σ_coll_default,
                k_unc = k_unc_coll_default
            )
        )

        push!(log.collision_cost_red,
            two_agent_collision_cost(
                scenario.red_self_belief,
                scenario.red_belief_on_blue;
                w_c = w_coll_default,
                d_safe = d_safe_coll_default,
                σ_coll = σ_coll_default,
                k_unc = k_unc_coll_default
            )
        )

        push!(log.blue_self_meas_hist, got_blue_self)
        push!(log.red_self_meas_hist, got_red_self)
        push!(log.blue_saw_red_hist, got_blue_on_red)
        push!(log.red_saw_blue_hist, got_red_on_blue)

        push!(log.blue_self_meas_hist_num, got_blue_self ? 1.0 : 0.0)
        push!(log.red_self_meas_hist_num, got_red_self ? 1.0 : 0.0)
        push!(log.blue_saw_red_hist_num, got_blue_on_red ? 1.0 : 0.0)
        push!(log.red_saw_blue_hist_num, got_red_on_blue ? 1.0 : 0.0)

        # ---------------------------
        # PCA diagnostics — prediction
        # ---------------------------
        push!(log.pca_pred_blue_self_active,   last_pca_pred_blue_self[].active)
        push!(log.pca_pred_red_self_active,    last_pca_pred_red_self[].active)
        push!(log.pca_pred_blue_on_red_active, last_pca_pred_blue_on_red[].active)
        push!(log.pca_pred_red_on_blue_active, last_pca_pred_red_on_blue[].active)

        push!(log.pca_pred_blue_self_rank,   last_pca_pred_blue_self[].retained_rank)
        push!(log.pca_pred_red_self_rank,    last_pca_pred_red_self[].retained_rank)
        push!(log.pca_pred_blue_on_red_rank, last_pca_pred_blue_on_red[].retained_rank)
        push!(log.pca_pred_red_on_blue_rank, last_pca_pred_red_on_blue[].retained_rank)

        push!(log.pca_pred_blue_self_lambda_max,
            isempty(last_pca_pred_blue_self[].eigvals) ? 0.0 : last_pca_pred_blue_self[].eigvals[1]
        )
        push!(log.pca_pred_red_self_lambda_max,
            isempty(last_pca_pred_red_self[].eigvals) ? 0.0 : last_pca_pred_red_self[].eigvals[1]
        )
        push!(log.pca_pred_blue_on_red_lambda_max,
            isempty(last_pca_pred_blue_on_red[].eigvals) ? 0.0 : last_pca_pred_blue_on_red[].eigvals[1]
        )
        push!(log.pca_pred_red_on_blue_lambda_max,
            isempty(last_pca_pred_red_on_blue[].eigvals) ? 0.0 : last_pca_pred_red_on_blue[].eigvals[1]
        )

        # ---------------------------
        # PCA diagnostics — update
        # ---------------------------
        push!(log.pca_upd_blue_self_active,   last_pca_upd_blue_self[].active)
        push!(log.pca_upd_red_self_active,    last_pca_upd_red_self[].active)
        push!(log.pca_upd_blue_on_red_active, last_pca_upd_blue_on_red[].active)
        push!(log.pca_upd_red_on_blue_active, last_pca_upd_red_on_blue[].active)

        push!(log.pca_upd_blue_self_rank,   last_pca_upd_blue_self[].retained_rank)
        push!(log.pca_upd_red_self_rank,    last_pca_upd_red_self[].retained_rank)
        push!(log.pca_upd_blue_on_red_rank, last_pca_upd_blue_on_red[].retained_rank)
        push!(log.pca_upd_red_on_blue_rank, last_pca_upd_red_on_blue[].retained_rank)

        push!(log.pca_upd_blue_self_lambda_max,
            isempty(last_pca_upd_blue_self[].eigvals) ? 0.0 : last_pca_upd_blue_self[].eigvals[1]
        )
        push!(log.pca_upd_red_self_lambda_max,
            isempty(last_pca_upd_red_self[].eigvals) ? 0.0 : last_pca_upd_red_self[].eigvals[1]
        )
        push!(log.pca_upd_blue_on_red_lambda_max,
            isempty(last_pca_upd_blue_on_red[].eigvals) ? 0.0 : last_pca_upd_blue_on_red[].eigvals[1]
        )
        push!(log.pca_upd_red_on_blue_lambda_max,
            isempty(last_pca_upd_red_on_blue[].eigvals) ? 0.0 : last_pca_upd_red_on_blue[].eigvals[1]
        )

        # ---------------------------
        # readable block / wall diagnostics
        # ---------------------------
        push!(log.blue_block_clear_hist, blue_block_state.clearance)
        push!(log.red_block_clear_hist,  red_block_state.clearance)

        push!(log.blue_inner_wall_clear_hist, blue_wall.inner_clear)
        push!(log.red_inner_wall_clear_hist,  red_wall.inner_clear)

        push!(log.blue_outer_wall_clear_hist, blue_wall.outer_clear)
        push!(log.red_outer_wall_clear_hist,  red_wall.outer_clear)

        push!(log.blue_block_wrong_side_hist, blue_block_state.wrong_side)
        push!(log.red_block_wrong_side_hist,  red_block_state.wrong_side)

        push!(log.blue_block_safe_side_hist, blue_block_state.safely_committed)
        push!(log.red_block_safe_side_hist,  red_block_state.safely_committed)

        push!(log.blue_block_near_sector_hist, blue_block_state.near_sector)
        push!(log.red_block_near_sector_hist,  red_block_state.near_sector)

        push!(log.blue_critical_moment_hist, blue_critical_now)
        push!(log.red_critical_moment_hist,  red_critical_now)

        # ---------------------------
        # estimation error diagnostics
        # ---------------------------
        push!(log.err_blue_self_pos, err_blue_self)
        push!(log.err_red_self_pos,  err_red_self)
        push!(log.err_blue_on_red_pos, err_blue_on_red)
        push!(log.err_red_on_blue_pos, err_red_on_blue)

        # ---------------------------
        # control smoothness diagnostics
        # ---------------------------
        push!(log.du_blue_r_hist, du_blue_r)
        push!(log.du_red_r_hist,  du_red_r)

        # ---------------------------
        # future-risk placeholders
        # ---------------------------
        push!(log.future_min_clear_blue, NaN)
        push!(log.future_min_clear_red,  NaN)

        # ---------------------------
        # event ids
        # ---------------------------
        push!(log.blue_block_event_id, blue_event_id)
        push!(log.red_block_event_id,  red_event_id)

        # ----------------------------------------------------
        # 7) Release commitment once block has been passed
        # ----------------------------------------------------
        maybe_release_block_pass_memory!(
            scenario.blue_block_memory,
            scenario.blue_true.x,
            scenario.blue_true.y;
            forbidden_blocks = forbidden_blocks
        )

        maybe_release_block_pass_memory!(
            scenario.red_block_memory,
            scenario.red_true.x,
            scenario.red_true.y;
            forbidden_blocks = forbidden_blocks
        )

        # ----------------------------------------------------
        # 8) Diagnostics printout
        # ----------------------------------------------------
        blue_clear_now = blue_block_state.clearance
        red_clear_now  = red_block_state.clearance

        e_blue_on_red = err_blue_on_red
        e_red_on_blue = err_red_on_blue

        if k % 10 == 0
            println(
                "k=", k,
                " | blue_clear=", isfinite(blue_clear_now) ? string(round(blue_clear_now, digits = 3)) : "Inf",
                " | red_clear=",  isfinite(red_clear_now)  ? string(round(red_clear_now,  digits = 3)) : "Inf",
                " | uB_r=", round(u_blue[2], digits = 3),
                " | uR_r=", round(u_red[2], digits = 3)
            )

            println(
                "k=", k,
                " | dtrue=", round(dtrue, digits = 3),
                " | e_BonR=", round(e_blue_on_red, digits = 3),
                " | e_RonB=", round(e_red_on_blue, digits = 3),
                " | B_saw_R=", got_blue_on_red,
                " | R_saw_B=", got_red_on_blue
            )

            println(
                "BLUE MEM | active=", scenario.blue_block_memory.active,
                " | block=", scenario.blue_block_memory.block_index,
                " | pass_side=", scenario.blue_block_memory.pass_side
            )

            println(
                "RED  MEM | active=", scenario.red_block_memory.active,
                " | block=", scenario.red_block_memory.block_index,
                " | pass_side=", scenario.red_block_memory.pass_side
            )
        end

        if k <= length(nominal_rollout.steps)
            push!(log.nom_blue_self_x, nominal_rollout.steps[k].blue_self_mean[1])
            push!(log.nom_blue_self_y, nominal_rollout.steps[k].blue_self_mean[2])
            push!(log.nom_red_self_x, nominal_rollout.steps[k].red_self_mean[1])
            push!(log.nom_red_self_y, nominal_rollout.steps[k].red_self_mean[2])
        else
            push!(log.nom_blue_self_x, NaN)
            push!(log.nom_blue_self_y, NaN)
            push!(log.nom_red_self_x, NaN)
            push!(log.nom_red_self_y, NaN)
        end

        next!(prog; showvalues = [
            (:step, k),
            (:dist, round(dtrue, digits = 3)),
            (:uB_r, round(u_blue[2], digits = 3)),
            (:uR_r, round(u_red[2], digits = 3))
        ])
    end

    # ------------------------------------------------------------
    # Finalize post-run diagnostics
    # ------------------------------------------------------------
    finalize_log_diagnostics!(log)

    blue_ant = block_anticipation_score(
        log.blue_block_clear_hist,
        log.pca_pred_blue_self_active;
        risk_threshold = risky_clearance_threshold,
        lookback_steps = anticipation_lookback_steps
    )

    red_ant = block_anticipation_score(
        log.red_block_clear_hist,
        log.pca_pred_red_self_active;
        risk_threshold = risky_clearance_threshold,
        lookback_steps = anticipation_lookback_steps
    )

    blue_predrel = predictive_relevance_summary(
        log.pca_pred_blue_self_lambda_max,
        log.pca_pred_blue_self_active,
        log.future_min_clear_blue,
        log.blue_critical_moment_hist
    )

    red_predrel = predictive_relevance_summary(
        log.pca_pred_red_self_lambda_max,
        log.pca_pred_red_self_active,
        log.future_min_clear_red,
        log.red_critical_moment_hist
    )

    peakvals = peak_radial_action(log)

    # ------------------------------------------------------------
    # Existing plots
    # ------------------------------------------------------------
    p1 = plot_stage16_two_agent_simulation(log; ellipse_stride = 10)
    display(p1)
    Plots.savefig(p1, "stage16_two_agent_simulation.pdf")

    p2 = plot_stage16_truth_only(log)
    display(p2)
    Plots.savefig(p2, "stage16_truth_only.pdf")

    p3 = plot_stage16_covariance_traces(log)
    display(p3)
    Plots.savefig(p3, "stage16_covariance_traces.pdf")

    p4 = plot_stage16_collision_vs_distance(log)
    display(p4)
    Plots.savefig(p4, "stage16_collision_vs_distance.pdf")

    p5 = plot_stage16_controls(log)
    display(p5)
    Plots.savefig(p5, "stage16_controls.pdf")

    p6 = plot_stage16_belief_errors(log)
    display(p6)
    Plots.savefig(p6, "stage16_belief_errors.pdf")

    p7 = plot_stage16_vs_nominal(log, nominal_rollout)
    display(p7)
    Plots.savefig(p7, "stage16_vs_nominal.pdf")

    p8 = plot_stage16_visibility_flags(log)
    display(p8)
    Plots.savefig(p8, "stage16_visibility_flags.pdf")

    p_pca_rank = plot_stage16_pca_rank(log)
    display(p_pca_rank)
    Plots.savefig(p_pca_rank, "stage16_pca_rank.pdf")

    p_pca_lambda = plot_stage16_pca_lambda_max(log)
    display(p_pca_lambda)
    Plots.savefig(p_pca_lambda, "stage16_pca_lambda_max.pdf")

    p9 = plot_stage16_block_response(log)
    display(p9)
    Plots.savefig(p9, "stage16_block_response.pdf")

    p10 = plot_stage16_truth_with_lookahead(log; H = H, dt = dt, every_seconds = 0.5)
    display(p10)
    Plots.savefig(p10, "stage16_truth_with_lookahead.pdf")

    # ------------------------------------------------------------
    # NEW plots
    # ------------------------------------------------------------
    p_pca_pred_upd_lambda = plot_stage16_pca_pred_vs_upd(log)
    display(p_pca_pred_upd_lambda)
    Plots.savefig(p_pca_pred_upd_lambda, "stage16_pca_pred_vs_upd_lambda.pdf")

    p_pca_pred_upd_rank = plot_stage16_pca_rank_pred_vs_upd(log)
    display(p_pca_pred_upd_rank)
    Plots.savefig(p_pca_pred_upd_rank, "stage16_pca_pred_vs_upd_rank.pdf")

    fig_future = plot_future_risk_vs_pca(log)
    display(fig_future)

    fig_decision = plot_decision_vs_pca_critical(log)
    display(fig_decision)


    # ------------------------------------------------------------
    # Summary prints
    # ------------------------------------------------------------
    println("Final true distance = ", round(log.distance_true[end], digits = 4))
    println("Final blue collision cost = ", round(log.collision_cost_blue[end], digits = 4))
    println("Final red collision cost  = ", round(log.collision_cost_red[end], digits = 4))
    println("Blue min block clearance = ", round(min_true_block_clearance(log.blue_true), digits = 4))
    println("Red min block clearance  = ", round(min_true_block_clearance(log.red_true), digits = 4))

    print_pca_summary(log)

    println("\n=== PCA cognition diagnostics ===")
    println("Blue block anticipation score [%] = ", round(blue_ant.score_pct, digits = 3),
            " | anticipated=", blue_ant.n_anticipated, "/", blue_ant.n_events)
    println("Red block anticipation score  [%] = ", round(red_ant.score_pct, digits = 3),
            " | anticipated=", red_ant.n_anticipated, "/", red_ant.n_events)

    println("Blue predictive relevance | Pearson(λmax, future clear) = ",
            round(blue_predrel.pearson_lambda_future_clear, digits = 4),
            " | Spearman = ", round(blue_predrel.spearman_lambda_future_clear, digits = 4),
            " | Pearson(active, future clear) = ", round(blue_predrel.pearson_active_future_clear, digits = 4),
            " | n = ", blue_predrel.n_points)

    println("Red predictive relevance  | Pearson(λmax, future clear) = ",
            round(red_predrel.pearson_lambda_future_clear, digits = 4),
            " | Spearman = ", round(red_predrel.spearman_lambda_future_clear, digits = 4),
            " | Pearson(active, future clear) = ", round(red_predrel.pearson_active_future_clear, digits = 4),
            " | n = ", red_predrel.n_points)

    println("Peak |a_r| blue = ", round(peakvals.blue, digits = 4))
    println("Peak |a_r| red  = ", round(peakvals.red, digits = 4))

    # ------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------
    if SAVE_DIAGNOSTIC_CSV
        export_log_timeseries_csv(log, string(estimator_mode[]))
        export_summary_csv(
            string(estimator_mode[]);
            blue_anticipation = blue_ant,
            red_anticipation = red_ant,
            blue_predictive = blue_predrel,
            red_predictive = red_predrel,
            peakB = peakvals.blue,
            peakR = peakvals.red
        )
    end

    println("=== Done ===")

    return log
end

function run_stage16_quick_diagnostic(;
    N_sim::Int = 120,
    H::Int = H,
    N_outer::Int = 1,
    α_line::Float64 = 0.35,
    theta0_blue::Float64 = theta0_default,
    theta0_red::Float64 = theta0_default + 0.25,
    r0_blue::Float64 = r0_default,
    r0_red::Float64 = r0_default + 1.0,
    v_t0_blue::Float64 = v_t0_default,
    v_t0_red::Float64 = v_t0_default
)
    return run_stage16_closed_loop_game_simulation(
        N_sim = N_sim,
        H = H,
        N_outer = N_outer,
        α_line = α_line,
        theta0_blue = theta0_blue,
        theta0_red = theta0_red,
        r0_blue = r0_blue,
        r0_red = r0_red,
        v_t0_blue = v_t0_blue,
        v_t0_red = v_t0_red
    )
end

function run_stage16_radius_case(
    rB::Float64,
    rR::Float64
)
    return run_stage16_closed_loop_game_simulation(
        N_sim = 1000,
        H = H,
        N_outer = 1,
        α_line = 0.35,
        theta0_blue = theta0_default,
        theta0_red  = theta0_default + 0.25,
        r0_blue = rB,
        r0_red  = rR,
        v_t0_blue = v_t0_default,
        v_t0_red  = v_t0_default
    )
end

# ============================================================
# EXECUTION
# ============================================================
if RUN_ALL_ESTIMATOR_MODES
    stage16_results = run_stage16_all_estimator_modes(
        N_sim = 300,
        H = H,
        N_outer = 2,
        α_line = 0.35,
        theta0_blue = theta0_default,
        theta0_red  = theta0_default + 0.15,
        r0_blue = r0_default - 2.5,
        r0_red  = r0_default - 2.0,
        v_t0_blue = v_t0_default,
        v_t0_red  = v_t0_default
    )

else
    # choose manually:
    #use_gupta_estimator!()
    #use_gupta_pca_full_estimator!()
    #use_gupta_pca_trunc95_estimator!()
    #use_gupta_pca_full_exact_projection_estimator!()
    #use_gupta_pca_trunc95_exact_projection_estimator!()


    stage16_D = run_stage16_closed_loop_game_simulation(
        N_sim = 300,
        H = H,
        N_outer = 2,
        α_line = 0.35,
        theta0_blue = theta0_default,
        theta0_red  = theta0_default + 0.15,
        r0_blue = r0_default - 2.5,
        r0_red  = r0_default - 2.0,
        v_t0_blue = v_t0_default,
        v_t0_red  = v_t0_default
    )
end