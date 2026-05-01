# plot_timelines_from_csv.jl
#
# Standalone post-processing script for timeline CSV files
# Supports multiple estimator modes.
#
# Recommended packages:
#   ] add CSV DataFrames Plots

using CSV
using DataFrames
using LinearAlgebra
import Plots

# ============================================================
# USER SETTINGS
# ============================================================

const OUTPUT_DIR = "timeseries_plots"

const CSV_MODES = Dict(
    :gupta                      => "Testcases_Final/testcaseA1_Final/timeseries_gupta.csv",
    :gupta_pca_full             => "Testcases_Final/testcaseA1_Final/timeseries_gupta_pca_full.csv",
    :gupta_pca_trunc95          => "Testcases_Final/testcaseA1_Final/timeseries_gupta_pca_trunc95.csv",
    :gupta_exact                => "Testcases_Final/testcaseA1_Final/timeseries_gupta_pca_full_ep.csv",
    :gupta_pca_trunc95_exact    => "Testcases_Final/testcaseA1_Final/timeseries_gupta_pca_trunc95_ep.csv"
)

# ------------------------------------------------------------
# Paper-quality plotting constants
# ------------------------------------------------------------
const PAPER_DPI = 600
const PAPER_SIZE_WIDE = (1500, 850)
const PAPER_SIZE_RACE = (1500, 950)
const PAPER_SIZE_HEATMAP = (1500, 800)
const PAPER_MARGIN = 4.0

# ------------------------------------------------------------
# Track / map constants
# These should match the simulation code
# ------------------------------------------------------------
const R_inner_plot = 30.0
const track_width_plot = 12.0
const R_outer_plot = R_inner_plot + track_width_plot
const light_width_plot = 3.0

const light_zones_plot = [
    (center = 0.0,   half_angle = 0.3, side = "inner"),
    (center = π/4,   half_angle = 0.3, side = "inner"),
    (center = 2π/3,  half_angle = 0.3, side = "outer"),
    (center = 6π/4,  half_angle = 0.3, side = "inner")
]

const forbidden_blocks_plot = NamedTuple[
    (theta_center = π/2.3, half_angle = 0.16, side = "inner", radial_depth = 3.0),
    (theta_center = π,     half_angle = 0.18, side = "inner", radial_depth = 6.0),
    (theta_center = 11π/6, half_angle = 0.14, side = "inner", radial_depth = 10.0)
]

# ============================================================
# BASIC UTILITIES
# ============================================================

function ensure_output_dir(dir::String = OUTPUT_DIR)
    isdir(dir) || mkpath(dir)
    return dir
end

function safe_filename(s::AbstractString)
    s2 = replace(lowercase(s), r"[^a-z0-9]+" => "_")
    s2 = replace(s2, r"_+" => "_")
    s2 = strip(s2, '_')
    return isempty(s2) ? "plot" : s2
end

function load_timeline_csv(path::String)
    df = CSV.read(path, DataFrame)
    rename!(df, Symbol.(names(df)))
    return df
end

function has_col(df::DataFrame, col::Symbol)
    return col in propertynames(df)
end

function first_existing(df::DataFrame, candidates::Vector{Symbol})
    for c in candidates
        if has_col(df, c)
            return c
        end
    end
    return nothing
end

function get_time_col(df::DataFrame)
    tcol = first_existing(df, [:t, :time, :time_s, :time_sec])
    tcol === nothing && error("Could not find a time column in CSV. Tried: :t, :time, :time_s, :time_sec")
    return tcol
end

function finite_min(v)
    vals = [x for x in v if x isa Number && isfinite(x)]
    isempty(vals) && return NaN
    return minimum(vals)
end

function finite_max(v)
    vals = [x for x in v if x isa Number && isfinite(x)]
    isempty(vals) && return NaN
    return maximum(vals)
end

function maybe_save_plot(p, filename::String; outdir::String = OUTPUT_DIR)
    ensure_output_dir(outdir)
    fullpath = joinpath(outdir, filename)
    Plots.savefig(p, fullpath)
    println("Saved: ", fullpath)
end

function unit_vectors(x::Float64, y::Float64)
    r = hypot(x, y)
    if r < 1e-12
        return [0.0, 1.0], [1.0, 0.0], 0.0
    end
    θ = atan(y, x)
    t_hat = [-sin(θ), cos(θ)]
    r_hat = [cos(θ), sin(θ)]
    return t_hat, r_hat, θ
end

function mode_display_name(mode::Symbol)
    if mode == :gupta
        return "Gupta"
    elseif mode == :gupta_pca_full
        return "Gupta + PCA full"
    elseif mode == :gupta_pca_trunc95
        return "Gupta + PCA trunc95"
    elseif mode == :gupta_exact
        return "Gupta + PCA full + exact projection"
    elseif mode == :gupta_pca_trunc95_exact
        return "Gupta + PCA trunc95 + exact projection"
    else
        return String(mode)
    end
end

# ============================================================
# STYLING
# ============================================================

function apply_style!(
    p;
    title_str::String = "",
    xlabel::String = "time [s]",
    ylabel::String = "",
    legend_pos = :topright,
    size_tuple = PAPER_SIZE_WIDE
)
    Plots.plot!(
        p;
        title = "",                 # titles intentionally removed; captions are in paper
        xlabel = xlabel,
        ylabel = ylabel,
        grid = true,
        framestyle = :box,
        dpi = PAPER_DPI,
        size = size_tuple,
        background_color = :white,
        legend = legend_pos,
        bottom_margin = 8Plots.mm,
        guidefontsize = 13,
        tickfontsize = 11,
        legendfontsize = 10,
        linewidth = 2.2
    )
    return p
end

function apply_race_style!(p; show_legend::Bool = false)
    Plots.plot!(
        p;
        title = "",
        xlabel = "",
        ylabel = "",
        grid = true,
        framestyle = :box,
        dpi = PAPER_DPI,
        size = PAPER_SIZE_RACE,
        background_color = :white,
        legend = show_legend ? :best : false,
        guidefontsize = 13,
        tickfontsize = 11,
        legendfontsize = 10
    )
    return p
end

function zoom_to_relevant_race_region!(
    p,
    df::DataFrame;
    margin::Float64 = PAPER_MARGIN
)
    cols = [
        :blue_true_x, :blue_true_y,
        :red_true_x,  :red_true_y,
        :blue_self_x, :blue_self_y,
        :red_self_x,  :red_self_y,
        :blue_on_red_x, :blue_on_red_y,
        :red_on_blue_x, :red_on_blue_y
    ]

    xs = Float64[]
    ys = Float64[]

    for i in 1:2:length(cols)
        cx = cols[i]
        cy = cols[i+1]
        if has_col(df, cx) && has_col(df, cy)
            append!(xs, collect(skipmissing(df[!, cx])))
            append!(ys, collect(skipmissing(df[!, cy])))
        end
    end

    xs = [x for x in xs if isfinite(x)]
    ys = [y for y in ys if isfinite(y)]

    if isempty(xs) || isempty(ys)
        return p
    end

    xmin = minimum(xs) - margin
    xmax = maximum(xs) + margin
    ymin = minimum(ys) - margin
    ymax = maximum(ys) + margin

    dx = xmax - xmin
    dy = ymax - ymin
    ratio = PAPER_SIZE_RACE[1] / PAPER_SIZE_RACE[2]

    if dx / max(dy, 1e-12) > ratio
        yc = 0.5 * (ymin + ymax)
        dy_new = dx / ratio
        ymin = yc - 0.5 * dy_new
        ymax = yc + 0.5 * dy_new
    else
        xc = 0.5 * (xmin + xmax)
        dx_new = dy * ratio
        xmin = xc - 0.5 * dx_new
        xmax = xc + 0.5 * dx_new
    end

    Plots.plot!(p; xlims = (xmin, xmax), ylims = (ymin, ymax))
    return p
end

# ============================================================
# TRACK / RACE PLOTTING HELPERS
# ============================================================

function plot_track_offline(; show_legend::Bool = false)
    θ = range(0, 2π, length = 700)

    inner_x = R_inner_plot .* cos.(θ)
    inner_y = R_inner_plot .* sin.(θ)

    outer_x = R_outer_plot .* cos.(θ)
    outer_y = R_outer_plot .* sin.(θ)

    p = Plots.plot(inner_x, inner_y;
        label = show_legend ? "Inner boundary" : false,
        linewidth = 2.2,
        color = :deepskyblue,
        aspect_ratio = 1
    )

    Plots.plot!(p, outer_x, outer_y;
        label = show_legend ? "Outer boundary" : false,
        linewidth = 2.2,
        color = :coral
    )

    drew_light_label = false
    for zone in light_zones_plot
        θs = range(zone.center - zone.half_angle,
                   zone.center + zone.half_angle,
                   length = 120)

        if zone.side == "inner"
            r1 = R_inner_plot
            r2 = R_inner_plot + light_width_plot
        else
            r1 = R_outer_plot - light_width_plot
            r2 = R_outer_plot
        end

        xs = vcat(r1 .* cos.(θs), reverse(r2 .* cos.(θs)))
        ys = vcat(r1 .* sin.(θs), reverse(r2 .* sin.(θs)))

        Plots.plot!(p, xs, ys;
            seriestype = :shape,
            color = :gold,
            fillalpha = 0.42,
            linealpha = 0.12,
            label = show_legend && !drew_light_label ? "Light zones" : false
        )
        drew_light_label = true
    end

    drew_block_label = false
    for block in forbidden_blocks_plot
        θs = range(block.theta_center - block.half_angle,
                   block.theta_center + block.half_angle,
                   length = 120)

        if block.side == "inner"
            r1 = R_inner_plot
            r2 = R_inner_plot + block.radial_depth
        else
            r1 = R_outer_plot - block.radial_depth
            r2 = R_outer_plot
        end

        xs = vcat(r1 .* cos.(θs), reverse(r2 .* cos.(θs)))
        ys = vcat(r1 .* sin.(θs), reverse(r2 .* sin.(θs)))

        Plots.plot!(p, xs, ys;
            seriestype = :shape,
            color = :gray60,
            fillalpha = 0.50,
            linealpha = 0.12,
            label = show_legend && !drew_block_label ? "Forbidden blocks" : false
        )
        drew_block_label = true
    end

    apply_race_style!(p; show_legend = show_legend)
    return p
end

function draw_cov_ellipse_from_entries!(
    p,
    x::Float64,
    y::Float64,
    P11::Float64,
    P12::Float64,
    P22::Float64;
    n_sigma::Float64 = 2.0,
    n_pts::Int = 220,
    color = :blue,
    alpha::Float64 = 0.45,
    linewidth::Float64 = 1.25
)
    P = [P11 P12; P12 P22]
    vals, vecs = eigen(Symmetric(P))
    vals = max.(vals, 1e-10)

    θ = range(0, 2π, length = n_pts)
    circle = [cos.(θ)'; sin.(θ)']
    shape = n_sigma .* vecs * Diagonal(sqrt.(vals)) * circle

    xe = x .+ shape[1, :]
    ye = y .+ shape[2, :]

    Plots.plot!(p, xe, ye;
        color = color,
        alpha = alpha,
        linewidth = linewidth,
        label = false
    )
    return p
end

function add_lookahead_sector_offline!(
    p,
    x::Float64, y::Float64,
    vx::Float64, vy::Float64;
    H::Int,
    dt::Float64,
    R_in::Float64 = R_inner_plot,
    R_out::Float64 = R_outer_plot,
    n_pts::Int = 100,
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

    Plots.plot!(p, xs, ys;
        seriestype = :shape,
        color = color,
        fillalpha = alpha,
        linealpha = 0.0,
        label = label
    )
    return p
end

# ============================================================
# GENERIC PLOTTING HELPERS
# ============================================================

function plot_single_series(df::DataFrame, ycol::Symbol; label_str::String = String(ycol))
    tcol = get_time_col(df)

    p = Plots.plot(
        df[!, tcol], df[!, ycol];
        linewidth = 2.2,
        label = label_str
    )

    apply_style!(p; title_str = "", ylabel = String(ycol))
    return p
end

function plot_multi_compare_series(
    dfs::Dict{Symbol,DataFrame},
    ycol::Symbol;
    title_str::String = String(ycol),
    ylabel::String = String(ycol)
)
    p = Plots.plot()
    found_any = false

    for mode in sort!(collect(keys(dfs)); by = x -> String(x))
        df = dfs[mode]
        if has_col(df, ycol)
            tcol = get_time_col(df)
            Plots.plot!(
                p,
                df[!, tcol],
                df[!, ycol];
                linewidth = 2.2,
                label = mode_display_name(mode)
            )
            found_any = true
        end
    end

    found_any || return nothing
    apply_style!(p; title_str = "", ylabel = ylabel)
    return p
end

function plot_multi_compare_two_pairs(
    dfs::Dict{Symbol,DataFrame},
    y1::Symbol,
    y2::Symbol;
    tag1::String = "blue",
    tag2::String = "red",
    title_str::String = "comparison",
    ylabel::String = "value"
)
    p = Plots.plot()
    found_any = false

    for mode in sort!(collect(keys(dfs)); by = x -> String(x))
        df = dfs[mode]
        if has_col(df, y1)
            tcol = get_time_col(df)
            Plots.plot!(
                p,
                df[!, tcol], df[!, y1];
                linewidth = 2.2,
                color = :blue,
                label = "$tag1 $(mode_display_name(mode))"
            )
            found_any = true
        end

        if has_col(df, y2)
            tcol = get_time_col(df)
            Plots.plot!(
                p,
                df[!, tcol], df[!, y2];
                linewidth = 2.2,
                linestyle = :dash,
                color = :red,
                label = "$tag2 $(mode_display_name(mode))"
            )
            found_any = true
        end
    end

    found_any || return nothing
    apply_style!(p; title_str = "", ylabel = ylabel)
    return p
end

# ============================================================
# CSV-BASED VERSIONS OF MAIN FIGURES
# ============================================================

function plot_stage16_two_agent_simulation_csv(df::DataFrame; ellipse_stride::Int = 10)
    needed = [
        :blue_true_x,:blue_true_y,:red_true_x,:red_true_y,
        :blue_self_x,:blue_self_y,:red_self_x,:red_self_y,
        :blue_on_red_x,:blue_on_red_y,:red_on_blue_x,:red_on_blue_y,
        :blue_self_P11,:blue_self_P12,:blue_self_P22,
        :red_self_P11,:red_self_P12,:red_self_P22,
        :blue_on_red_P11,:blue_on_red_P12,:blue_on_red_P22,
        :red_on_blue_P11,:red_on_blue_P12,:red_on_blue_P22
    ]
    all(c -> has_col(df,c), needed) || return nothing

    p = plot_track_offline(show_legend = false)

    # true trajectories
    Plots.plot!(p, df.blue_true_x, df.blue_true_y;
        linewidth = 2.8, color = :blue, label = false)

    Plots.plot!(p, df.red_true_x, df.red_true_y;
        linewidth = 2.8, color = :red, label = false)

    # dashed links between agents
    for k in 1:8:nrow(df)
        Plots.plot!(p,
            [df.blue_true_x[k], df.red_true_x[k]],
            [df.blue_true_y[k], df.red_true_y[k]];
            color = :black,
            linestyle = :dash,
            linewidth = 0.8,
            alpha = 0.45,
            label = false
        )
    end

    # belief trajectories
    Plots.plot!(p, df.blue_self_x, df.blue_self_y;
        linestyle = :dash, linewidth = 1.8, color = :blue, alpha = 0.9, label = false)

    Plots.plot!(p, df.red_self_x, df.red_self_y;
        linestyle = :dash, linewidth = 1.8, color = :red, alpha = 0.9, label = false)

    Plots.plot!(p, df.blue_on_red_x, df.blue_on_red_y;
        linestyle = :dash, linewidth = 1.8, color = :purple, alpha = 0.9, label = false)

    Plots.plot!(p, df.red_on_blue_x, df.red_on_blue_y;
        linestyle = :dash, linewidth = 1.8, color = :forestgreen, alpha = 0.9, label = false)

    # covariance ellipses
    for k in 1:ellipse_stride:nrow(df)
        draw_cov_ellipse_from_entries!(p,
            df.blue_self_x[k], df.blue_self_y[k],
            df.blue_self_P11[k], df.blue_self_P12[k], df.blue_self_P22[k];
            color = :blue)

        draw_cov_ellipse_from_entries!(p,
            df.red_self_x[k], df.red_self_y[k],
            df.red_self_P11[k], df.red_self_P12[k], df.red_self_P22[k];
            color = :red)

        draw_cov_ellipse_from_entries!(p,
            df.blue_on_red_x[k], df.blue_on_red_y[k],
            df.blue_on_red_P11[k], df.blue_on_red_P12[k], df.blue_on_red_P22[k];
            color = :purple)

        draw_cov_ellipse_from_entries!(p,
            df.red_on_blue_x[k], df.red_on_blue_y[k],
            df.red_on_blue_P11[k], df.red_on_blue_P12[k], df.red_on_blue_P22[k];
            color = :forestgreen)
    end

    zoom_to_relevant_race_region!(p, df)
    apply_race_style!(p; show_legend = false)
    return p
end


function plot_stage16_truth_only_csv(df::DataFrame)
    needed = [:blue_true_x,:blue_true_y,:red_true_x,:red_true_y]
    all(c -> has_col(df,c), needed) || return nothing

    p = plot_track_offline(show_legend = false)

    Plots.plot!(p, df.blue_true_x, df.blue_true_y;
        linewidth = 2.8, color = :blue, label = false)

    Plots.plot!(p, df.red_true_x, df.red_true_y;
        linewidth = 2.8, color = :red, label = false)

    Plots.scatter!(p,
        df.blue_true_x[1:10:end], df.blue_true_y[1:10:end];
        markersize = 3.2,
        markerstrokewidth = 0.8,
        markercolor = :white,
        markerstrokecolor = :blue,
        label = false)

    Plots.scatter!(p,
        df.red_true_x[1:10:end], df.red_true_y[1:10:end];
        markersize = 3.2,
        markerstrokewidth = 0.8,
        markercolor = :white,
        markerstrokecolor = :red,
        label = false)

    zoom_to_relevant_race_region!(p, df)
    apply_race_style!(p; show_legend = false)
    return p
end


function plot_stage16_covariance_traces_csv(df::DataFrame)
    needed = [:tr_blue_self,:tr_red_self,:tr_blue_on_red,:tr_red_on_blue]
    all(c -> has_col(df,c), needed) || return nothing

    tcol = get_time_col(df)

    p = Plots.plot(df[!,tcol], df.tr_blue_self;
        linewidth = 2.2, color = :blue, label = "blue self")

    Plots.plot!(p, df[!,tcol], df.tr_red_self;
        linewidth = 2.2, color = :red, label = "red self")

    Plots.plot!(p, df[!,tcol], df.tr_blue_on_red;
        linewidth = 2.2, color = :forestgreen, label = "blue on red")

    Plots.plot!(p, df[!,tcol], df.tr_red_on_blue;
        linewidth = 2.2, color = :purple, label = "red on blue")

    apply_style!(p; ylabel = "trace")
    return p
end


function plot_stage16_collision_vs_distance_csv(df::DataFrame)
    has_col(df,:distance_true) || return nothing
    tcol = get_time_col(df)

    p = Plots.plot(df[!,tcol], df.distance_true;
        linewidth = 2.3, color = :black, label = "distance")

    if has_col(df,:collision_cost_blue)
        Plots.plot!(p, df[!,tcol], df.collision_cost_blue;
            linewidth = 2.0, linestyle = :dash, color = :blue, label = "blue risk")
    end

    if has_col(df,:collision_cost_red)
        Plots.plot!(p, df[!,tcol], df.collision_cost_red;
            linewidth = 2.0, linestyle = :dash, color = :red, label = "red risk")
    end

    apply_style!(p; ylabel = "value")
    return p
end


function plot_stage16_controls_csv(df::DataFrame)
    needed = [:u_blue_t,:u_blue_r,:u_red_t,:u_red_r]
    all(c -> has_col(df,c), needed) || return nothing

    tcol = get_time_col(df)

    p = Plots.plot(df[!,tcol], df.u_blue_t;
        linewidth = 2.1, color = :blue, label = "blue a_t")

    Plots.plot!(p, df[!,tcol], df.u_blue_r;
        linewidth = 2.1, linestyle = :dash, color = :blue, label = "blue a_r")

    Plots.plot!(p, df[!,tcol], df.u_red_t;
        linewidth = 2.1, color = :red, label = "red a_t")

    Plots.plot!(p, df[!,tcol], df.u_red_r;
        linewidth = 2.1, linestyle = :dash, color = :red, label = "red a_r")

    apply_style!(p; ylabel = "control")
    return p
end


function plot_stage16_belief_errors_csv(df::DataFrame)
    needed = [:err_blue_self_pos,:err_red_self_pos,:err_blue_on_red_pos,:err_red_on_blue_pos]
    all(c -> has_col(df,c), needed) || return nothing

    tcol = get_time_col(df)

    p = Plots.plot(df[!,tcol], df.err_blue_self_pos;
        linewidth = 2.1, color = :blue, label = "blue self")

    Plots.plot!(p, df[!,tcol], df.err_red_self_pos;
        linewidth = 2.1, color = :red, label = "red self")

    Plots.plot!(p, df[!,tcol], df.err_blue_on_red_pos;
        linewidth = 2.1, color = :forestgreen, label = "blue on red")

    Plots.plot!(p, df[!,tcol], df.err_red_on_blue_pos;
        linewidth = 2.1, color = :purple, label = "red on blue")

    apply_style!(p; ylabel = "position error")
    return p
end


function plot_stage16_vs_nominal_csv(df::DataFrame)
    needed = [
        :blue_true_x,:blue_true_y,:red_true_x,:red_true_y,
        :nom_blue_self_x,:nom_blue_self_y,:nom_red_self_x,:nom_red_self_y
    ]
    all(c -> has_col(df,c), needed) || return nothing

    p = plot_track_offline(show_legend = false)

    Plots.plot!(p, df.blue_true_x, df.blue_true_y;
        linewidth = 2.8, color = :blue, label = false)

    Plots.plot!(p, df.red_true_x, df.red_true_y;
        linewidth = 2.8, color = :red, label = false)

    Plots.plot!(p, df.nom_blue_self_x, df.nom_blue_self_y;
        linewidth = 1.9, linestyle = :dash, color = :blue, alpha = 0.8, label = false)

    Plots.plot!(p, df.nom_red_self_x, df.nom_red_self_y;
        linewidth = 1.9, linestyle = :dash, color = :red, alpha = 0.8, label = false)

    zoom_to_relevant_race_region!(p, df)
    apply_race_style!(p; show_legend = false)
    return p
end


function plot_stage16_visibility_flags_csv(df::DataFrame)
    needed = [:blue_saw_red,:red_saw_blue,:blue_self_meas,:red_self_meas]
    all(c -> has_col(df,c), needed) || return nothing

    tcol = get_time_col(df)

    p = Plots.plot(df[!,tcol], df.blue_saw_red;
        linewidth = 2.0, label = "blue saw red")

    Plots.plot!(p, df[!,tcol], df.red_saw_blue;
        linewidth = 2.0, label = "red saw blue")

    Plots.plot!(p, df[!,tcol], df.blue_self_meas;
        linewidth = 2.0, linestyle = :dash, label = "blue self meas")

    Plots.plot!(p, df[!,tcol], df.red_self_meas;
        linewidth = 2.0, linestyle = :dash, label = "red self meas")

    Plots.plot!(p; ylims = (-0.05, 1.05))
    apply_style!(p; ylabel = "flag")
    return p
end


function plot_stage16_pca_rank_csv(df::DataFrame)
    needed = [:pca_pred_blue_self_rank_keep,:pca_pred_red_self_rank_keep,
              :pca_upd_blue_self_rank_keep,:pca_upd_red_self_rank_keep]
    all(c -> has_col(df,c), needed) || return nothing

    tcol = get_time_col(df)

    p = Plots.plot(df[!,tcol], df.pca_pred_blue_self_rank_keep;
        linewidth = 2.0, color = :blue, label = "blue pred")

    Plots.plot!(p, df[!,tcol], df.pca_pred_red_self_rank_keep;
        linewidth = 2.0, color = :red, label = "red pred")

    Plots.plot!(p, df[!,tcol], df.pca_upd_blue_self_rank_keep;
        linewidth = 2.0, linestyle = :dash, color = :blue, label = "blue upd")

    Plots.plot!(p, df[!,tcol], df.pca_upd_red_self_rank_keep;
        linewidth = 2.0, linestyle = :dash, color = :red, label = "red upd")

    apply_style!(p; ylabel = "rank")
    return p
end


function plot_stage16_pca_lambda_max_csv(df::DataFrame)
    needed = [:pca_pred_blue_self_lambda_max,:pca_pred_red_self_lambda_max,
              :pca_upd_blue_self_lambda_max,:pca_upd_red_self_lambda_max]
    all(c -> has_col(df,c), needed) || return nothing

    tcol = get_time_col(df)

    p = Plots.plot(df[!,tcol], df.pca_pred_blue_self_lambda_max;
        linewidth = 2.0, color = :blue, label = "blue pred")

    Plots.plot!(p, df[!,tcol], df.pca_pred_red_self_lambda_max;
        linewidth = 2.0, color = :red, label = "red pred")

    Plots.plot!(p, df[!,tcol], df.pca_upd_blue_self_lambda_max;
        linewidth = 2.0, linestyle = :dash, color = :blue, label = "blue upd")

    Plots.plot!(p, df[!,tcol], df.pca_upd_red_self_lambda_max;
        linewidth = 2.0, linestyle = :dash, color = :red, label = "red upd")

    apply_style!(p; ylabel = "largest eigenvalue")
    return p
end

function plot_stage16_block_response_csv(df::DataFrame)
    needed = [:u_blue_r,:u_red_r,:blue_block_clear_hist,:red_block_clear_hist]
    all(c -> has_col(df,c), needed) || return nothing

    tcol = get_time_col(df)

    p = Plots.plot(df[!,tcol], df.u_blue_r;
        linewidth = 2.1, color = :blue, label = "blue a_r")

    Plots.plot!(p, df[!,tcol], df.u_red_r;
        linewidth = 2.1, color = :red, label = "red a_r")

    Plots.plot!(p, df[!,tcol], df.blue_block_clear_hist;
        linewidth = 2.1, linestyle = :dash, color = :blue, label = "blue clearance")

    Plots.plot!(p, df[!,tcol], df.red_block_clear_hist;
        linewidth = 2.1, linestyle = :dash, color = :red, label = "red clearance")

    if has_col(df, :pca_pred_blue_self_lambda_max)
        Plots.plot!(p, df[!,tcol], df.pca_pred_blue_self_lambda_max;
            linewidth = 1.9, linestyle = :dot, color = :navy, label = "blue λmax")
    end

    if has_col(df, :pca_pred_red_self_lambda_max)
        Plots.plot!(p, df[!,tcol], df.pca_pred_red_self_lambda_max;
            linewidth = 1.9, linestyle = :dot, color = :darkred, label = "red λmax")
    end

    apply_style!(p; ylabel = "value")
    return p
end


function plot_stage16_truth_with_lookahead_csv(
    df::DataFrame;
    H::Int = 24,
    dt::Float64 = 0.02,
    every_seconds::Float64 = 0.5
)
    needed = [
        :blue_true_x,:blue_true_y,:red_true_x,:red_true_y,
        :blue_true_vx,:blue_true_vy,:red_true_vx,:red_true_vy
    ]
    all(c -> has_col(df,c), needed) || return nothing

    p = plot_track_offline(show_legend = false)

    Plots.plot!(p, df.blue_true_x, df.blue_true_y;
        linewidth = 2.8, color = :blue, label = false)

    Plots.plot!(p, df.red_true_x, df.red_true_y;
        linewidth = 2.8, color = :red, label = false)

    stride = max(1, round(Int, every_seconds / dt))

    for k in 1:stride:nrow(df)
        add_lookahead_sector_offline!(
            p,
            df.blue_true_x[k], df.blue_true_y[k],
            df.blue_true_vx[k], df.blue_true_vy[k];
            H = H,
            dt = dt,
            color = :blue,
            alpha = 0.06,
            label = false
        )

        add_lookahead_sector_offline!(
            p,
            df.red_true_x[k], df.red_true_y[k],
            df.red_true_vx[k], df.red_true_vy[k];
            H = H,
            dt = dt,
            color = :red,
            alpha = 0.06,
            label = false
        )
    end

    zoom_to_relevant_race_region!(p, df)
    apply_race_style!(p; show_legend = false)
    return p
end

# ============================================================
# MULTI-MODE COMPARISON PLOTS
# ============================================================

function maybe_plot_compare_block_clearance(dfs::Dict{Symbol,DataFrame})
    return plot_multi_compare_two_pairs(
        dfs,
        :blue_block_clear_hist,
        :red_block_clear_hist;
        tag1 = "blue",
        tag2 = "red",
        title_str = "",
        ylabel = "clearance"
    )
end

function maybe_plot_compare_radial_controls(dfs::Dict{Symbol,DataFrame})
    return plot_multi_compare_two_pairs(
        dfs,
        :u_blue_r,
        :u_red_r;
        tag1 = "blue a_r",
        tag2 = "red a_r",
        title_str = "",
        ylabel = "a_r"
    )
end

function maybe_plot_compare_self_errors(dfs::Dict{Symbol,DataFrame})
    return plot_multi_compare_two_pairs(
        dfs,
        :err_blue_self_pos,
        :err_red_self_pos;
        tag1 = "blue self",
        tag2 = "red self",
        title_str = "",
        ylabel = "position error"
    )
end

function maybe_plot_compare_cross_errors(dfs::Dict{Symbol,DataFrame})
    return plot_multi_compare_two_pairs(
        dfs,
        :err_blue_on_red_pos,
        :err_red_on_blue_pos;
        tag1 = "blue on red",
        tag2 = "red on blue",
        title_str = "",
        ylabel = "position error"
    )
end

function maybe_plot_compare_future_clearance(dfs::Dict{Symbol,DataFrame})
    return plot_multi_compare_two_pairs(
        dfs,
        :future_min_clear_blue,
        :future_min_clear_red;
        tag1 = "blue future clear",
        tag2 = "red future clear",
        title_str = "",
        ylabel = "future minimum clearance"
    )
end

function maybe_plot_compare_pca_lambda(dfs::Dict{Symbol,DataFrame})
    p = Plots.plot()
    found_any = false

    for mode in sort!(collect(keys(dfs)); by = x -> String(x))
        df = dfs[mode]

        if has_col(df, :pca_pred_blue_self_lambda_max)
            tcol = get_time_col(df)
            Plots.plot!(
                p,
                df[!,tcol],
                df.pca_pred_blue_self_lambda_max;
                linewidth = 2.2,
                label = "blue $(mode_display_name(mode))"
            )
            found_any = true
        end

        if has_col(df, :pca_pred_red_self_lambda_max)
            tcol = get_time_col(df)
            Plots.plot!(
                p,
                df[!,tcol],
                df.pca_pred_red_self_lambda_max;
                linewidth = 2.2,
                linestyle = :dash,
                label = "red $(mode_display_name(mode))"
            )
            found_any = true
        end
    end

    found_any || return nothing
    apply_style!(p; ylabel = "λmax")
    return p
end

function maybe_plot_compare_distance(dfs::Dict{Symbol,DataFrame})
    return plot_multi_compare_series(
        dfs,
        :distance_true;
        title_str = "",
        ylabel = "distance"
    )
end

function maybe_plot_compare_race(dfs::Dict{Symbol,DataFrame})
    p = plot_track_offline(show_legend = false)
    found_any = false

    xs = Float64[]
    ys = Float64[]

    for mode in sort!(collect(keys(dfs)); by = x -> String(x))
        df = dfs[mode]

        if all(c -> has_col(df,c), [:blue_true_x,:blue_true_y])
            Plots.plot!(p, df.blue_true_x, df.blue_true_y;
                color = :blue,
                linewidth = 2.2,
                alpha = 0.75,
                label = false)
            append!(xs, collect(skipmissing(df.blue_true_x)))
            append!(ys, collect(skipmissing(df.blue_true_y)))
            found_any = true
        end

        if all(c -> has_col(df,c), [:red_true_x,:red_true_y])
            Plots.plot!(p, df.red_true_x, df.red_true_y;
                color = :red,
                linestyle = :dash,
                linewidth = 2.2,
                alpha = 0.75,
                label = false)
            append!(xs, collect(skipmissing(df.red_true_x)))
            append!(ys, collect(skipmissing(df.red_true_y)))
            found_any = true
        end
    end

    found_any || return nothing

    xs = [x for x in xs if isfinite(x)]
    ys = [y for y in ys if isfinite(y)]

    if !isempty(xs) && !isempty(ys)
        xmin = minimum(xs) - PAPER_MARGIN
        xmax = maximum(xs) + PAPER_MARGIN
        ymin = minimum(ys) - PAPER_MARGIN
        ymax = maximum(ys) + PAPER_MARGIN

        dx = xmax - xmin
        dy = ymax - ymin
        ratio = PAPER_SIZE_RACE[1] / PAPER_SIZE_RACE[2]

        if dx / max(dy, 1e-12) > ratio
            yc = 0.5 * (ymin + ymax)
            dy_new = dx / ratio
            ymin = yc - 0.5 * dy_new
            ymax = yc + 0.5 * dy_new
        else
            xc = 0.5 * (xmin + xmax)
            dx_new = dy * ratio
            xmin = xc - 0.5 * dx_new
            xmax = xc + 0.5 * dx_new
        end

        Plots.plot!(p; xlims = (xmin, xmax), ylims = (ymin, ymax))
    end

    apply_race_style!(p; show_legend = false)
    return p
end

function maybe_plot_compare_pca_activity_map(dfs::Dict{Symbol,DataFrame})
    p = Plots.plot()
    ytick_vals = Float64[]
    ytick_labs = String[]
    row = 1.0

    for mode in sort!(collect(keys(dfs)); by = x -> String(x))
        df = dfs[mode]
        tcol = get_time_col(df)

        for (agent_label, col) in [
            ("blue", :pca_pred_blue_self_active),
            ("red",  :pca_pred_red_self_active)
        ]
            if has_col(df, col)
                active = Float64.(df[!, col])
                idx = findall(active .> 0.5)

                if !isempty(idx)
                    Plots.scatter!(
                        p,
                        df[!, tcol][idx],
                        fill(row, length(idx));
                        markersize = 5,
                        label = false
                    )
                end

                push!(ytick_vals, row)
                push!(ytick_labs, "$(agent_label) $(mode_display_name(mode))")
                row += 1.0
            end
        end
    end

    Plots.plot!(
        p;
        title = "",
        xlabel = "time [s]",
        ylabel = "mode / agent",
        yticks = (ytick_vals, ytick_labs),
        grid = true,
        framestyle = :box,
        dpi = PAPER_DPI,
        size = PAPER_SIZE_HEATMAP,
        legend = false,
        background_color = :white,
        guidefontsize = 13,
        tickfontsize = 10
    )

    return p
end

function maybe_plot_compare_pca_rank(dfs::Dict{Symbol,DataFrame})
    return plot_multi_compare_two_pairs(
        dfs,
        :pca_pred_blue_self_rank_keep,
        :pca_pred_red_self_rank_keep;
        tag1 = "blue rank",
        tag2 = "red rank",
        title_str = "",
        ylabel = "retained rank"
    )
end

function maybe_plot_compare_pca_rank_heatmap(
    dfs::Dict{Symbol,DataFrame};
    agent::Symbol = :blue
)
    rank_col = agent == :blue ? :pca_pred_blue_self_rank_keep : :pca_pred_red_self_rank_keep

    pca_modes = [
        :gupta_pca_full,
        :gupta_pca_trunc95,
        :gupta_exact,
        :gupta_pca_trunc95_exact
    ]

    labels = String[]
    t_ref = nothing
    Z = nothing

    for mode in pca_modes
        if haskey(dfs, mode) && has_col(dfs[mode], rank_col)
            df = dfs[mode]
            tcol = get_time_col(df)

            if t_ref === nothing
                t_ref = collect(df[!, tcol])
                Z = zeros(0, length(t_ref))
            end

            r = collect(Float64.(df[!, rank_col]))

            n = min(length(t_ref), length(r), size(Z, 2))
            t_ref = t_ref[1:n]
            Z = Z[:, 1:n]
            r = r[1:n]

            Z = vcat(Z, reshape(r, 1, :))
            push!(labels, mode_display_name(mode))
        end
    end

    if Z === nothing || size(Z, 1) == 0
        return nothing
    end

    maxrank = maximum(Z)

    p = Plots.heatmap(
        t_ref,
        1:size(Z, 1),
        Z;
        yticks = (1:size(Z, 1), labels),
        xlabel = "time [s]",
        ylabel = "PCA estimator mode",
        title = "",
        colorbar_title = "retained rank",
        clim = (0, max(1, maxrank)),
        size = PAPER_SIZE_HEATMAP,
        dpi = PAPER_DPI,
        framestyle = :box,
        background_color = :white,
        guidefontsize = 13,
        tickfontsize = 10,
        colorbar_tickfontsize = 10,
        colorbar_titlefontsize = 11,
        bottom_margin = 12Plots.mm,
        left_margin = 16Plots.mm,
        right_margin = 5Plots.mm,
        top_margin = 3Plots.mm
    )

    return p
end

function to_float_or_nan(v)
    return [ismissing(x) ? NaN : Float64(x) for x in v]
end

function maybe_plot_predictive_relevance_scatter(
    df::DataFrame;
    mode::Symbol
)
    needed = [
        :pca_pred_blue_self_lambda_max,
        :pca_pred_red_self_lambda_max,
        :future_min_clear_blue,
        :future_min_clear_red
    ]

    all(c -> has_col(df, c), needed) || return nothing

    xb = to_float_or_nan(df.pca_pred_blue_self_lambda_max)
    yb = to_float_or_nan(df.future_min_clear_blue)

    xr = to_float_or_nan(df.pca_pred_red_self_lambda_max)
    yr = to_float_or_nan(df.future_min_clear_red)

    t = has_col(df, :t) ? to_float_or_nan(df.t) : collect(1:nrow(df))

    idx_b = findall(isfinite.(xb) .& isfinite.(yb) .& isfinite.(t))
    idx_r = findall(isfinite.(xr) .& isfinite.(yr) .& isfinite.(t))

    isempty(idx_b) && isempty(idx_r) && return nothing

    p = Plots.plot(
        xlabel = "PCA λmax now",
        ylabel = "minimum clearance over future horizon",
        title = "",
        grid = true,
        framestyle = :box,
        size = PAPER_SIZE_WIDE,
        dpi = PAPER_DPI,
        background_color = :white,
        legend = :topright,
        guidefontsize = 13,
        tickfontsize = 11,
        legendfontsize = 10,
        bottom_margin = 12Plots.mm,
        left_margin = 12Plots.mm,
        right_margin = 12Plots.mm,
        top_margin = 3Plots.mm
    )

    if !isempty(idx_b)
        Plots.scatter!(
            p,
            xb[idx_b],
            yb[idx_b];
            marker_z = t[idx_b],
            markersize = 5.2,
            markerstrokewidth = 0.45,
            label = "blue",
            colorbar_title = "time [s]"
        )
    end

    if !isempty(idx_r)
        Plots.scatter!(
            p,
            xr[idx_r],
            yr[idx_r];
            marker_z = t[idx_r],
            markersize = 5.2,
            markerstrokewidth = 0.45,
            markerstrokecolor = :red,
            label = "red",
            colorbar_title = "time [s]"
        )
    end

    return p
end

function maybe_plot_predictive_rank_scatter(
    df::DataFrame;
    mode::Symbol,
    agent::Symbol = :blue
)
    rcol = agent == :blue ? :pca_pred_blue_self_rank_keep : :pca_pred_red_self_rank_keep
    fcol = agent == :blue ? :future_min_clear_blue : :future_min_clear_red

    has_col(df, rcol) && has_col(df, fcol) || return nothing

    x = to_float_or_nan(df[!, rcol])
    y = to_float_or_nan(df[!, fcol])

    idx = findall(isfinite.(x) .& isfinite.(y))
    isempty(idx) && return nothing

    p = Plots.scatter(
        x[idx], y[idx];
        markersize = 5.2,
        markerstrokewidth = 0.45,
        xlabel = "retained PCA rank",
        ylabel = "minimum clearance over future horizon",
        title = "",
        label = false,
        grid = true,
        framestyle = :box,
        size = PAPER_SIZE_WIDE,
        dpi = PAPER_DPI,
        background_color = :white,
        guidefontsize = 13,
        tickfontsize = 11
    )

    return p
end

function maybe_plot_compare_constraint_space_uncertainty(dfs::Dict{Symbol,DataFrame})
    p = Plots.plot()
    found_any = false

    for mode in sort!(collect(keys(dfs)); by = x -> String(x))
        df = dfs[mode]
        tcol = get_time_col(df)

        if has_col(df, :pca_pred_blue_self_lambda_max)
            Plots.plot!(
                p,
                df[!, tcol],
                df.pca_pred_blue_self_lambda_max;
                linewidth = 2.2,
                label = "blue $(mode_display_name(mode))"
            )
            found_any = true
        end

        if has_col(df, :pca_pred_red_self_lambda_max)
            Plots.plot!(
                p,
                df[!, tcol],
                df.pca_pred_red_self_lambda_max;
                linewidth = 2.2,
                linestyle = :dash,
                label = "red $(mode_display_name(mode))"
            )
            found_any = true
        end
    end

    found_any || return nothing

    apply_style!(
        p;
        title_str = "",
        ylabel = "PCA λmax"
    )

    return p
end

# ============================================================
# Optional comparison activity heatmap
# This function was referenced by the original driver but was
# not explicitly defined in the supplied script. It is kept here
# so the driver remains executable.
# ============================================================

function maybe_plot_pca_mode_activity_heatmap(dfs::Dict{Symbol,DataFrame})
    activity_cols = [
        :pca_pred_blue_self_active,
        :pca_pred_red_self_active,
        :pca_upd_blue_self_active,
        :pca_upd_red_self_active
    ]

    labels = String[]
    t_ref = nothing
    Z = nothing

    for mode in sort!(collect(keys(dfs)); by = x -> String(x))
        df = dfs[mode]
        tcol = get_time_col(df)

        for col in activity_cols
            if has_col(df, col)
                if t_ref === nothing
                    t_ref = collect(df[!, tcol])
                    Z = zeros(0, length(t_ref))
                end

                r = collect(Float64.(df[!, col]))

                n = min(length(t_ref), length(r), size(Z, 2))
                t_ref = t_ref[1:n]
                Z = Z[:, 1:n]
                r = r[1:n]

                Z = vcat(Z, reshape(r, 1, :))
                push!(labels, "$(mode_display_name(mode)) / $(String(col))")
            end
        end
    end

    if Z === nothing || size(Z, 1) == 0
        return nothing
    end

    p = Plots.heatmap(
        t_ref,
        1:size(Z, 1),
        Z;
        yticks = (1:size(Z, 1), labels),
        xlabel = "time [s]",
        ylabel = "activity channel",
        title = "",
        colorbar_title = "active",
        clim = (0, 1),
        size = PAPER_SIZE_HEATMAP,
        dpi = PAPER_DPI,
        framestyle = :box,
        background_color = :white,
        guidefontsize = 13,
        tickfontsize = 8,
        colorbar_tickfontsize = 10,
        colorbar_titlefontsize = 11,
        bottom_margin = 12Plots.mm,
        left_margin = 18Plots.mm,
        right_margin = 5Plots.mm,
        top_margin = 3Plots.mm
    )

    return p
end

# ============================================================
# COLUMN RENAMING ADAPTER
# ============================================================

function normalize_columns!(df::DataFrame)
    rename_map = Dict{Symbol,Symbol}()

    aliases = Dict(
        :time => :t,
        :time_s => :t,
        :time_sec => :t,

        :uB_t => :u_blue_t,
        :uB_r => :u_blue_r,
        :uR_t => :u_red_t,
        :uR_r => :u_red_r,

        :blue_block_clear => :blue_block_clear_hist,
        :red_block_clear  => :red_block_clear_hist,

        :pca_pred_blue_self_lambda => :pca_pred_blue_self_lambda_max,
        :pca_pred_red_self_lambda  => :pca_pred_red_self_lambda_max,
        :pca_pred_blue_self_rank   => :pca_pred_blue_self_rank_keep,
        :pca_pred_red_self_rank    => :pca_pred_red_self_rank_keep,

        :pca_upd_blue_self_lambda => :pca_upd_blue_self_lambda_max,
        :pca_upd_red_self_lambda  => :pca_upd_red_self_lambda_max,
        :pca_upd_blue_self_rank   => :pca_upd_blue_self_rank_keep,
        :pca_upd_red_self_rank    => :pca_upd_red_self_rank_keep
    )

    for (old, new) in aliases
        if has_col(df, old) && !has_col(df, new)
            rename_map[old] = new
        end
    end

    !isempty(rename_map) && rename!(df, rename_map)
    return df
end

# ============================================================
# MAIN DRIVER
# ============================================================

function run_postprocess()
    ensure_output_dir()

    println("Loading all estimator-mode CSV files...")
    dfs = Dict{Symbol,DataFrame}()

    for mode in sort!(collect(keys(CSV_MODES)); by = x -> String(x))
        path = CSV_MODES[mode]
        println("Loading ", mode, ": ", path)
        dfs[mode] = normalize_columns!(load_timeline_csv(path))
    end

    println("Generating per-mode figures...")

    for mode in sort!(collect(keys(dfs)); by = x -> String(x))
        df = dfs[mode]
        prefix = String(mode)

        figs = [
            ("$(prefix)_stage16_two_agent_simulation.pdf",
                plot_stage16_two_agent_simulation_csv(df)),

            ("$(prefix)_stage16_truth_only.pdf",
                plot_stage16_truth_only_csv(df)),

            ("$(prefix)_stage16_covariance_traces.pdf",
                plot_stage16_covariance_traces_csv(df)),

            ("$(prefix)_stage16_collision_vs_distance.pdf",
                plot_stage16_collision_vs_distance_csv(df)),

            ("$(prefix)_stage16_controls.pdf",
                plot_stage16_controls_csv(df)),

            ("$(prefix)_stage16_belief_errors.pdf",
                plot_stage16_belief_errors_csv(df)),

            ("$(prefix)_stage16_vs_nominal.pdf",
                plot_stage16_vs_nominal_csv(df)),

            ("$(prefix)_stage16_visibility_flags.pdf",
                plot_stage16_visibility_flags_csv(df)),

            ("$(prefix)_stage16_pca_rank.pdf",
                plot_stage16_pca_rank_csv(df)),

            ("$(prefix)_stage16_pca_lambda_max.pdf",
                plot_stage16_pca_lambda_max_csv(df)),

            ("$(prefix)_stage16_block_response.pdf",
                plot_stage16_block_response_csv(df)),

            ("$(prefix)_stage16_truth_with_lookahead.pdf",
                plot_stage16_truth_with_lookahead_csv(df)),

            ("compare_pca_rank_heatmap_blue.pdf",
                maybe_plot_compare_pca_rank_heatmap(dfs; agent = :blue)),

            ("compare_pca_rank_heatmap_red.pdf",
                maybe_plot_compare_pca_rank_heatmap(dfs; agent = :red))
        ]

        for (fname, fig) in figs
            fig === nothing || maybe_save_plot(fig, fname)
        end

        # --------------------------------------------
        # Predictive relevance scatter figures
        # --------------------------------------------
        p_scatter = maybe_plot_predictive_relevance_scatter(df; mode = mode)
        p_scatter === nothing || maybe_save_plot(
            p_scatter,
            "$(prefix)_predictive_relevance_lambda_scatter.pdf"
        )

        for agent in [:blue, :red]
            p_rank_scatter = maybe_plot_predictive_rank_scatter(
                df;
                mode = mode,
                agent = agent
            )

            p_rank_scatter === nothing || maybe_save_plot(
                p_rank_scatter,
                "$(prefix)_predictive_relevance_rank_$(String(agent)).pdf"
            )
        end

        # --------------------------------------------
        # Auto-generate numeric column time series
        # --------------------------------------------
        for c in names(df)
            csym = Symbol(c)

            if csym == get_time_col(df)
                continue
            end

            if eltype(df[!, csym]) <: Number
                p = plot_single_series(df, csym; label_str = String(csym))

                maybe_save_plot(
                    p,
                    "$(prefix)_series_" *
                    safe_filename(String(csym)) *
                    ".pdf"
                )
            end
        end
    end

    println("Generating multi-mode comparison figures...")

    comp_figs = [
        ("compare_block_clearance_all_modes.pdf",
            maybe_plot_compare_block_clearance(dfs)),

        ("compare_radial_controls_all_modes.pdf",
            maybe_plot_compare_radial_controls(dfs)),

        ("compare_self_errors_all_modes.pdf",
            maybe_plot_compare_self_errors(dfs)),

        ("compare_cross_errors_all_modes.pdf",
            maybe_plot_compare_cross_errors(dfs)),

        ("compare_future_min_clearance_all_modes.pdf",
            maybe_plot_compare_future_clearance(dfs)),

        ("compare_pca_prediction_lambda_all_modes.pdf",
            maybe_plot_compare_pca_lambda(dfs)),

        ("compare_pca_prediction_rank_all_modes.pdf",
            maybe_plot_compare_pca_rank(dfs)),

        ("compare_pca_activity_map_all_modes.pdf",
            maybe_plot_compare_pca_activity_map(dfs)),

        ("compare_true_distance_all_modes.pdf",
            maybe_plot_compare_distance(dfs)),

        ("compare_race_all_modes.pdf",
            maybe_plot_compare_race(dfs)),

        ("compare_pca_mode_activity_heatmap.pdf",
            maybe_plot_pca_mode_activity_heatmap(dfs)),

        ("compare_constraint_space_uncertainty_all_modes.pdf",
            maybe_plot_compare_constraint_space_uncertainty(dfs))
    ]

    for (fname, fig) in comp_figs
        fig === nothing || maybe_save_plot(fig, fname)
    end

    println("Done.")
    return nothing
end

# ============================================================
# EXECUTION
# ============================================================

run_postprocess()