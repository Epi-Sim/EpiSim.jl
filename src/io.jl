import MMCAcovid19
import MMCACovid19Vac

function update_config!(config, cmd_line_args)
    # Define dictionary containing epidemic parameters

    # overwrite config with command line
    if cmd_line_args["start-date"] !== nothing
        config["simulation"]["start_date"] = cmd_line_args["start-date"]
    end
    if cmd_line_args["end-date"] !== nothing
        config["simulation"]["end_date"] = cmd_line_args["end-date"]
    end
    if cmd_line_args["export-compartments-time-t"] !== nothing
        config["simulation"]["export_compartments_time_t"] = cmd_line_args["export-compartments-time-t"]
    end
    if cmd_line_args["export-compartments-full"] == true
        config["simulation"]["export_compartments_full"] = true
    end

    nothing
end

abstract type AbstractOutputFormat end

struct NetCDFFormat <: AbstractOutputFormat end
struct HDF5Format <: AbstractOutputFormat end

const OUTPUT_FORMATS = Dict("netcdf" => NetCDFFormat(), "hdf5" => HDF5Format())

get_output_format(output_format::String) = get(OUTPUT_FORMATS, output_format, NetCDFFormat())
get_output_format_str(output_format::AbstractOutputFormat) = findfirst(==(output_format), OUTPUT_FORMATS)

function save_full(epi_params, population, output_path::String, output_format::Union{String,AbstractOutputFormat}; kwargs...)
    format = output_format isa String ? get_output_format(output_format) : output_format
    _save_full(epi_params, population, output_path, format; kwargs...)
end

function _save_full(epi_params, population, output_path::String, ::NetCDFFormat; G_coords=String[], M_coords=String[], T_coords=String[])
    filename = joinpath(output_path, "compartments_full.nc")
    @info "Storing full simulation output in NetCDF: $filename"
    try
        save_simulation_netCDF(epi_params, population, filename; G_coords, M_coords, T_coords)
    catch e
        @error "Error saving simulation output" exception=(e, catch_backtrace())
        rethrow(e)
    end
end

function _save_full(epi_params, population, output_path::String, ::HDF5Format; kwargs...)
    filename = joinpath(output_path, "compartments_full.h5")
    @info "Storing full simulation output in HDF5: $filename"
    @warn "HDF5 format is deprecated"
    throw(ArgumentError("HDF5 format is deprecated"))
    save_simulation_hdf5(epi_params, population, filename)
end

function save_time_step(epi_params, population, output_path::String, export_compartments_time_t::Int) 
    export_compartments_date = first_day + Day(export_compartments_time_t - 1)
    filename = joinpath(output_path, "compartments_t_$(export_compartments_date).h5")
    @info "Storing compartments at single date $(export_compartments_date):"
    @info "\t- Simulation step: $(export_compartments_time_t)"
    @info "\t- filename: $(filename)"
    save_simulation_hdf5(epi_params, population, filename; 
                        export_time_t = export_compartments_time_t)
end


"""
save_simulation_netCDF(epi_params::MMCACovid19Vac.Epidemic_Params, 
                                    population::MMCACovid19Vac.Population_Params,
                                    output_fname::String;
                                    G_coords= nothing,
                                    M_coords = nothing,
                                    T_coords = nothing
                                    )

    Save the full simulations.

    # Arguments

    - `epi_params::MMCACovid19Vac.Epidemic_Params`: Structure that contains all epidemic parameters
    and the epidemic spreading information.
    - `population::MMCACovid19Vac.Population_Params`: Structure that contains all the parameters
    related with the population.
    - `output_fname::String`: Output filename.

    ## Optional
    - `G_coords = nothing`: Array::{String} of size G containing the labels for age strata
    - `M_coords = nothing`: Array::{String} of size M containing the labels for the patches
    - `T_coords = nothing`: Array::{String} of size t containing the labels for the time (dates)
    - `export_time_t = -1`: Time step to ve saved instead of the full simulation.
"""
function save_simulation_netCDF( epi_params::MMCACovid19Vac.Epidemic_Params, 
                                 population::MMCACovid19Vac.Population_Params,
                                 output_fname::String;
                                 G_coords = nothing,
                                 M_coords = nothing,
                                 T_coords = nothing,
                                 V_coords = nothing
                                )
    G = population.G
    M = population.M
    T = epi_params.T
    V = epi_params.V
    S = epi_params.NumComps
    S_coords = epi_params.CompLabels
    V_coords = epi_params.VaccLabels

    if isnothing(G_coords)
        G_coords = collect(1:G)
    end
    if isnothing(M_coords)
        M_coords = collect(1:M)
    end
    if isnothing(T_coords)
        T_coords = collect(1:T) 
    end

    compartments = zeros(Float64, G, M, T, V, S);

    # Adding 
    compartments[:, :, :, :, 1]  .= (epi_params.ρˢᵍᵥ + epi_params.CHᵢᵍᵥ) .* population.nᵢᵍ
    compartments[:, :, :, :, 2]  .= epi_params.ρᴱᵍᵥ .* population.nᵢᵍ
    compartments[:, :, :, :, 3]  .= epi_params.ρᴬᵍᵥ .* population.nᵢᵍ
    compartments[:, :, :, :, 4]  .= epi_params.ρᴵᵍᵥ .* population.nᵢᵍ
    compartments[:, :, :, :, 5]  .= epi_params.ρᴾᴴᵍᵥ .* population.nᵢᵍ
    compartments[:, :, :, :, 6]  .= epi_params.ρᴾᴰᵍᵥ .* population.nᵢᵍ
    compartments[:, :, :, :, 7]  .= epi_params.ρᴴᴿᵍᵥ .* population.nᵢᵍ
    compartments[:, :, :, :, 8]  .= epi_params.ρᴴᴰᵍᵥ .* population.nᵢᵍ
    compartments[:, :, :, :, 9]  .= epi_params.ρᴿᵍᵥ .* population.nᵢᵍ
    compartments[:, :, :, :, 10] .= epi_params.ρᴰᵍᵥ .* population.nᵢᵍ
    isfile(output_fname) && rm(output_fname)

    nccreate(output_fname, "data", "G", G_coords, "M", M_coords, "T", T_coords, "V", V_coords, "epi_states", S_coords)
    ncwrite(compartments, output_fname, "data")

end

function save_simulation_netCDF( epi_params::MMCAcovid19.Epidemic_Params, 
                                 population::MMCAcovid19.Population_Params,
                                 output_fname::String;
                                 G_coords = nothing,
                                 M_coords = nothing,
                                 T_coords = nothing,
                                )
    G = population.G
    M = population.M
    T = epi_params.T
    S = 10
    S_coords = ["S", "E", "A", "I", "PH", "PD", "HR", "HD", "R", "D"]

    if isnothing(G_coords)
        G_coords = collect(1:G)
    end
    if isnothing(M_coords)
        M_coords = collect(1:M)
    end
    if isnothing(T_coords)
        T_coords = collect(1:T) 
    end

    compartments = zeros(Float64, G, M, T, S);

    # Adding 
    compartments[:, :, :, 1]  .= epi_params.ρˢᵍ .* population.nᵢᵍ
    compartments[:, :, :, 2]  .= epi_params.ρᴱᵍ .* population.nᵢᵍ
    compartments[:, :, :, 3]  .= epi_params.ρᴬᵍ .* population.nᵢᵍ
    compartments[:, :, :, 4]  .= epi_params.ρᴵᵍ .* population.nᵢᵍ
    compartments[:, :, :, 5]  .= epi_params.ρᴾᴴᵍ .* population.nᵢᵍ
    compartments[:, :, :, 6]  .= epi_params.ρᴾᴰᵍ .* population.nᵢᵍ
    compartments[:, :, :, 7]  .= epi_params.ρᴴᴿᵍ .* population.nᵢᵍ
    compartments[:, :, :, 8]  .= epi_params.ρᴴᴰᵍ .* population.nᵢᵍ
    compartments[:, :, :, 9]  .= epi_params.ρᴿᵍ .* population.nᵢᵍ
    compartments[:, :, :, 10] .= epi_params.ρᴰᵍ .* population.nᵢᵍ
    isfile(output_fname) && rm(output_fname)

    @info "compartments shape: $(size(compartments))"

    nccreate(output_fname, "data", "G", G_coords, "M", M_coords, "T", T_coords, "epi_states", S_coords)
    ncwrite(compartments, output_fname, "data")
end