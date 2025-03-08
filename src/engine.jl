using NetCDF
using NCDatasets
using DataFrames
using MMCAcovid19
using MMCACovid19Vac

include("io.jl")


function get_engine(engine_name::String)
    engine_type = get(ENGINE_TYPES, engine_name, nothing)
    isnothing(engine_type) && error("Unknown engine: $engine_name")
    return engine_type()
end

function validate_config(config)
    @assert haskey(config, "simulation")
    simulation_dict = config["simulation"]
    @assert haskey(simulation_dict, "engine")
    engine_name = simulation_dict["engine"]
    engine = get_engine(engine_name)
    validate_config(config, engine)
    return engine
end

function validate_config(config, ::MMCACovid19VacEngine)
    @assert haskey(config, "simulation")
    @assert haskey(config, "data")
    @assert haskey(config, "epidemic_params")
    @assert haskey(config, "population_params")
    @assert haskey(config, "vaccination")
    @assert haskey(config, "NPI")
end

function validate_config(config, ::MMCACovid19Engine)
    @assert haskey(config, "simulation")
    @assert haskey(config, "data")
    @assert haskey(config, "epidemic_params")
    @assert haskey(config, "population_params")
    @assert haskey(config, "NPI") 
end


function read_input_files(::AbstractEngine, config::Dict, data_path::String, instance_path::String, init_condition_path::String)
    data_dict       = config["data"]
    simulation_dict = config["simulation"]
    pop_params_dict = config["population_params"]
    npi_params_dict = config["NPI"]
    init_format      = get(simulation_dict, "init_format", "netcdf")

    #########################
    # Simulation output 
    #########################
    output_path = joinpath(instance_path, "output")
    if !isdir(output_path)
        println("Creating output folder: $output_path")
        mkpath(output_path)
    end

    #########################################################
    # Containment measures
    #########################################################

    # Daily Mobility reduction
    kappa0_filename = get(data_dict, "kappa0_filename", nothing)
    first_day = Date(simulation_dict["start_date"])
    npi_params = init_NPI_parameters_struct(data_path, npi_params_dict, kappa0_filename, first_day)
    # vac_parms = Vaccination_Params(tᵛs, ϵᵍs)

    #####################
    # Initial Condition
    #####################

    if !isfile(init_condition_path) || length(init_condition_path) == 0
        init_condition_path = joinpath(data_path, get(data_dict, "initial_condition_filename", nothing))
    end

    @info "- Reading initial conditions from: $(init_condition_path)"
    # use initial compartments matrix to initialize simulations
    if init_format == "netcdf"
        initial_compartments = ncread(init_condition_path, "data")
    elseif init_format == "hdf5"
        # TODO: does this path work?
        initial_compartments = h5open(init_condition_path, "r") do file
            read(file, "data")
        end
    else
        @error "init_format must be one of : netcdf/hdf5"
        return 1
    end

    # Loading mobility network
    mobility_matrix_filename = joinpath(data_path, data_dict["mobility_matrix_filename"])
    network_df  = CSV.read(mobility_matrix_filename, DataFrame)

    G_labels = map(String, pop_params_dict["G_labels"])

    # Loading metapopulation patches info (surface, label, population by age)

    dtypes = Dict(vcat("id" => String, "area" => Float64, [i => Float64 for i in G_labels], "total" => Float64))
    
    metapop_data_filename = joinpath(data_path, data_dict["metapopulation_data_filename"])
    metapop_df = CSV.read(metapop_data_filename, DataFrame, types=dtypes)

    return npi_params, network_df, metapop_df, initial_compartments
end

"""
Run the engine using input files (which must be available in the data_path and instance_path)
and save the output to the output folder.
"""
function run_engine_io(engine::AbstractEngine, config::Dict, data_path::String, instance_path::String, init_condition_path::String)
    simulation_dict  = config["simulation"]
    output_format    = simulation_dict["output_format"]
    first_day        = Date(simulation_dict["start_date"])
    save_full_output = get(simulation_dict, "save_full_output", false)
    save_obs_output  = get(simulation_dict, "save_observables", false)
    time_step_tosave = get(simulation_dict, "save_time_step", nothing)
    output_path      = joinpath(instance_path, "output")

    # if output_path does not exist, create it
    if !isdir(output_path)
        mkpath(output_path)
    end

    npi_params, network_df, metapop_df, initial_compartments = read_input_files(engine, config, data_path, instance_path, init_condition_path)
    @info "Running simulation using: $(engine)"
    epi_params, population, coords = run_engine(engine, config, npi_params, network_df, metapop_df, initial_compartments)

    @info "\t- Save full output = $(save_full_output)" 
    if save_full_output
        save_full(engine, epi_params, population, output_path, output_format; coords...)
    end
    if save_obs_output
        save_observables(engine, epi_params, population, output_path; coords...)
    end
    if time_step_tosave !== nothing
        export_date = first_day + Day(time_step_tosave - 1)
        if  time_step_tosave <= epi_params.T
            @info "Storing compartments at single date $(export_date):"
            @info "\t- Simulation step: $(time_step_tosave)"
            save_time_step(engine, epi_params, population, output_path, time_step_tosave, export_date)
        else
            @error "Can't save simulation step ($(time_step_tosave)) larget then the last time step ($(params.T))"
        end
    end

    @info "done running engine io"
end

"""
Run the engine using Julia data structures as inputs. Does not save the output to file.

TODO: decouple from MMCACovid19Vac.jl (NPI_Params)
"""
function run_engine(engine::MMCACovid19VacEngine, config::Dict, 
                    npi_params::NPI_Params, network_df::DataFrame, 
                    metapop_df::DataFrame, initial_compartments::Array{Float64, 4})
    
    simulation_dict = config["simulation"]
    epi_params_dict = config["epidemic_params"]
    pop_params_dict = config["population_params"]
    vac_params_dict = config["vaccination"]

    ########################################
    ####### VARIABLES INITIALIZATION #######
    ########################################
    @info "Initializing variables"

    # Reading simulation start and end dates
    first_day = Date(simulation_dict["start_date"])
    last_day  = Date(simulation_dict["end_date"])
    # Converting dates to time steps
    T = (last_day - first_day).value + 1
    # Array with time coordinates (dates)
    T_coords  = string.(collect(first_day:last_day))

    # Metapopulations patches coordinates (labels)
    M_coords = map(String,metapop_df[:, "id"])
    M = length(M_coords)

    # Coordinates for each age strata (labels)
    G_coords = map(String, pop_params_dict["G_labels"])
    G = length(G_coords)

    ####################################################
    #####   INITIALIZATION OF DATA Structures   ########
    ####################################################
    @info "Initializing data structures"

    ## POPULATION PARAMETERS
    population = MMCACovid19Vac.init_pop_param_struct(G, M, G_coords, pop_params_dict, metapop_df, network_df)
    ## EPIDEMIC PARAMETERS 
    epi_params = MMCACovid19Vac.init_epi_parameters_struct(G, M, T, G_coords, epi_params_dict)

    @assert size(initial_compartments) == (G, M, epi_params.V, epi_params.NumComps)


    #########################################################
    # Vaccination parameters
    #########################################################
    @info "Initializing vaccination parameters"

    # vaccionation dates
    start_vacc = vac_params_dict["start_vacc"]
    dur_vacc   = vac_params_dict["dur_vacc"]
    end_vacc   = start_vacc + dur_vacc

    # total vaccinations per age strata
    total_population = sum(population.nᵢᵍ)
    ϵᵍ = vac_params_dict["ϵᵍ"] * round( total_population * vac_params_dict["percentage_of_vacc_per_day"] )
    tᵛs = [start_vacc, end_vacc, T]
    ϵᵍs = ϵᵍ .* [0  Int(vac_params_dict["are_there_vaccines"])  0] 

    ##################################################

    @info "- Initializing MMCA epidemic simulations for engine $(engine)"
    @info "\t- N. of epi compartments = $(epi_params.NumComps)" 
    @info "\t- G (agent class) = $(G)"
    @info "\t- M (n. of metapopulations) = $(M)"
    @info "\t- T (simulation steps) = $(T)"
    @info "\t- V (vaccination states) = $(epi_params.V)"
    @info "\t- first_day_simulation = $(first_day)"  
    @info "\t- last_day_simulation = $(last_day)"


    ########################################################
    ################ RUN THE SIMULATION ####################
    ########################################################

    MMCACovid19Vac.set_compartments!(epi_params, population, initial_compartments)

    MMCACovid19Vac.run_epidemic_spreading_mmca!(epi_params, population, npi_params, tᵛs, ϵᵍs; verbose = true )

    return epi_params, population, Dict(:T_coords => T_coords, :G_coords => G_coords, :M_coords => M_coords)
end


function run_engine(engine::MMCACovid19Engine, config::Dict, 
                    npi_params::NPI_Params, network_df::DataFrame, 
                    metapop_df::DataFrame, initial_compartments::Array{Float64, 3})
    
    n_compartments = 10

    @info "Running MMCACovid19Engine"
    simulation_dict = config["simulation"]
    epi_params_dict = config["epidemic_params"]
    pop_params_dict = config["population_params"]

    ###########################################
    ############# FILE READING ################
    ###########################################
    
    simulation_dict = config["simulation"]
    data_dict       = config["data"]
    epi_params_dict = config["epidemic_params"]
    pop_params_dict = config["population_params"]
    npi_params_dict = config["NPI"]

    ########################################
    ####### VARIABLES INITIALIZATION #######
    ########################################
    @info "Initializing variables"

    # Reading simulation start and end dates
    first_day = Date(simulation_dict["start_date"])
    last_day  = Date(simulation_dict["end_date"])
    # Converting dates to time steps
    T = (last_day - first_day).value + 1
    # Array with time coordinates (dates)
    T_coords  = string.(collect(first_day:last_day))

    # Metapopulations patches coordinates (labels)
    M_coords = map(String,metapop_df[:, "id"])
    M = length(M_coords)

    # Coordinates for each age strata (labels)
    G_coords = map(String, pop_params_dict["G_labels"])
    G = length(G_coords)

    ####################################################
    #####   INITIALIZATION OF DATA Structures   ########
    ####################################################
    @info "Initializing data structures"


    #######################################################
    ## POPULATION PARAMETERS
    #######################################################
    
    # Subpopulations' patch surface
    sᵢ = metapop_df[:, "area"]
    # Subpopulation by age strata
    nᵢᵍ = copy(transpose(Array{Float64,2}(metapop_df[:, G_coords])))
    nᵢᵍ = round.( nᵢᵍ)
    # Age Contact Matrix
    C = Float64.(mapreduce(permutedims, vcat, pop_params_dict["C"]))
    # Average number of contacts per strata
    kᵍ = Float64.(pop_params_dict["kᵍ"])
    # Average number of contacts at home per strata
    kᵍ_h = Float64.(pop_params_dict["kᵍ_h"])
    # Average number of contacts at work per strata
    kᵍ_w = Float64.(pop_params_dict["kᵍ_w"])
    # Degree of mobility per strata
    pᵍ = Float64.(pop_params_dict["pᵍ"])
    # Density factor
    ξ = pop_params_dict["ξ"]
    # Average household size
    σ = pop_params_dict["σ"]

    edgelist = Array{Int64, 2}(network_df[:, 1:2])
    Rᵢⱼ      = copy(network_df[:, 3])
    edgelist, Rᵢⱼ = correct_self_loops(edgelist, Rᵢⱼ, M)
    
    population = MMCAcovid19.Population_Params(G, M, nᵢᵍ, kᵍ, kᵍ_h, kᵍ_w, C, pᵍ, edgelist, Rᵢⱼ, sᵢ, ξ, σ)


    #######################################################
    ## EPIDEMIC PARAMETERS
    #######################################################

    # Scaling of the asymptomatic infectivity
    scale_β = Float64.(epi_params_dict["scale_β"])
    # Infectivity of Symptomatic
    βᴵ = Float64.(epi_params_dict["βᴵ"])
    # Infectivity of Asymptomatic
    if haskey(epi_params_dict, "βᴬ")
        βᴬ = Float64.(epi_params_dict["βᴬ"])
    elseif haskey(epi_params_dict, "scale_β")
        βᴬ = scale_β * βᴵ
    else
        @error "Either βᴬ or scale_β should be provided"
    end
    # Exposed rate
    ηᵍ = Float64.(epi_params_dict["ηᵍ"])
    # Asymptomatic rate
    αᵍ = Float64.(epi_params_dict["αᵍ"])
    # Infectious rate
    μᵍ = Float64.(epi_params_dict["μᵍ"])
    # Direct death probability
    θᵍ = Float64.(epi_params_dict["θᵍ"])
    # Hospitalization probability
    γᵍ = Float64.(epi_params_dict["γᵍ"])
    # Fatality probability in ICU
    ωᵍ = Float64.(epi_params_dict["ωᵍ"])
    # Pre-deceased rate
    ζᵍ = Float64.(epi_params_dict["ζᵍ"])
    # Pre-hospitalized in ICU rate
    λᵍ = Float64.(epi_params_dict["λᵍ"])
    # Death rate in ICU
    ψᵍ = Float64.(epi_params_dict["ψᵍ"])
    # ICU discharge rate
    χᵍ = Float64.(epi_params_dict["χᵍ"])
    # Relative risk reduction of the probability of infection
    rᵥ = Float64.(epi_params_dict["rᵥ"])
    # Relative risk reduction of the probability of transmission
    kᵥ = Float64.(epi_params_dict["kᵥ"])

    epi_params = MMCAcovid19.Epidemic_Params(βᴵ, βᴬ, ηᵍ, αᵍ, μᵍ, θᵍ, γᵍ, ζᵍ, λᵍ, ωᵍ, ψᵍ, χᵍ, G, M, T)
    
    @assert size(initial_compartments) == (G, M, n_compartments)

    #########################################################
    # Containment measures
    #########################################################

    # Timesteps when the containment measures will be applied
    tᶜs = npi_params.tᶜs
    # Array of level of confinement
    κ₀s = npi_params.κ₀s
    # Array of premeabilities of confined households
    ϕs = npi_params.ϕs
    # Array of social distancing measures
    δs = npi_params.δs

    ##################################################

    @info "- Initializing MMCA epidemic simulations for engine $(engine)"
    @info "\t- N. of epi compartments = $(n_compartments)" 
    @info "\t- G (agent class) = $(G)"
    @info "\t- M (n. of metapopulations) = $(M)"
    @info "\t- T (simulation steps) = $(T)"
    @info "\t- first_day_simulation = $(first_day)"  
    @info "\t- last_day_simulation = $(last_day)"
    
    t₀ = 1
    epi_params.ρˢᵍ[:,:,t₀]  .= initial_compartments[:, :, 1] ./ population.nᵢᵍ
    epi_params.ρᴱᵍ[:,:,t₀]  .= initial_compartments[:, :, 2] ./ population.nᵢᵍ
    epi_params.ρᴬᵍ[:,:,t₀]  .= initial_compartments[:, :, 3] ./ population.nᵢᵍ
    epi_params.ρᴵᵍ[:,:,t₀]  .= initial_compartments[:, :, 4] ./ population.nᵢᵍ
    epi_params.ρᴾᴴᵍ[:,:,t₀] .= initial_compartments[:, :, 5] ./ population.nᵢᵍ
    epi_params.ρᴾᴰᵍ[:,:,t₀] .= initial_compartments[:, :, 6] ./ population.nᵢᵍ
    epi_params.ρᴴᴿᵍ[:,:,t₀] .= initial_compartments[:, :, 7] ./ population.nᵢᵍ
    epi_params.ρᴴᴰᵍ[:,:,t₀] .= initial_compartments[:, :, 8] ./ population.nᵢᵍ
    epi_params.ρᴿᵍ[:,:,t₀]  .= initial_compartments[:, :, 9] ./ population.nᵢᵍ
    epi_params.ρᴰᵍ[:,:,t₀]  .= initial_compartments[:, :, 10] ./ population.nᵢᵍ

    epi_params.ρˢᵍ[isnan.(epi_params.ρˢᵍ)]   .= 0
    epi_params.ρᴱᵍ[isnan.(epi_params.ρᴱᵍ)]   .= 0
    epi_params.ρᴬᵍ[isnan.(epi_params.ρᴬᵍ)]   .= 0
    epi_params.ρᴵᵍ[isnan.(epi_params.ρᴵᵍ)]   .= 0
    epi_params.ρᴾᴴᵍ[isnan.(epi_params.ρᴾᴴᵍ)] .= 0
    epi_params.ρᴾᴰᵍ[isnan.(epi_params.ρᴾᴰᵍ)] .= 0
    epi_params.ρᴴᴿᵍ[isnan.(epi_params.ρᴴᴿᵍ)] .= 0
    epi_params.ρᴴᴰᵍ[isnan.(epi_params.ρᴴᴰᵍ)] .= 0
    epi_params.ρᴿᵍ[isnan.(epi_params.ρᴿᵍ)]   .= 0
    epi_params.ρᴰᵍ[isnan.(epi_params.ρᴰᵍ)]   .= 0

    ########################################################
    ################ RUN THE SIMULATION ####################
    ########################################################

    MMCAcovid19.run_epidemic_spreading_mmca!(epi_params, population, tᶜs, κ₀s, ϕs, δs, verbose=false)

    return epi_params, population, Dict(:T_coords => T_coords, :G_coords => G_coords, :M_coords => M_coords)
end
