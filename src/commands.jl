include("engine.jl")

function parse_command_line()
    s = ArgParseSettings()

    @add_arg_table s begin
        "run"
            help = "Run an epidemic simulation using the given engine"
            action = :command
        "setup"
            help = "Setup a model (config, and the required (empty) datafiles) for the give engine"
            action = :command
        "init"
            help = "Create an initial condition for the given engine"
            action = :command
    end

    @add_arg_table s["run"] begin
        "--config", "-c"
            help = "config file (json file)"
            required = true
        "--data-folder", "-d"
            help = "data folder"
            required = true
        "--instance-folder", "-i"
            help = "instance folder (experiment folder)"
            default = "." 
        "--export-compartments-full"
            help = "export compartments of simulations"
            action = :store_true
        "--export-compartments-time-t"
            help = "export compartments of simulations at a given time"
            default = nothing
            arg_type = Int
        "--initial-condition"
            help = "compartments to initialize simulation. If missing, use the seeds to initialize the simulations"
            default = ""
        "--start-date"
            help = "starting date of simulation. Overwrites the one provided in config.json"
            default = nothing
        "--end-date"
            help = "end date of simulation. Overwrites the one provided in config.json"
            default = nothing
    end

    @add_arg_table s["setup"] begin
        "--name", "-n"
            help = "Model name (will be used to create a folder)"
            required = true
            arg_type = String
        "--metapop", "-M"
            help = "Number of metapopulation compartments or regions"
            required = true
            arg_type = Int
        "--agents", "-G"
            help = "instance folder (experiment folder)"
            required = true
            arg_type = Int
        "--output", "-o"
            help = "Path where template model will be created"
            default = "models"
        "--engine", "-e"
            help = "Simulator Engine"
            default = "MMCACovid19Vac"
    end

    @add_arg_table s["init"] begin
        "--config", "-c"
            help = "Config file (json file)"
            required = true
        "--data-folder", "-d"
            help = "Data folder. Folder where the data files are stored"
            required = true
        "--seeds"
            help = "CSV file with initial seeds (initial infected individuals). It is used to create the initial condition file"
            required = true
        "--out", "-o"
            help = "Output file name for storing the condition in NetCDF format"
            required = false
            default = "initial_conditions.nc"
    end
    return parse_args(s)
end





## ----------------------------------------
## Command function
## ----------------------------------------

function execute_run(args)

    data_path     = args["data-folder"]
    config_fname  = args["config"]
    instance_path = args["instance-folder"]

    init_condition_path = args["initial-condition"]
    
    config = JSON.parsefile(config_fname);
    update_config!(config, args)

    @assert isfile(config_fname);
    @assert isdir(data_path);
    @assert isdir(instance_path);
    
    engine = validate_config(config)

    run_engine_io(engine, config, data_path, instance_path, init_condition_path)
    @info "done executing run command"
end

function execute_setup(args)
    name = args["name"]

    
    engine = get_engine(args["engine"])
    @info "Creating config file for engine: $engine"

    M = args["metapop"]
    G = args["agents"]
    
    output_path = args["output"]
    @assert ispath(output_path)
    model_path = joinpath(output_path, name)
    if !ispath(model_path)
        @info "Creating folder for storing model: $model_path"
        mkdir(model_path)
    end
    
    config_fname = joinpath(model_path, BASE_CONFIG_NAME)
    config = create_config_template(engine, M, G)
    @info "Writing model definition (JSON): $config_fname"
    open(config_fname, "w") do fh
        JSON.print(fh, config, 4)
    end
    G_labels = copy(config["population_params"]["G_labels"])
    cols = vcat("area", G_labels)

    df = DataFrame([i=>ones(M) for i in cols])
    df[!, :total] = ones(M) * G
    metapop_fname = joinpath(model_path, BASE_METAPOP_NAME)
    CSV.write(metapop_fname, df)

end

function execute_init(args)
    config_fname  = args["config"]
    data_folder   = args["data-folder"]
    output_folder = args["out"]
    seeds_fname   = args["seeds"]

    config          = JSON.parsefile(config_fname);
    data_dict       = config["data"]
    pop_params_dict = config["population_params"]
    epi_params_dict = config["epidemic_params"]

    output_fname = joinpath(output_folder, "initial_conditions_10.nc")

    # Reading metapopulation Dataframe
    metapop_data_filename = joinpath(data_folder, data_dict["metapopulation_data_filename"])
    metapop_df = CSV.read(metapop_data_filename, DataFrame, types=Dict("id" => String, 
    "area"=>Float64, "Y"=>Float64, "M"=>Float64, "O"=>Float64, "Total"=>Float64))

    # Loading mobility network
    mobility_matrix_filename = joinpath(data_folder, data_dict["mobility_matrix_filename"])
    network_df  = CSV.read(mobility_matrix_filename, DataFrame)

    # Metapopulations patches coordinates (labels)
    M_coords = map(String,metapop_df[:, "id"])
    M = length(M_coords)

    # Coordinates for each age strata (labels)
    G_coords = map(String, pop_params_dict["G_labels"])
    G = length(G_coords)

    T = 1

    population = init_pop_param_struct(G, M, G_coords, pop_params_dict, metapop_df, network_df)
    epi_params = init_epi_parameters_struct(G, M, T, G_coords, epi_params_dict)

    S = epi_params.NumComps
    comp_coords = epi_params.CompLabels
    
    V = epi_params.V
    V_coords = epi_params.VaccLabels

    conditions₀ = CSV.read(seeds_fname, DataFrame)
    patches_idxs = Int.(conditions₀[:, "idx"])
    
    G_fractions = [0.12 0.16 0.72]
    
    println("- Creating compartment array")
    compartments = zeros(Float64, G, M, V, S);
    
    S_idx = 1
    A_idx = 3
     @printf("- Setting infected %.1f seeds in compartment A\n", sum(conditions₀[:, "seed"]))
    compartments[1, patches_idxs, 1, A_idx] .= G_fractions[1] .* conditions₀[:, "seed"]
    compartments[2, patches_idxs, 1, A_idx] .= G_fractions[2] .* conditions₀[:, "seed"]
    compartments[3, patches_idxs, 1, A_idx] .= G_fractions[3] .* conditions₀[:, "seed"]
    
    compartments[:, :, 1, S_idx]  .= population.nᵢᵍ - compartments[:, :, 1, A_idx]
    @printf("- Setting remaining population %.1f in compartment S\n", sum(compartments[:, :, 1, S_idx]))
    @printf("- Saving intital conditions in '%s' \n", output_fname)
    nccreate(output_fname, "data", "G", G_coords, "M", M_coords, "V", V_coords, "epi_states", collect(comp_coords))
    ncwrite(compartments, output_fname, "data")
end





## ------------------------------------------------------------
## Auxiliary functions
## ------------------------------------------------------------

function read_config()
    
    args = parse_commandline()
    
    config_fname = args["config"]
    data_path    = args["data-folder"]
    seeds_fname  = args["seeds"]
    output_fname = args["out"]
    
    
    @assert isfile(config_fname);
    @assert isdir(data_path);
    @assert isfile(seeds_fname);
    
    if isfile(output_fname)
        @printf("- ERROR output file '%s' already existe\n", output_fname)
        exit(1)
    end
    
    config = JSON.parsefile(config_fname);
    
    println("- Loading required data")
    data_dict       = config["data"]
    epi_params_dict = config["epidemic_params"]
    pop_params_dict = config["population_params"]
    
    # Loading metapopulation patches info (surface, label, population by age)
    metapop_data_filename = joinpath(data_path, data_dict["metapopulation_data_filename"])
    metapop_df = CSV.read(metapop_data_filename, DataFrame, types= Dict("id"=>String, 
        "area"=>Float64, "Y"=>Float64, "M"=>Float64, "O"=>Float64, "Total"=>Float64))
    
    # Loading mobility network
    mobility_matrix_filename = joinpath(data_path, data_dict["mobility_matrix_filename"])
    network_df  = CSV.read(mobility_matrix_filename, DataFrame)
    
    # Single time step
    T = 1
    # Metapopulations patches coordinates (labels)
    M_coords = map(String, metapop_df[:, "id"])
    M = length(M_coords)
    # Coordinates for each age strata (labels)
    G_coords = map(String, pop_params_dict["age_labels"])
    G = length(G_coords)
    # Num. of vaccination statuses Vaccinated/Non-vaccinated
    V_coords = ["NV", "V"]
    V = length(epi_params_dict["kᵥ"])
    
    
    ## POPULATION PARAMETERS
    population       = init_pop_param_struct(G, M, G_coords, pop_params_dict, metapop_df, network_df)
    ## EPIDEMIC PARAMETERS 
    epi_params       = init_epi_parameters_struct(G, M, 1, G_coords, epi_params_dict)
    
    
    comp_coords = epi_params.CompLabels
    S = length(comp_coords)
end

function create_core_config()
    config = Dict( "simulation" => Dict(), 
                   "data" => Dict(), 
                   "epidemic_params" => Dict(), 
                   "population_params" => Dict()
                   )

    config["simulation"]["start-date"] = "01-01-2020"
    config["simulation"]["end-date"]   = "02-15-2020"
    config["simulation"]["save_full_output"] = true
    config["simulation"]["export_compartments_time_t"] = nothing
    config["simulation"]["output_folder"] = "output"
    config["simulation"]["output_format"] = "netcdf"
    
    config["data"]["initial_condition_filename"] = "initial_conditions.nc"
    config["data"]["metapopulation_data_filename"] = "metapopulation_data.csv"
    config["data"]["mobility_matrix_filename"] = "R_mobility_matrix.csv"
    config["data"]["kappa0_filename"] = "kappa0.csv"
    
    return config
end

function create_config_template(::MMCACovid19Engine, M::Int, G::Int)
    config = create_core_config()
    config["simulation"]["engine"] = "MMCACovid19"

    epiparams_dict = Dict()
    epiparams_dict["scale_β"] = 0.5
    epiparams_dict["βᴬ"] = 0.05
    epiparams_dict["βᴵ"] = 0.09
    epiparams_dict["ηᵍ"] = ones(G) * 0.275
    epiparams_dict["αᵍ"] = ones(G) * 0.65
    epiparams_dict["μᵍ"] = ones(G) * 0.3
    epiparams_dict["θᵍ"] = zeros(G)
    epiparams_dict["γᵍ"] = ones(G) * 0.03
    epiparams_dict["ζᵍ"] = ones(G) * 0.12
    epiparams_dict["λᵍ"] = ones(G) * 0.275
    epiparams_dict["ωᵍ"] = ones(G) * 0.1
    epiparams_dict["ψᵍ"] = ones(G) * 0.14
    epiparams_dict["χᵍ"] = ones(G) * 0.047

    population = Dict()
    population["G_labels"] = ["G" * string(i) for i in 1:G]
    population["C"] = ones(G,G) * 1/G
    population["kᵍ"] = ones(G) * 10
    population["kᵍ_h"] = ones(G) * 3
    population["kᵍ_w"] = ones(G) 
    population["pᵍ"] = ones(G) 
    population["ξ"] = 0.01
    population["σ"] = 2.5

    npiparams_dict = Dict()
    npiparams_dict["κ₀s"] = [0.0]
    npiparams_dict["ϕs"] = [1.0]
    npiparams_dict["δs"] = [0.0]
    npiparams_dict["tᶜs"] =  [1]
    npiparams_dict["are_there_npi"] = true

    config["epidemic_params"] = epiparams_dict
    config["population_params"] = population
    config["NPI"] = npiparams_dict

    return config
end

function create_config_template(::MMCACovid19VacEngine, M::Int, G::Int)
    config = create_core_config()
    config["simulation"]["engine"] = "MMCACovid19Vac"
    config = merge(config, MMCACovid19Vac.create_config_template(G))
    return config
end
