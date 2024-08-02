

function run_MMCACovid19Vac(config::Dict, data_path::String, instance_path::String, init_condition_path::String)

    ###########################################
    ############# FILE READING ################
    ###########################################
    
    simulation_dict = config["simulation"]
    data_dict       = config["data"]
    epi_params_dict = config["epidemic_params"]
    pop_params_dict = config["population_params"]
    vac_params_dict = config["vaccination"]
    npi_params_dict = config["NPI"]

    #########################
    # Simulation output 
    #########################
    output_path = joinpath(instance_path, "output")
    if !isdir(output_path)
        println("Creating output folder: $output_path")
        mkpath(output_path)
    end

    output_format    = simulation_dict["output_format"]
    save_full_output = get(simulation_dict, "save_full_output", false)
    save_time_step   = get(simulation_dict, "export_compartments_time_t", nothing)
    init_format      = get(simulation_dict, "init_format", "netcdf")

    #####################
    # Initial Condition
    #####################

    if !isfile(init_condition_path) || length(init_condition_path) == 0
        init_condition_path = joinpath(data_path, get(data_dict, "initial_condition_filename", nothing))
    end


    # use initial compartments matrix to initialize simulations
    if init_format == "netcdf"
        @info "Reading initial conditions from: $(init_condition_path)"
        initial_compartments = ncread(init_condition_path, "data")
    elseif init_format == "hdf5"
        initial_compartments = h5open(init_condition_path, "r") do file
            read(file, "data")
        end
    else
        @error "init_format must be one of : netcdf/hdf5"
        return 1
    end


    ########################################
    ####### VARIABLES INITIALIZATION #######
    ########################################

    # Reading simulation start and end dates
    first_day = Date(simulation_dict["start_date"])
    last_day  = Date(simulation_dict["end_date"])
    # Converting dates to time steps
    T = (last_day - first_day).value + 1
    # Array with time coordinates (dates)
    T_coords  = string.(collect(first_day:last_day))

    # Loading metapopulation patches info (surface, label, population by age)
    metapop_data_filename = joinpath(data_path, data_dict["metapopulation_data_filename"])
    metapop_df = CSV.read(metapop_data_filename, DataFrame, types=Dict(:id => String))

    # Loading mobility network
    mobility_matrix_filename = joinpath(data_path, data_dict["mobility_matrix_filename"])
    network_df  = CSV.read(mobility_matrix_filename, DataFrame)

    # Metapopulations patches coordinates (labels)
    M_coords = map(String,metapop_df[:, "id"])
    M = length(M_coords)

    # Coordinates for each age strata (labels)
    G_coords = map(String, pop_params_dict["G_labels"])
    G = length(G_coords)

    ####################################################
    #####   INITIALIZATION OF DATA Structures   ########
    ####################################################

    ## POPULATION PARAMETERS
    population = MMCACovid19Vac.init_pop_param_struct(G, M, G_coords, pop_params_dict, metapop_df, network_df)
    ## EPIDEMIC PARAMETERS 
    epi_params = MMCACovid19Vac.init_epi_parameters_struct(G, M, T, G_coords, epi_params_dict)

    @assert size(initial_compartments) == (G, M, epi_params.V, epi_params.NumComps)


    #########################################################
    # Vaccination parameters
    #########################################################

    # vaccionation dates
    start_vacc = vac_params_dict["start_vacc"]
    dur_vacc   = vac_params_dict["dur_vacc"]
    end_vacc   = start_vacc + dur_vacc

    # total vaccinations per age strata
    total_population = sum(population.nᵢᵍ)
    ϵᵍ = vac_params_dict["ϵᵍ"] * round( total_population * vac_params_dict["percentage_of_vacc_per_day"] )
    tᵛs = [start_vacc, end_vacc, T]
    ϵᵍs = ϵᵍ .* [0  Int(vac_params_dict["are_there_vaccines"])  0] 

    #########################################################
    # Containment measures
    #########################################################

    # Daily Mobility reduction
    kappa0_filename = get(data_dict, "kappa0_filename", nothing)
    npi_params = MMCACovid19Vac.init_NPI_parameters_struct(data_path, npi_params_dict, kappa0_filename, first_day)

    # vac_parms = Vaccination_Params(tᵛs, ϵᵍs)

    ##################################################

    @info "- Initializing MMCA epidemic simulations"
    @info "\t- first_day_simulation = "  first_day
    @info "\t- last_day_simulation = " last_day
    @info "\t- G (agent class) = " G
    @info "\t- M (n. of metapopulations) = "  M
    @info "\t- T (simulation steps) = " T
    @info "\t- V (vaccination states) = " epi_params.V
    @info "\t- N. of epi compartments = " epi_params.NumComps
    @info "\t- Save full output = " save_full_output
    if save_time_step !== nothing
        @info "\t- Save time step at t=" save_time_step
    end

    ########################################################
    ################ RUN THE SIMULATION ####################
    ########################################################

    MMCACovid19Vac.set_compartments!(epi_params, population, initial_compartments)

    MMCACovid19Vac.run_epidemic_spreading_mmca!(epi_params, population, npi_params, tᵛs, ϵᵍs; verbose = true )

    ##############################################################
    ################## STORING THE RESULTS #######################
    ##############################################################

    if save_full_output
        @info "Storing full simulation output in $(output_format)"
        if output_format == "netcdf"
            filename = joinpath(output_path, "compartments_full.nc")
            @info "\t- Output filename: $(filename)"
            save_simulation_netCDF(epi_params, population, filename;G_coords=G_coords, M_coords=M_coords, T_coords=T_coords)
        elseif output_format == "hdf5"
            filename = joinpath(output_path, "compartments_full.h5")
            @info "\t- Output filename: $(filename)"
            save_simulation_hdf5(epi_params, population, filename)
        end
    end

    if save_time_step !== nothing
        export_compartments_date = first_day + Day(export_compartments_time_t - 1)
        filename = joinpath(output_path, "compartments_t_$(export_compartments_date).h5")
        @info "Storing compartments at single date $(export_compartments_date):"
        @info "\t- Simulation step: $(export_compartments_time_t)"
        @info "\t- filename: $(filename)"
        save_simulation_hdf5(epi_params, population, filename; 
                            export_time_t = export_compartments_time_t)
    end

end


function run_MMCACovid19(config::Dict, data_path::String, instance_path::String, init_condition_path::String)

    ###########################################
    ############# FILE READING ################
    ###########################################
    
    simulation_dict = config["simulation"]
    data_dict       = config["data"]
    epi_params_dict = config["epidemic_params"]
    pop_params_dict = config["population_params"]
    npi_params_dict = config["NPI"]

    #########################
    # Simulation output 
    #########################
    output_path = joinpath(instance_path, "output")
    if !isdir(output_path)
        println("Creating output folder: $output_path")
        mkpath(output_path)
    end

    output_format    = simulation_dict["output_format"]
    save_full_output = get(simulation_dict, "save_full_output", false)
    save_time_step   = get(simulation_dict, "save_time_step", nothing)
    init_format      = get(simulation_dict, "init_format", "netcdf")

    #####################
    # Initial Condition
    #####################

    if !isfile(init_condition_path) || length(init_condition_path) == 0
        init_condition_path = joinpath(data_path, get(data_dict, "initial_condition_filename", nothing))
    end


    # use initial compartments matrix to initialize simulations
    if init_format == "netcdf"
        @info "Reading initial conditions from: $(init_condition_path)"
        initial_compartments = ncread(init_condition_path, "data")
    elseif init_format == "hdf5"
        initial_compartments = h5open(init_condition_path, "r") do file
            read(file, "data")
        end
    else
        @error "init_format must be one of : netcdf/hdf5"
        return 1
    end


    ########################################
    ####### VARIABLES INITIALIZATION #######
    ########################################

    # Reading simulation start and end dates
    first_day = Date(simulation_dict["start_date"])
    last_day  = Date(simulation_dict["end_date"])
    # Converting dates to time steps
    T = (last_day - first_day).value + 1
    # Array with time coordinates (dates)
    T_coords  = string.(collect(first_day:last_day))

    # Loading metapopulation patches info (surface, label, population by age)
    metapop_data_filename = joinpath(data_path, data_dict["metapopulation_data_filename"])
    metapop_df = CSV.read(metapop_data_filename, DataFrame, types=Dict(:id => String))

    # Loading mobility network
    mobility_matrix_filename = joinpath(data_path, data_dict["mobility_matrix_filename"])
    network_df  = CSV.read(mobility_matrix_filename, DataFrame)

    # Metapopulations patches coordinates (labels)
    M_coords = map(String,metapop_df[:, "id"])
    M = length(M_coords)

    # Coordinates for each age strata (labels)
    G_coords = map(String, pop_params_dict["G_labels"])
    G = length(G_coords)

    ####################################################
    #####   INITIALIZATION OF DATA Structures   ########
    ####################################################

    ## POPULATION PARAMETERS
    # population = MMCACovid19Vac.init_pop_param_struct(G, M, G_coords, pop_params_dict, metapop_df, network_df)
    #  crear estructura a a mano
    ## EPIDEMIC PARAMETERS 
    # epi_params = MMCACovid19Vac.init_epi_parameters_struct(G, M, T, G_coords, epi_params_dict)
    #  crear estructura a a mano

    # @assert size(initial_compartments) == (G, M, epi_params.V, epi_params.NumComps)

    #########################################################
    # Containment measures
    #########################################################

    # Daily Mobility reduction
    kappa0_filename = get(data_dict, "kappa0_filename", nothing)
    npi_params = MMCACovid19Vac.init_NPI_parameters_struct(data_path, npi_params_dict, kappa0_filename, first_day)

    # vac_parms = Vaccination_Params(tᵛs, ϵᵍs)

    ##################################################

    @info "- Initializing MMCA epidemic simulations"
    @info "\t- first_day_simulation = "  first_day
    @info "\t- last_day_simulation = " last_day
    @info "\t- G (agent class) = " G
    @info "\t- M (n. of metapopulations) = "  M
    @info "\t- T (simulation steps) = " T
    @info "\t- N. of epi compartments = " epi_params.NumComps
    @info "\t- Save full output = " save_full_output
    if save_time_step !== nothing
        @info "\t- Save time step at t=" save_time_step
    end

    ########################################################
    ################ RUN THE SIMULATION ####################
    ########################################################

    MMCACovid19Vac.set_compartments!(epi_params, population, initial_compartments)

    MMCACovid19Vac.run_epidemic_spreading_mmca!(epi_params, population, npi_params, tᵛs, ϵᵍs; verbose = true )

    ##############################################################
    ################## STORING THE RESULTS #######################
    ##############################################################

    if save_full_output
        @info "Storing full simulation output in $(output_format)"
        if output_format == "netcdf"
            filename = joinpath(output_path, "compartments_full.nc")
            @info "\t- Output filename: $(filename)"
            save_simulation_netCDF(epi_params, population, filename;G_coords=G_coords, M_coords=M_coords, T_coords=T_coords)
        elseif output_format == "hdf5"
            filename = joinpath(output_path, "compartments_full.h5")
            @info "\t- Output filename: $(filename)"
            save_simulation_hdf5(epi_params, population, filename)
        end
    end

    if save_time_step !== nothing
        export_compartments_date = first_day + Day(export_compartments_time_t - 1)
        filename = joinpath(output_path, "compartments_t_$(export_compartments_date).h5")
        @info "Storing compartments at single date $(export_compartments_date):"
        @info "\t- Simulation step: $(export_compartments_time_t)"
        @info "\t- filename: $(filename)"
        save_simulation_hdf5(epi_params, population, filename; 
                            export_time_t = export_compartments_time_t)
    end

end