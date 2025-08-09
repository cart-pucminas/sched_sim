//==================================================================================================
// Structures
//==================================================================================================
use ::anyhow::Result;

pub struct Args{
    /// This defines the total number of VMs to be running in the simulation
    num_vms: usize,
    /// This defines the total number of vcpus in each VM
    num_vcpus: usize,
    /// This defines the total number of tasks in the simulation
    num_tasks: usize,
    /// This defines which scheduler VM's will be using
    scheduler: String,
    /// This defines which mapper VM's will be using
    mapper: String,
    ///

    seed: usize,
}

//==================================================================================================
// Implementation
//==================================================================================================
impl Args {
    const OPT_HELP: &'static str = "--help";
    const OPT_NUM_VMS: &'static str = "--num_vms";
    const OPT_NUM_VCPUS: &'static str = "--num_vcpus";
    const OPT_NUM_TASKS: &'static str = "--num_tasks";
    const OPT_SCHEDULER: &'static str = "--scheduler";
    const OPT_MAPPER: &'static str = "--mapper";
    const OPT_SEED: &'static str = "--seed";

    pub fn parse(args: Vec<String>) -> Result<Self> {
	let mut num_vms: usize = 1;
	let mut num_vcpus: usize = 1;
	let mut num_tasks: usize = 25;
	let mut scheduler: String = "RoundRobin".to_string();
	let mut mapper: String = "FIFO".to_string();
	let mut seed: usize = 1;
	
	let mut i: usize = 1;
	while i < args.len() {
	    match args[i].as_str() {
		Self::OPT_HELP => {
		    Self::usage(args[0].as_str());
		    return Err(anyhow::anyhow!("wrong usage"));
		}
		Self::OPT_NUM_VMS => {
		    i += 1;
		    num_vms = args[i].parse::<usize>().unwrap();
		},
		Self::OPT_NUM_TASKS => {
		    i += 1;
		    num_tasks = args[i].parse::<usize>().unwrap();
		},
		Self::OPT_NUM_VCPUS => {
		    i += 1;
		    num_vcpus = args[i].parse::<usize>().unwrap();
		},
		Self::OPT_SCHEDULER => {
		    i += 1;
		    scheduler = args[i].clone();
		}
		Self::OPT_MAPPER => {
		    i += 1;
		    mapper = args[i].clone();
		}
		Self::OPT_SEED => {
		    i += 1;
		    seed = args[i].parse::<usize>().unwrap();
		}
		&_ => {
		    return Err(anyhow::anyhow!("invalid argument"));
		}
	    }
	    
	    i += 1;
	}

	Ok(Self {
	    num_vms,
	    num_tasks,
	    num_vcpus,
	    scheduler,
	    mapper,
	    seed,
	})
    }

    pub fn usage(program_name: &str) {
	println!(
	    "Usage: {} [{} <num_vms> {} <num_vcpus> {} <['FIFO', 'RoundRobin']> {} <['FIFO']> {} <seed>]",
	    program_name,
	    Self::OPT_NUM_VMS,
	    Self::OPT_NUM_VCPUS,
	    Self::OPT_SCHEDULER,
	    Self::OPT_MAPPER,
	    Self::OPT_SEED,
	);
    }
	

    pub fn number_of_vms(&self) -> usize {
	self.num_vms
    }

    pub fn number_of_vcpus(&self) -> usize {
	self.num_vcpus
    }

    pub fn number_of_tasks(&self) -> usize {
	self.num_tasks
    }

    pub fn seed(&self) -> usize {
	self.seed
    }

    pub fn scheduler(&self) -> String {
        self.scheduler.clone()
    }
    
    pub fn mapper(&self) -> String {
        self.mapper.clone()
    }
}
