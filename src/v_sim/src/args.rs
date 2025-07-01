//==================================================================================================
// Structures
//==================================================================================================
use ::anyhow::Result;

pub struct Args{
    /// This defines the total number of VMs to be running in the simulation
    num_vms: usize,
    /// This defines the total number of vcpus in each VM
    num_vcpus: usize,
    /// This defines which scheduler VM's will be using
    scheduler: String,
    /// This defines which mapper VM's will be using
    mapper: String,
}

//==================================================================================================
// Implementation
//==================================================================================================
impl Args {
    const OPT_HELP: &'static str = "-help";
    const OPT_NUM_VMS: &'static str = "-num_vms";
    const OPT_NUM_VCPUS: &'static str = "-num_vcpus";
    const OPT_SCHEDULER: &'static str = "-scheduler";
    const OPT_MAPPER: &'static str = "-mapper";

    pub fn parse(args: Vec<String>) -> Result<Self> {
	let mut num_vms: usize = 1;
	let mut num_vcpus: usize = 4;
	let mut scheduler: String = "RoundRobin".to_string();
	let mut mapper: String = "FIFO".to_string();

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
		&_ => {
		    return Err(anyhow::anyhow!("invalid argument"));
		}
	    }
	    
	    i += 1;
	}

	Ok(Self {
	    num_vms,
	    num_vcpus,
	    scheduler,
	    mapper,
	})
    }

    pub fn usage(program_name: &str) {
	println!(
	    "Usage: {} [{} <num_vms> {} <num_vcpus> {} <['FIFO', 'RoundRobin']> {} <['FIFO']>]",
	    program_name,
	    Self::OPT_NUM_VMS,
	    Self::OPT_NUM_VCPUS,
	    Self::OPT_SCHEDULER,
	    Self::OPT_MAPPER,
	);
    }
	

    pub fn number_of_vms(&self) -> usize {
	self.num_vms
    }

    pub fn number_of_vcpus(&self) -> usize {
	self.num_vcpus
    }

    pub fn scheduler(&self) -> String {
        self.scheduler.clone()
    }
    
    pub fn mapper(&self) -> String {
        self.mapper.clone()
    }
}
