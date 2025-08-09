//==================================================================================================
// Configuration
//==================================================================================================
#![deny(clippy::all)]

//==================================================================================================
// Imports
//==================================================================================================
use mem_lib::Addr;
use ram_lib::RAM;
use anyhow::Result;
use log::debug;
use std::{
    sync::{Arc},
    thread,
    collections::{
	HashMap,
	VecDeque,
    },
    time::{
	Instant,
	Duration,
    },
};
use tokio::sync::{Mutex};

//==================================================================================================
// Structures
//==================================================================================================
/// Represents a Task to be scheduled in a Virtual Machine (VM) in the simulation.
#[derive(Clone)]
pub struct Task
{
    /// Task unique identifier (private field)
    pub ts_id		: u8,
    /// Task application name (private field)
    ts_name		: String,
    /// Task total workload to be processed (private field)
    ts_total_workload	: u64,
    /// Task workload left to be processed (private field)
    ts_left_workload	: u64,
    /// Task current execution time to process (private field)
    ts_current_exectime : u64,
    /// Task total memory addresses to be accessed during simulation (private field)
    ts_mem_addresses	: Vec<Addr>,
    /// Task indexer to the last memory address accessed (private field)
    ts_mem_indexer      : u64,
    /// Task cr3 addr.
    ts_cr3              : u64,

    ts_cache_acesses    : HashMap<usize, u64>, // Global hotness
    ts_recent_acesses   : VecDeque<usize>,    // Last N acesses

    ts_last_vcpu_id      : usize,
    ts_current_vcpu_id   : usize,


    // Times
    pub arrival_time		: Instant,
    pub start_time		: Option<Instant>,
    pub first_response_time     : Option<Duration>,
    pub accumulated_wait_time   : Duration,
    pub accumulated_exec_time   : Duration,
    pub last_left_time          : Option<Instant>,
					 
    
}

//==================================================================================================
// Implementations
//==================================================================================================
impl Task
{
    /// Creates a new Task instance.
    ///
    /// # Arguments
    /// * `id`        - The unique identifier of a Task;
    /// * `name`      - The application name; (*MIGHT BE REMOVED IN A NEAR FUTURE*)
    /// * `workload`  - The Task's total workload to be processed (Duration);
    /// * `addresses` - Memory addresses to be accessed during simulation
    ///
    /// # Returns
    /// * A new `Task` instance
    pub async fn new(
	id: u8,
	ram: Arc<Mutex<RAM>>,
	name: &str,
	workload: u64,
	addresses: Vec<Addr>
    ) -> Self
    {

	let mut ram_guard = ram.lock().await;
	let cr3 = ram_guard.ram_alloc_frame();
	ram_guard.ram_create_plm4_for_task(cr3);
	
	debug!(
	    "Creating Task ({}) ({:?})",
	    id,
	    thread::current().id(),
	);
	Self {
	    ts_id: id,
	    ts_name: name.to_string(),
	    ts_total_workload: workload,
	    ts_left_workload: workload,
	    ts_current_exectime: 0,
	    ts_mem_addresses: addresses,
	    ts_mem_indexer: 0,
	    ts_cr3: cr3,
	    
	    ts_cache_acesses: HashMap::new(),
	    ts_recent_acesses: VecDeque::with_capacity(300),
	    ts_last_vcpu_id: usize::MAX,
	    ts_current_vcpu_id: usize::MAX,

	    arrival_time		: Instant::now(),
	    start_time			: None,
	    first_response_time		: None,
	    accumulated_wait_time	: Duration::from_nanos(0),
	    accumulated_exec_time	: Duration::from_nanos(0),
	    last_left_time		: None,
	}
	
    }

    /// Represents the operation of processing a Task's workload. Basically, reduces Task's left_workload based in specified argument, and increment Task's `mem_indexer`
    ///
    /// # Arguments
    /// * `processed_workload` - The amount of workload that was processed in the scheduling iteration.
    ///
    /// # Returns
    ///
    /// * `Ok(u64)`     - `ts_left_workload` if `processed_workload` amount is valid
    /// * `Err(String)` - if `processed_workload` is greater than the left amount
    ///
    /// # Panics
    /// 
    /// * This function does **not** panic. It returns a `Result` instead.
    /// 
    /// # Examples
    ///
    /// ```rust
    /// let mut task = Task::new(0, "foo", 100);
    /// let remaining = task.task_process_workload(100).unwrap();
    /// println!("Remainder {}", remaining);
    /// ```
    pub fn task_process_workload(&mut self, processed_workload: u64) -> Result<u64, String>
    {
	/* Sanity Check */
	self.ts_left_workload = self.ts_left_workload
	    .checked_sub(processed_workload)
	    .ok_or(format!(
		"ERROR at task_process_workload: The processed_workload ({}) is greater than what is left to be processed ({}) in Task ({}).",
		processed_workload,
		self.ts_left_workload,
		self.ts_id,
	    ))?;
	debug!(
	    "Task {}: Processing... Current workload ({}) ({:?})",
	    self.ts_id,
	    self.ts_left_workload,
	    thread::current().id(),
	);

	self.ts_mem_indexer += processed_workload;

	Ok(self.ts_left_workload)
    }

    /// Returns Task's `ts_left_workload`
    /// # Returns
    ///
    /// * `ts_left_workload` - Task's current amount of workload to still be processed.
    pub fn task_left_workload(&self) -> u64
    {
	self.ts_left_workload
    }
    
    /// Returns Task's `ts_current_exectime`
    /// # Returns
    ///
    /// * `ts_current_exectime` - Task's current amount of workload processed.
    pub fn task_current_exectime(&self) -> u64
    {
	self.ts_current_exectime
    }

    pub fn task_set_current_exectime(&mut self, processed_workload: u64) {
	self.ts_current_exectime = processed_workload;
    }
    
    /// Returns Task's `ts_id`
    ///
    /// # Returns
    ///
    /// * `ts_id` - Task's unique identifier.
    pub fn task_ts_id(&self) -> u8
    {
	self.ts_id
    }

    /// Returns Task's `ts_mem_indexer`
    ///
    /// # Returns
    ///
    /// * `ts_mem_indexer` - Task's memory indexer.
    pub fn task_mem_indexer(&self) -> u64
    {
	self.ts_mem_indexer
    }

    pub fn task_get_cr3(&self) -> u64
    {
	self.ts_cr3
    }

    pub fn task_get_current_vcpu_id(&self) -> usize {
	self.ts_current_vcpu_id
    }

    pub fn task_set_current_vcpu_id(&mut self, id: usize) {
	self.ts_current_vcpu_id = id
    }

    pub fn task_get_last_vcpu_id(&self) -> usize {
	self.ts_last_vcpu_id
    }

    pub fn task_set_last_vcpu_id(&mut self, id: usize) {
	self.ts_last_vcpu_id = id
    }
    
    /// Returns Task's mem. `addresses` as a slice
    ///
    /// # Arguments
    /// * `initial_position` - The initial position of the slice
    /// * `final_position`   - The final position of the slice
    ///
    /// # Returns
    /// * `task_addresses[initial_position..final_position]` - Task's addreses from `initial` to `final`
    pub fn task_addresses(&mut self, initial_position: usize, final_position: usize) -> Result<&mut [Addr]>
    {
	let total = self.ts_mem_addresses.len();
	// Sanity Check
	if ( initial_position > final_position) || ( final_position > total ) {
	    return Err(anyhow::anyhow!(
		"Slice range [{}, {}) is out of bounds (len = {})",
		initial_position,
		final_position,
		total
	    ))
	}
	Ok(&mut self.ts_mem_addresses[initial_position..final_position])
    }

    pub fn update_hotness(&mut self, index: usize, n: usize) {
	*self.ts_cache_acesses.entry(index).or_insert(0) += 1;

	if self.ts_recent_acesses.len() == n {
	    self.ts_recent_acesses.pop_front();
	}
	self.ts_recent_acesses.push_back(index);
    }

    pub fn get_hotness_global(&self) -> &HashMap<usize, u64> {
	&self.ts_cache_acesses
    }


    pub fn get_hotness_recent(&self) -> HashMap<usize, u64> {
	let mut counts = HashMap::new();

	for &i in &self.ts_recent_acesses {
	    *counts.entry(i).or_insert(0) += 1;
	}

	counts
    }

}
