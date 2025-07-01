//==================================================================================================
// Configuration
//==================================================================================================
#![deny(clippy::all)]

//==================================================================================================
// Imports
//==================================================================================================
use log::debug;
use std::thread;

//==================================================================================================
// Structures
//==================================================================================================
/// Represents a Task to be scheduled in a Virtual Machine (VM) in the simulation.
#[derive(Clone)]
pub struct Task
{
    /// Task unique identifier (private field)
    ts_id		: u8,
    /// Task application name (private field)
    ts_name		: String,
    /// Task total workload to be processed (private field)
    ts_total_workload	: u64,
    /// Task workload left to be processed (private field)
    ts_left_workload	: u64,
    /// Task current workload to be processed (private field)
    ts_current_workload : u64,
    
}

//==================================================================================================
// Implementations
//==================================================================================================
impl Task
{
    /// Creates a new Task instance.
    ///
    /// # Arguments
    /// * `id`       - The unique identifier of a Task;
    /// * `name`     - The application name; (*MIGHT BE REMOVED IN A NEAR FUTURE*)
    /// * `workload` - The Task's total workload to be processed (Duration);
    ///
    /// # Returns
    /// * A new `Task` instance
    pub fn new(id: u8, name: &str, workload: u64) -> Self
    {
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
	    ts_current_workload: 0,
	} 
    }

    /// Represents the operation of processing a Task's workload. Basically, reduces Task's current_workload based in specified argument.
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
		"ERROR at task_process_workload: The processed_workload ({}) is greater than what is left to be processed ({}) in Task.",
		processed_workload,
		self.ts_left_workload
	    ))?;
	debug!(
	    "Task {}: Processing... Current workload ({}) ({:?})",
	    self.ts_id,
	    self.ts_left_workload,
	    thread::current().id(),
	);

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
    
    /// Returns Task's `ts_current_workload`
    /// # Returns
    ///
    /// * `ts_current_workload` - Task's current amount of workload processed.
    pub fn task_current_workload(&self) -> u64
    {
	self.ts_current_workload
    }

    pub fn task_set_current_workload(&mut self, processed_workload: u64) -> () {
	self.ts_current_workload = processed_workload;
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
}
