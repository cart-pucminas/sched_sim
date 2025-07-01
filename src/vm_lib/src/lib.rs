//==================================================================================================
// Configuration
//==================================================================================================
#![deny(clippy::all)]

//==================================================================================================
// Imports
//==================================================================================================
use ::anyhow::Result;
use task_lib::Task;
use vcpu_lib::VCPU;

use std::{
    time::Instant,
    collections::VecDeque,
    sync::{
	atomic::{
	    AtomicUsize,
	    Ordering,
	},
	Arc,
	// Mutex,
    },
    thread,
};
use tokio::{
    sync::{
	mpsc,
	oneshot,
	Mutex	    
    },
    time::{
	sleep,
	Duration,
	timeout,
    },
};

use log::debug;

//==================================================================================================
// Enum
//==================================================================================================
enum Scheduler {
    Fifo {
	current: usize,
    },
    RoundRobin {
	current: usize,
    },
}

enum Mapper {
    Fifo,
}

//==================================================================================================
// Aliases
//==================================================================================================
type TaskMessage = (Task, oneshot::Sender<Task>);

//==================================================================================================
// Structures
//==================================================================================================
/// Represents a virtual machine (VM) in the simulation.
#[derive(Clone)]
pub struct VM
{
    /// `VM` unique identifier (private field)
    vm_id			: u8,
    /// Pool of `VCPUs` that VM has (private field)
    vm_vcpus			: Vec<Arc<Mutex<VCPU>>>,
    
    vm_total_task_count		: Arc<AtomicUsize>,
    
    pub contention_time_nanos	: Arc<AtomicUsize>,
    
    /// VCPU's senders (mpsc) (private field)
    vcpu_senders		: Arc<Vec<mpsc::Sender<TaskMessage>>>,
    /// List of finished `Task`s inside the `VM` (private field)
    vm_tasks_finished		: Arc<Mutex<VecDeque<Task>>>,
    /// List of ready `Task`s inside the `VM` (private field)
    vm_tasks_ready		: Arc<Mutex<VecDeque<Task>>>,
    /// List of running `Task`s inside the `VM` (private field)
    vm_tasks_running		: Arc<Mutex<VecDeque<Task>>>,
    /// `VM`'s Task-vCPU Scheduler (private field)
    scheduler			: Arc<Mutex<Scheduler>>,
    /// `VM`'s Task-vCPU Mapper (private field)
    mapper			: Arc<Mutex<Mapper>>,
    /// `VM`'s Quantum, if preemptive (private field)
    quantum			: u64,
    /// If current scheduling strategy is preemptive (private field)
    preemptive			: bool,
}

//==================================================================================================
// Implementations
//==================================================================================================
impl Scheduler {
    // TODO: Add a strategy that analyzes which vCPU currently has less workload in it.
    pub async fn select_vcpu(&mut self, vm: &VM) -> usize {
	match self {
	    Scheduler::Fifo { current } => {
		debug!(
		    "[VM {}] Selected vCPU ({}) - FIFO ({:?})",
		    vm.clone().vm_get_id(),
		    *current,
		    thread::current().id(),
		);
		let selected = *current;
		*current = (*current + 1) % vm.vm_vcpus_senders().len();
		
		selected
	    },
	    Scheduler::RoundRobin { current } => {
		debug!(
		    "[VM {}] Selected vCPU ({}) - RoundRobin ({:?})",
		    vm.clone().vm_get_id(),
		    *current,
		    thread::current().id(),
		);
		let selected = *current;
		*current = (*current + 1) % vm.vm_vcpus_senders().len();
		
		selected
	    },
	}
    }

    pub fn is_preemptive(&self) -> bool {
	match self {
	    Scheduler::Fifo { .. } => false,
	    Scheduler::RoundRobin { .. } => true,
	}}
}

impl Mapper {
    pub async fn select_task(&mut self, vm: &VM) -> Option<Task> {
	match self {
	    Mapper::Fifo => {
		let start = Instant::now();
		let task_opt = vm.clone().vm_tasks_ready().lock().await.pop_front();
		vm.contention_time_nanos.fetch_add(start.elapsed().as_nanos() as usize, Ordering::Relaxed);
		    match &task_opt {
		    Some(task) => {
			debug!(
			    "[VM {}] Selected Task ({}) - FIFO ({:?})",
			    vm.clone().vm_get_id(),
			    task.task_ts_id(),
			    thread::current().id(),
			);
		    }
		    None => {
			debug!(
			    "[VM {}] No task selected - FIFO ({:?})",
			    vm.clone().vm_get_id(),
			    thread::current().id(),
			);	    
		    }

		}
		task_opt
	    },
	}
    }
       
}

impl VM
{
    /// Creates a new `VM` instance.
    ///
    /// # Arguments
    /// 
    /// * `id`        - The unique of a `VM`.
    /// * `num_vcpus` - Total number of vCPUs that `VM` has.
    /// * `scheduler` - `VM`'s Task-vCPU scheduler strategy.
    /// * `mapper`    - `VM`'s Task-vCPU mapping strategy.
    /// * `num_tasks` - Total number of Tasks
    ///
    /// # Returns
    ///
    /// * A new `VM` instance with an empty task list.
    pub async fn new(
	id: u8,
	num_vcpus: usize,
	desired_scheduler: &str,
	desired_mapper: &str,
	num_tasks: Arc<AtomicUsize>,
    ) -> Result<Self> {
	debug!(
	    "Creating VM ({}) with {} vcpus",
	    id,
	    num_vcpus
	);
	let vm_scheduler = match desired_scheduler {
	    "FIFO" => {
		Arc::new(Mutex::new(Scheduler::Fifo { current: 0 }))
	    },
	    "RoundRobin" => {
		Arc::new(Mutex::new(Scheduler::RoundRobin { current: 0 }))
	    },
	    &_ => {
		return Err(anyhow::anyhow!("invalid scheduler"));
	    },
	};
	
	let start = Instant::now();
	let preemptive: bool = {
	    let preemptive_guard = vm_scheduler.lock().await;
	    preemptive_guard.is_preemptive()
	};
	let mut contention_time_nanos = Arc::new(AtomicUsize::new(0));
	contention_time_nanos.fetch_add(start.elapsed().as_nanos() as usize, Ordering::Relaxed);
	
	let vm_mapper = match desired_mapper {
	    "FIFO" => {
		Arc::new(Mutex::new(Mapper::Fifo))
	    },
	    &_ => {
		return Err(anyhow::anyhow!("invalid mapper"));
	    },
	};

	let mut senders = Vec::new();
	
	let mut vcpus: Vec<Arc<Mutex<VCPU>>> = Vec::with_capacity(num_vcpus);
	for i in 0..num_vcpus {
	    let (transmitter, receiver) = mpsc::channel(100);
	    senders.push(transmitter);
	    
	    let vcpu = Arc::new(Mutex::new(VCPU::new(id + (i as u8), id)?));
	    let vcpu_clone = Arc::clone(&vcpu);

	    std::thread::spawn(move || {
		let runtime = tokio::runtime::Builder::new_current_thread()
		    .enable_all()
		    .build()
		    .unwrap();

		    runtime.block_on(async move {
			VCPU::vcpu_loop(vcpu_clone, receiver).await;
		    });
	    });
	    
	    vcpus.push(vcpu);
	}

	Ok(Self {
	       vm_id: id,
	       vm_vcpus: vcpus,
	       vm_total_task_count: num_tasks,
	       contention_time_nanos,
	       vcpu_senders: Arc::new(senders),
	       vm_tasks_finished: Arc::new(Mutex::new(VecDeque::new())),
               vm_tasks_ready: Arc::new(Mutex::new(VecDeque::new())),
	       vm_tasks_running: Arc::new(Mutex::new(VecDeque::new())),
	       scheduler: vm_scheduler,
	       mapper: vm_mapper,
	       quantum: 100,
	       preemptive,		
	})
    }

    pub async fn completion_loop(
	self: Arc<VM>,
	mut completion_receiver: mpsc::Receiver<(Task, u64)>
    ) {
	while let Some((mut task, execution_time)) = completion_receiver.recv().await {
	    let start = Instant::now();
	    self.vm_tasks_running.lock().await.retain(|t| t.task_ts_id() != task.task_ts_id());
	    self.contention_time_nanos.fetch_add(start.elapsed().as_nanos() as usize, Ordering::Relaxed);
	    
	    task.task_process_workload(execution_time);
	    
	    // Still has work to do (Occurs only when there is preeption)
	    if task.task_left_workload() > 0 {
		let start = Instant::now();
		self.vm_tasks_ready.lock().await.push_back(task);
		self.contention_time_nanos.fetch_add(start.elapsed().as_nanos() as usize, Ordering::Relaxed);
	    } else {
		let start = Instant::now();
		self.vm_total_task_count.fetch_sub(1, Ordering::SeqCst);
		debug!("Task {} finished.", task.task_ts_id());
		self.vm_tasks_finished.lock().await.push_back(task);
		self.contention_time_nanos.fetch_add(start.elapsed().as_nanos() as usize, Ordering::Relaxed);
	    }
	    
	}
    }
    
    pub async fn schedule_loop(
	self: Arc<VM>,
	completion_sender: mpsc::Sender<(Task, u64)>
    ) {
	loop {

	    // If no Task left, all done.
	    if self.vm_total_task_count.load(Ordering::SeqCst) == 0 {
		break;
	    }

	    let start = Instant::now();
	    let selected_task_opt = self.mapper.lock().await.select_task(&self).await;
	    self.contention_time_nanos.fetch_add(start.elapsed().as_nanos() as usize, Ordering::Relaxed);

	    if let Some(mut task) = selected_task_opt {
		let start = Instant::now();
		self.vm_tasks_running.lock().await.push_back(task.clone());
		self.contention_time_nanos.fetch_add(start.elapsed().as_nanos() as usize, Ordering::Relaxed);

		let start = Instant::now();
		let selected_vcpu = self.scheduler.lock().await.select_vcpu(&self).await;
		self.contention_time_nanos.fetch_add(start.elapsed().as_nanos() as usize, Ordering::Relaxed);

		let mut execution_time = task.task_left_workload();
		if self.preemptive {
		    execution_time = execution_time.min(self.quantum);
		}
		task.task_set_current_workload(execution_time);

		let (response_transmitter, response_receiver) = oneshot::channel();
		if self.vcpu_senders[selected_vcpu].send((task.clone(), response_transmitter)).await.is_ok() {
		    let completion_sender = completion_sender.clone();
		    tokio::spawn(async move {
			if let Ok(returned_task) = response_receiver.await {
			    let _ = completion_sender.send((returned_task, execution_time)).await;
			}
		    });
		    
		    // If failed to send task, readd task to be sent elsewhere
		} else {
		    let start = Instant::now();
		    self.vm_tasks_ready.lock().await.push_front(task);
		    self.contention_time_nanos.fetch_add(start.elapsed().as_nanos() as usize, Ordering::Relaxed);
		}
	    } else {
		tokio::time::sleep(Duration::from_millis(100)).await;
	    }
		
	}
	
	debug!("[VM {}] Finished processing.", self.vm_get_id());
    }
    
    /// Returns (if exists) the index of a `Task` in `vm_tasks_ready` with specified `task_id`
    /// 
    /// # Arguments
    ///
    /// * `task_id` - A Task's `ts_id`
    ///
    /// # Returns
    /// 
    /// * `Ok(usize)`   - Task's index in `vm_tasks` if `task_id` is found.
    /// * `Err(String)` - if `vm_tasks` is empty or no Task with `task_id` is found.
    ///
    /// # Panics
    /// 
    /// * This function does **not** panic. It returns a `Result` instead.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let element_index: usize = self.index_of(id).await;
    /// println!("index {}", element_index)
    /// ```
    fn index_of(task_queue: VecDeque<Task> ,task_id: u8) -> Result<usize, String> {
	/* Sanity check. */
	if task_queue.is_empty() {
	    return Err("ERROR at VM's index_of: VM has no Task currently".to_string());
	}
	
	task_queue.iter()
	    .position(|task| task.task_ts_id() == task_id)
	    .ok_or_else(|| format!("ERROR at VM's index_of: The specified task_id ({}) was not found.",
				   task_id))
    }
     
    /// Adds a new `Task` instance into VM's `tasks_ready`
    ///
    /// # Arguments
    /// 
    /// * `task` - A VM's task.
    pub async fn vm_task_ready_add(&mut self, task: Task) {
	debug!("[VM {:?}] Adding task {:?}", self.vm_id, task.task_ts_id());
	let start = Instant::now();
	self.vm_tasks_ready.lock().await.push_back(task);
	self.contention_time_nanos.fetch_add(start.elapsed().as_nanos() as usize, Ordering::Relaxed);
    }

    fn vm_tasks_ready(self) -> Arc<Mutex<VecDeque<Task>>> {
	self.vm_tasks_ready
    }
   
    fn vm_vcpus_senders(&self) -> Arc<Vec<mpsc::Sender<TaskMessage>>>{ 
	self.vcpu_senders.clone()
    }
    
    fn vm_change_preemption(&mut self, preempt: bool) {
	self.preemptive = preempt
    }
    
    fn vm_is_preemptive(self) -> bool {
	self.preemptive
    }

    fn vm_get_id(&self) -> u8 {
	self.vm_id
    }
}
