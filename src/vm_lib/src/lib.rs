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
use cache_lib::Cache;
use ram_lib::RAM;

use std::{
    time::Instant,
    collections::{
	VecDeque,
	HashMap,
	HashSet,
    },
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
    Jaccard {},
    CDF {},
    Balanced {},
    GraphBased {},
    
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
    pub vm_vcpus		: Vec<Arc<Mutex<VCPU>>>,
    pub l2_caches               : Vec<Arc<Mutex<Cache>>>,
    pub l3_cache                : Arc<Mutex<Cache>>,
    
    vm_total_task_count		: Arc<AtomicUsize>,
    
    pub contention_time_nanos	: Arc<AtomicUsize>,
    pub schedule_time           : Arc<AtomicUsize>,
    
    /// VCPU's senders (mpsc) (private field)
    vcpu_senders		: Arc<Vec<mpsc::Sender<TaskMessage>>>,
    /// List of finished `Task`s inside the `VM` (private field)
    pub vm_tasks_finished		: Arc<Mutex<VecDeque<Task>>>, // This one might not be needed to be ArcMutex
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

    ram                         : Arc<Mutex<RAM>>,

    pub vcpu_groups             : HashMap<usize, Vec<usize>>,

    pub task_conflict_dag       : Arc<Mutex<HashMap<usize, HashMap<usize, f64>>>>,
}

//==================================================================================================
// Implementations
//==================================================================================================
impl Scheduler {
    // TODO: Add a strategy that analyzes which vCPU currently has less workload in it.
    pub async fn select_vcpu(&mut self, vm: &VM, task: &mut Task) -> usize {
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
	    Scheduler::Jaccard { } => {

		// TODO: Remember to change "recent" size (in Task) to Quantum's Size
		//                                                  (Or some percentage or Quantum's size)
		let recent = task.get_hotness_recent();
		let global = task.get_hotness_global();

		if !Self::hotness_drift_significant(global, &recent, 0.2) {
		    let selected = task.task_get_last_vcpu_id();

		    debug!(
			"[VM {}] Selected vCPU ({}) - Jaccard ({:?})",
			vm.clone().vm_get_id(),
			selected,
			thread::current().id(),
		    );
		    
		    task.task_set_current_vcpu_id(selected);
		    return selected;
		}
		
		let task_indices: HashSet<usize> = recent.keys().copied().collect();
		let group_indices = vm.get_hot_indices_by_group().await;

		// If there is any group that doesn't have tasks in it, we select that group
		// Otherwise, the selection is based in the distance of jaccard index, i.e.,
		// the group that the Tasks in it has the least cache conflict with the task to be Scheduled
		let selected_group_id = if let Some((group_id, _)) = vm.vcpu_groups.iter()
		    .find(|(group_id, _)| !group_indices.contains_key(group_id))
		{
		    *group_id
		} else {
		    let mut best_group = None;
		    let mut best_distance = f64::INFINITY;

		    for (group_id, indices) in group_indices.iter() {
			let distance = 1.0 - Self::jaccard_index(&task_indices, indices);
			if distance < best_distance {
			    best_distance = distance;
			    best_group = Some(*group_id);
			}
		    }

		    best_group.expect("Schedule::Jaccard -> No group found.")
		};

		let group = &vm.vcpu_groups[&selected_group_id];
		let group_vcpus: HashSet<usize> = group.iter().copied().collect();
		let vcpu_task_counts = vm.get_task_count_per_vcpu_group(&group_vcpus).await;

		let selected = group
		    .iter()
		    .min_by_key(|vcpu_id| {
			*vcpu_task_counts.get(vcpu_id).unwrap_or(&0)
		    })
		    .copied()
		    .expect("Scheduler::Jaccard -> Expected to find a vCPU id");

		task.task_set_current_vcpu_id(selected);
		debug!(
		    "[VM {}] Selected vCPU ({}) - Jaccard ({:?})",
		    vm.clone().vm_get_id(),
		    selected,
		    thread::current().id(),
		);
		selected
	    },
	    Scheduler::CDF { } => {
		let recent = task.get_hotness_recent();
		let global = task.get_hotness_global();

		if !Self::hotness_drift_significant(global, &recent, 0.2) {
		    let selected = task.task_get_last_vcpu_id();

		    debug!(
			"[VM {}] Selected vCPU ({}) - CDF ({:?})",
			vm.clone().vm_get_id(),
			selected,
			thread::current().id(),
		    );
		    
		    task.task_set_current_vcpu_id(selected);
		    return selected;
		}

		let task_cdf = Self::compute_cdf(&recent);
		let task_percentiles = Self::extract_percentiles(&task_cdf, &[0.25, 0.5, 0.75]);

		let group_indices = vm.get_hot_indices_by_group().await;

		let mut good_groups = Vec::new();
		let tol = 0.2;

		for (group_id, hotness_values) in group_indices.iter() {
		    if hotness_values.is_empty() {
			good_groups.push(*group_id);
			continue;
		    }

		    let group_cdf = Self::compute_cdf_from_values(
			&hotness_values.iter().map(|&x| x as u64).collect::<Vec<_>>()
		    );

		    if group_cdf.is_empty() {
			good_groups.push(*group_id);
			continue;
		    }
		    
		    let group_percentiles = Self::extract_percentiles(&group_cdf, &[0.25, 0.5, 0.75]);

		    let mut valid_count = 0;
		    for i in 0..task_percentiles.len() {
			let lower = group_percentiles[i] * (1.0 - tol);
			let upper = group_percentiles[i] * (1.0 + tol);
			if task_percentiles[i] >= lower && task_percentiles[i] <= upper {
			    valid_count += 1;
			}
		    }

		    if valid_count >= 2 {
			good_groups.push(*group_id);
		    }
		}

		let candidate_groups: Vec<usize> = if !good_groups.is_empty() {
		    good_groups.iter().copied().collect()
		} else {
		    vm.vcpu_groups.keys().copied().collect()
		};

		let mut group_loads: Vec<(usize, usize)> = Vec::with_capacity(candidate_groups.len());
		for gid in candidate_groups.into_iter() {
		    let vcpus: HashSet<usize> = vm.vcpu_groups.get(&gid)
			.expect("group id must exist")
			.iter()
			.copied()
			.collect();

		    let counts_map = vm.get_task_count_per_vcpu_group(&vcpus).await;
		    let load: usize = counts_map.values().sum();
		    group_loads.push((gid, load));
			
		}

		let (selected_group_id, _) = group_loads
		    .into_iter()
		    .min_by_key(|(_, load)| *load)			
		    .expect("Schedule::CDF ther must be atleast one candidate group");

		let selected_vcpus_slice = &vm.vcpu_groups[&selected_group_id];
		let group_vcpus: HashSet<usize> = selected_vcpus_slice.iter().copied().collect();
		let vcpu_task_count = vm.get_task_count_per_vcpu_group(&group_vcpus).await;
		

		let selected = selected_vcpus_slice
		    .iter()
		    .min_by_key(|vcpu_id| {
			vcpu_task_count.get(vcpu_id).copied().unwrap_or(0)
		    })
		    .copied()
		    .expect("Scheduler::CDF -> Expected to find a vCPU id");

		task.task_set_current_vcpu_id(selected);

		debug!(
		    "[VM {}] Selected vCPU ({}) - CDF ({:?})",
		    vm.clone().vm_get_id(),
		    selected,
		    thread::current().id(),
		);

		selected
		    
			  
	    },
	    Scheduler::GraphBased { } => {

		let recent = task.get_hotness_recent();
		let global = task.get_hotness_global();

		if !Self::hotness_drift_significant(global, &recent, 0.2) {
		    let selected = task.task_get_last_vcpu_id();

		    debug!(
			"[VM {}] Selected vCPU ({}) - Graph ({:?})",
			vm.clone().vm_get_id(),
			selected,
			thread::current().id(),
		    );
		    
		    task.task_set_current_vcpu_id(selected);
		    return selected;
		}

		let task_id = task.ts_id;

		let mut dag = vm.task_conflict_dag.lock().await;
		let mut tasks_map = HashMap::new();

		let running_tasks = vm.vm_tasks_running.lock().await;

		// Adding task to DAG and getting weights
		// weights = frequence(a) * frequence(b) if key(a) == key(b)
		for t in running_tasks.iter() {
		    if t.ts_id == task_id {
			continue;
		    }

		    let weight = Self::conflict_weight(&task, t);
		    if weight > 0.0 {
			tasks_map.insert(t.ts_id as usize, weight);
			dag.entry(t.ts_id.into()).or_default().insert(task_id.into(), weight);
		    }
		}
		dag.insert(task_id.into(), tasks_map);

		let mut group_conflicts: HashMap<usize, f64> = HashMap::new();

		// For each grup, we get the amount of accum. conflict with the tasks
		// currently running in that group.
		for (&group_id, vcpus) in &vm.vcpu_groups {
		    let mut total_conflict = 0.0;
		    for t in running_tasks.iter() {
			if t.ts_id == task_id { continue }
			let vcpu_id = t.task_get_current_vcpu_id();

			// Identifies if vCPU is in current group
			if vcpus.contains(&vcpu_id) {

			    // If conflict found, sum it up
			    if let Some(conflicts) = dag.get(&task_id.into()) {
				total_conflict += conflicts.get(&t.ts_id.into()).copied().unwrap_or(0.0);
			    }
			    
			}
		    }
		    
		    group_conflicts.insert(group_id, total_conflict);
		}
		drop(running_tasks);

		// Choosing all groups with the least amount of accum. conflict
		let min_conflict = group_conflicts.values().copied().fold(f64::INFINITY, f64::min);
		let candidate_groups: Vec<usize> = group_conflicts.iter()
		    .filter_map(|(&gid, &conflict)|
				if conflict == min_conflict {
				    Some(gid)
				}
				else {
				    None
				}
		    ).collect();

		// If necessary, breaking the tie selecting the group with the least amount of workload
		let mut group_loads: Vec<(usize, usize)> = Vec::with_capacity(candidate_groups.len());
		for gid in candidate_groups.into_iter() {
		    let vcpus: HashSet<usize> = vm.vcpu_groups.get(&gid)
			.expect("group id must exist")
			.iter()
			.copied()
			.collect();

		    let counts_map = vm.get_task_count_per_vcpu_group(&vcpus).await;
		    let load: usize = counts_map.values().sum();
		    group_loads.push((gid, load));
			
		}
		
		let (selected_group_id, _) = group_loads
		    .into_iter()
		    .min_by_key(|(_, load)| *load)
		    .expect("Schedule::GraphBased -> Expected at least one candidate group");


		let selected_vcpus_slice = &vm.vcpu_groups[&selected_group_id];
		let group_vcpus: HashSet<usize> = selected_vcpus_slice.iter().copied().collect();
		let vcpu_task_count = vm.get_task_count_per_vcpu_group(&group_vcpus).await;

		let selected = selected_vcpus_slice
		    .iter()
		    .min_by_key(|vcpu_id| {
			vcpu_task_count.get(vcpu_id).copied().unwrap_or(0)
		    })
		    .copied()
		    .expect("Scheduler::GraphBased -> Expected to find a vCPU id");

		task.task_set_current_vcpu_id(selected);
		
		debug!(
		    "[VM {}] Selected vCPU ({}) - Graph ({:?})",
		    vm.clone().vm_get_id(),
		    selected,
		    thread::current().id(),
		);

		selected
		
	    }
	    Scheduler::Balanced { } => {
		let tasks_running = vm.vm_tasks_running.lock().await;
		let cache_groups = &vm.vcpu_groups;

		let mut group_loads: HashMap<usize, usize> = HashMap::new();

		for (group_id, vcpus) in cache_groups.iter() {
		    let mut total = 0;
		    for vcpu_id in vcpus {
			let count = tasks_running
			    .iter()
			    .filter(|t| t.task_get_current_vcpu_id() == *vcpu_id)
			    .count();
			total += count;
		    }
		    group_loads.insert(*group_id, total);
		}


		let (&best_group_id, _) = group_loads.iter().min_by_key(|(_, load)| *load).unwrap();

		let mut best_vcpu = None;
		let mut min_tasks = usize::MAX;

		for &vcpu_id in &cache_groups[&best_group_id] {
		    let count = tasks_running
			.iter()
			.filter(|t| t.task_get_current_vcpu_id() == vcpu_id)
			.count();
		    if count < min_tasks {
			best_vcpu = Some(vcpu_id);
			min_tasks = count;
		    }
		}
		let selected = best_vcpu.expect("Schedule::Balanced => No vCPU selected. ERROR");
		debug!(
		    "[VM {}] Selected vCPU ({}) - Jaccard ({:?})",
		    vm.clone().vm_get_id(),
		    selected,
		    thread::current().id(),
		);
		task.task_set_current_vcpu_id(selected);
		selected
	    },
	}
    }

    pub fn is_preemptive(&self) -> bool {
	match self {
	    Scheduler::Fifo { .. } => false,
	    Scheduler::RoundRobin { .. } => true,
	    Scheduler::Jaccard { .. } => true,
	    Scheduler::Balanced { .. } => true,
	    Scheduler::CDF { .. } => true,
	    Scheduler::GraphBased { .. } => true,
	}
    }


    // JACCARD AUXILIARY FUNCTIONS
    fn jaccard_index(a: &HashSet<usize>, b: &HashSet<usize>) -> f64 {
	let intersection_size = a.intersection(b).count();
	let union_size = a.union(b).count();

	if union_size == 0 {
	    0.0
	} else {
	    intersection_size as f64 / union_size as f64
	}
    }

    // Weighted Jaccard(A, B) = Σ min(A[i], B[i]) / Σ max(A[i], B[i])
    fn weighted_jaccard_index(a: &HashMap<usize, u64>, b: &HashMap<usize, u64>) -> f64 {
	let mut intersection_sum = 0usize;
	let mut union_sum = 0usize;

	let all_keys: std::collections::HashSet<_> = a.keys().chain(b.keys()).collect();

	for key in all_keys {
	    let freq_a = a.get(key).copied().unwrap_or(0) as usize;
	    let freq_b = b.get(key).copied().unwrap_or(0) as usize;

	    intersection_sum += freq_a.min(freq_b);
	    union_sum += freq_a.max(freq_b);
	}

	if union_sum == 0 {
	    0.0
	} else {
	    intersection_sum as f64 / union_sum as f64
	}
    }

    
    fn hotness_drift_significant(
	global: &HashMap<usize, u64>,
	recent: &HashMap<usize, u64>,
	threshold: f64
    ) -> bool {
	// If Task's hotness didn't change much since last iterations
	// It implies that a warm start is more advantageous
	// Because of that, if "similarity" of recent_hotness and global_hotness
	// we send the task back to the same vCPU
	let distance = 1.0 - Self::weighted_jaccard_index(global, recent);
	distance > threshold
    }

    // CDF FUNCTIONS
    fn compute_cdf(values: &HashMap<usize, u64>) -> Vec<f64> {
	let mut vals: Vec<u64> = values.values().copied().collect();
	vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
	let n = vals.len();
	vals.into_iter()
	    .enumerate()
	    .map(|(i, v)| (i + 1) as f64 / n as f64)
	    .collect()
	
    }
    
    fn compute_cdf_from_values(values: &[u64]) -> Vec<f64> {
	let mut vals: Vec<u64> = values.to_vec();
	vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
	let n = vals.len();
	vals.into_iter()
	    .enumerate()
	    .map(|(i, v)| (i + 1) as f64 / n as f64)
	    .collect()	
    }

    fn extract_percentiles(cdf: &[f64], percentiles: &[f64]) -> Vec<f64> {
	if cdf.is_empty() {
	    return Vec::new();
	}
	percentiles.iter().map(|&p| {
	    let idx = ((cdf.len() as f64) * p).ceil() as usize;
	    let idx = idx.saturating_sub(1);
	    cdf[idx.min(cdf.len() - 1)]
	}).collect()
    }
    
    // GraphBased functions
    fn conflict_weight(task_a: &Task, task_b: &Task) -> f64 {
	let mut weight = 0.0;
	let hotness_a = task_a.get_hotness_recent();
	let hotness_b = task_b.get_hotness_recent();

	for (&idx, &freq_a) in hotness_a.iter() {
	    if let Some(&freq_b) = hotness_b.get(&idx) {
		weight += (freq_a as f64) * (freq_b as f64);
	    }
	}
	
	weight
    }
    
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
			// debug!(
			//     "[VM {}] No task selected - FIFO ({:?})",
			//     vm.clone().vm_get_id(),
			//     thread::current().id(),
			// );	    
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
	max_num_tasks: usize,
	ram: Arc<Mutex<RAM>>,
    ) -> Result<Self> {
	debug!(
	    "Creating VM ({}) with {} vcpus",
	    id,
	    num_vcpus,
	);
	let vm_scheduler = match desired_scheduler {
	    "FIFO" => {
		Arc::new(Mutex::new(Scheduler::Fifo { current: 0 }))
	    },
	    "RoundRobin" => {
		Arc::new(Mutex::new(Scheduler::RoundRobin { current: 0 }))
	    },
	    "Jaccard" => {
		Arc::new(Mutex::new(Scheduler::Jaccard { } ))
	    },
	    "CDF" => {
		Arc::new(Mutex::new(Scheduler::CDF { } ))
	    }
	    "Balanced" => {
		Arc::new(Mutex::new(Scheduler::Balanced { } ))
	    },
	    "Graph" => {
		Arc::new(Mutex::new(Scheduler::GraphBased {} ))
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
	let mut schedule_time = Arc::new(AtomicUsize::new(0));
	
	let vm_mapper = match desired_mapper {
	    "FIFO" => {
		Arc::new(Mutex::new(Mapper::Fifo))
	    },
	    &_ => {
		return Err(anyhow::anyhow!("invalid mapper"));
	    },
	};

	let mut senders = Vec::new();

	let l3_shared = Arc::new(Mutex::new(Cache::new(0, "L3", 8*1024*1024, 4, 4, "Lru")?));

	let mut l2_groups: Vec<Arc<Mutex<Cache>>> = Vec::new();
	let num_l2_groups = (num_vcpus + 3) / 4;

	for g in 0..num_l2_groups {
	    let l2 = Arc::new(Mutex::new(Cache::new(
		g as u8,
		"L2",
		2*1024*1024,
		4,
		4,
		"Lru",
	    )?));
	    l2_groups.push(l2);
	}

	let mut vcpus: Vec<Arc<Mutex<VCPU>>> = Vec::with_capacity(num_vcpus);
	let mut vcpu_l2_groups = HashMap::new();
	for i in 0..num_vcpus {
	    let (transmitter, receiver) = mpsc::channel(max_num_tasks);
	    senders.push(transmitter);

	    // TODO: Currently caches must have the same block-size
	    //       In the future, it shall be needy nice to change it
	    //       (Use cache_extract_sub_block)
	    let vcpu_l1 = Cache::new(i as u8, "L1", 512*1024 , 4, 4, "Lru")?;

	    // Watch out if using more than one VM, "i" might cause some troubles
	    let l2_index = i / 4;
	    let l2_shared = l2_groups[l2_index].clone();
	    let l3_global = l3_shared.clone();

	    vcpu_l2_groups
		.entry(l2_index)
		.or_insert_with(Vec::new)
		.push(i);
	    
	    let vcpu = Arc::new(Mutex::new(VCPU::new(
	        i as u8,
		id,
		vcpu_l1,
		l2_shared,
		l3_global,
		ram.clone()
	    )?));
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
	       l2_caches: l2_groups,
	       l3_cache: l3_shared,
	       vm_total_task_count: num_tasks,
	       contention_time_nanos,
	       schedule_time,
	       vcpu_senders: Arc::new(senders),
	       vm_tasks_finished: Arc::new(Mutex::new(VecDeque::new())),
               vm_tasks_ready: Arc::new(Mutex::new(VecDeque::new())),
	       vm_tasks_running: Arc::new(Mutex::new(VecDeque::new())),
	       scheduler: vm_scheduler,
	       mapper: vm_mapper,
	       quantum: 300,
	       preemptive,
	       ram,
               vcpu_groups: vcpu_l2_groups,
	       task_conflict_dag: Arc::new(Mutex::new(HashMap::new())),
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

	    // This might not be as accurate, probably might change in the future to the real time spent
	    task.accumulated_exec_time += Duration::from_nanos(execution_time);
	    
	    // Still has work to do (Occurs only when there is preemption)
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

		// Setting Task's time metrics
		if task.start_time.is_none() {
		    let now = Instant::now();
		    task.start_time = Some(now);
		}

		let start = Instant::now();
		let selected_vcpu = self.scheduler.lock().await.select_vcpu(&self, &mut task).await;
		let elapsed = start.elapsed().as_nanos() as usize;
		self.contention_time_nanos.fetch_add(elapsed, Ordering::Relaxed);
		self.schedule_time.fetch_add(elapsed, Ordering::Relaxed);

		let mut execution_time = task.task_left_workload();
		if self.preemptive {
		    execution_time = execution_time.min(self.quantum);
		}

		task.task_set_current_exectime(execution_time);
		
		let start = Instant::now();
		self.vm_tasks_running.lock().await.push_back(task.clone());
		self.contention_time_nanos.fetch_add(start.elapsed().as_nanos() as usize, Ordering::Relaxed);

		

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

    pub async fn get_hot_indices_by_group(&self) -> HashMap<usize, HashSet<usize>> {
	let mut group_accesses: HashMap<usize, HashSet<usize>> = HashMap::new();

	// Better to access "vm_task_running", since it only afects the "completion" thread
	// If we had to access the mpsc of each vCPU, it would generate more overhead
	let running = self.vm_tasks_running.lock().await;	
	for task in running.iter() {
	    let vcpu_id = task.task_get_current_vcpu_id() as usize;
	    if let Some((group_id, _)) = self.vcpu_groups.iter()
		.find(|(_, vcpus)| vcpus.contains(&vcpu_id))
	    {
		let entry = group_accesses.entry(*group_id).or_default();
		entry.extend(task.get_hotness_recent().keys().copied());
	    }
	}


	// TODO: Check if necessary -> Currently, in the way that it is, Jaccard index will bug with this uncommented
	// // Granting that every group is "filled"
	for (group_id, _) in self.vcpu_groups.iter() {
	    group_accesses.entry(*group_id).or_insert_with(HashSet::new);
	}
	
	group_accesses
    }

    pub async fn get_task_count_per_vcpu_group(
	&self,
	group_vcpus: &HashSet<usize>,
    ) -> HashMap<usize, usize> {
	let mut counts = HashMap::new();
	
	let running = self.vm_tasks_running.lock().await;
	for task in running.iter() {
	    let vcpu_id = task.task_get_current_vcpu_id();
	    if group_vcpus.contains(&vcpu_id) {
		*counts.entry(vcpu_id).or_insert(0) += 1;
	    }
	}

	counts
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

    pub async fn print_cache_stats(&self) {
	for vcpu_arc in &self.vm_vcpus {
	    let vcpu = vcpu_arc.lock().await;
	    println!("vCPU {} - L1: accesses={}, hits={}, misses={}",
                vcpu.vcpu_id(),
                vcpu.cache_l1.cache_accesses(),
                vcpu.cache_l1.cache_hits(),
                vcpu.cache_l1.cache_misses()
            );
	}

	
        for (i, l2_arc) in self.l2_caches.iter().enumerate() {
            let l2 = l2_arc.lock().await;
            println!("L2 (grupo {}): accesses={}, hits={}, misses={}",
                i,
                l2.cache_accesses(),
                l2.cache_hits(),
                l2.cache_misses()
            );
        }

        let l3 = self.l3_cache.lock().await;
        println!("L3 (global): accesses={}, hits={}, misses={}",
            l3.cache_accesses(),
            l3.cache_hits(),
            l3.cache_misses()
        );
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
