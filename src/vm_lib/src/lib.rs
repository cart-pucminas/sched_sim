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
    CDF {
	tol: f64,
    },
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
    pub vm_tasks_finished	: Arc<Mutex<VecDeque<Task>>>,
    /// List of ready `Task`s inside the `VM` (private field)
    vm_tasks_ready		: Arc<Mutex<VecDeque<Task>>>,
    /// List of running `Task`s inside the `VM` (private field)
    vm_tasks_running		: Arc<Mutex<VecDeque<Task>>>,
    /// `VM`'s Task-vCPU Scheduler (private field)
    scheduler			: Arc<Mutex<Scheduler>>, // Not sure if Arc<Mutex<>> is really necessary
    /// `VM`'s Task-vCPU Mapper (private field)
    mapper			: Arc<Mutex<Mapper>>,
    /// `VM`'s Quantum, if preemptive (private field)
    quantum			: u64,
    /// If current scheduling strategy is preemptive (private field)
    preemptive			: bool,

    ram                         : Arc<Mutex<RAM>>,

    pub tasks_per_group         : Arc<Mutex<HashMap<usize, Vec<usize>>>>,
    pub vcpus_per_group         : HashMap<usize, Vec<usize>>,
    pub vcpus_workload          : Arc<Mutex<Vec<u64>>>,
    pub group_hotness_agg       : Arc<Mutex<HashMap<usize, HashMap<usize, u64>>>>,

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

		let recent = task.get_hotness_recent();
		// let previous = task.get_hotness_previous();

		// if !Self::hotness_drift_significant(&previous, &recent, 0.8) {
		//     let selected = task.task_get_last_vcpu_id();

		//     debug!(
		// 	"[VM {}] Selected vCPU ({}) - Jaccard ({:?})",
		// 	vm.clone().vm_get_id(),
		// 	selected,
		// 	thread::current().id(),
		//     );
		    
		//     task.task_set_current_vcpu_id(selected);
		//     return selected;
		// }

		let mut group_hotness_sets: HashMap<usize, HashSet<usize>> = HashMap::new();
		
		for &group_id in vm.vcpus_per_group.keys() {
		    let hotness = vm.group_task_hotness(group_id).await;
		    let hot_indices: HashSet<usize> = hotness.keys().copied().collect();
		    group_hotness_sets.insert(group_id, hot_indices);
		}


		let task_indices: HashSet<usize> = task.get_hotness_recent().keys().copied().collect();
		let mut group_distances: Vec<(usize, f64)> = vm.vcpus_per_group.keys()
		    .map(|&gid| {
			let dist = 1.0 - Self::jaccard_index(&task_indices, &group_hotness_sets[&gid]);
			(gid, dist)
		    })
		    .collect();

		// In case that there are 2 or more groups that are tied with the maximum distance
		// we must choose the group that has the least number of tasks currently
		// this ensures workload balancement
		// The way that it is done here, if a group has no tasks, it will select it as the best
		// this also ensures workload balancement, even more, it streamlines task's accesses
		let max_distance = group_distances.iter()
		    .map(|(_, d)| *d)
		    .fold(f64::MIN, f64::max);
		let candidate_groups: Vec<usize> = group_distances.iter()
		    .filter(|(_, d)| (*d - max_distance).abs() < f64::EPSILON)
		    .map(|(gid, _)| *gid)
		    .collect();
		
		let tasks_per_group = vm.tasks_per_group.lock().await;
		let best_group = candidate_groups.into_iter()
		    .min_by_key(|gid| tasks_per_group.get(gid)
				.map(|v| v.len())
				.unwrap_or(0))
		    .unwrap();
		drop(tasks_per_group);

		// After finding the "best_group"
		let workloads = vm.vcpus_workload.lock().await;
		let selected = vm.vcpus_per_group.get(&best_group)
		    .expect("Schedule::Jaccard -> Best group to select vCPU.")
		    .iter()
		    .min_by_key(|&&vcpu_id| workloads[vcpu_id])
		    .copied()
		    .expect("Schedule::Jaccard -> At least one vCPU in best group");
		drop(workloads);

		task.task_set_current_vcpu_id(selected);
		debug!(
		    "[VM {}] Selected vCPU ({}) - Jaccard ({:?})",
		    vm.clone().vm_get_id(),
		    selected,
		    thread::current().id(),
		);

		selected
	    },
	    Scheduler::CDF { tol } => {
		let recent = task.get_hotness_recent();
		let previous = task.get_hotness_previous();

		if !Self::hotness_drift_significant(&previous, &recent, 0.8) {
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

		let task_p = Self::percentiles3_from_hotness(&recent);

		let agg_snapshot: HashMap<usize, HashMap<usize, u64>> = {
		    let agg = vm.group_hotness_agg.lock().await;
		    agg.clone()
		};

		let mut group_valid_counts: HashMap<usize, usize> = HashMap::with_capacity(agg_snapshot.len());
		for (&gid, group_map) in agg_snapshot.iter() {
		    if group_map.is_empty() {
			group_valid_counts.insert(gid, 3);
		    } else {
			let gp = Self::percentiles3_from_hotness(group_map);
			let vc = Self::percentile_valid_count(task_p, gp, *tol);
			group_valid_counts.insert(gid, vc);
		    }
		}

		let max_valid = group_valid_counts.values().copied().max().unwrap_or(0);

		let tasks_per_group_len: HashMap<usize, usize> = {
		    let tpg = vm.tasks_per_group.lock().await;
		    tpg.iter().map(|(&g, v)| (g, v.len())).collect()
		};

		let mut best_group = None;
		let mut best_len = usize::MAX;
		for (&gid, &count) in group_valid_counts.iter() {
		    if count == max_valid {
			let len = *tasks_per_group_len.get(&gid).unwrap_or(&0);
			if len < best_len {
			    best_len = len;
			    best_group = Some(gid);
			}
		    }
		}
		let best_group = best_group.expect("Schedule::CDF -> Best group wasn't found.");

		let selected = {
		    let workloads = vm.vcpus_workload.lock().await;
		    vm.vcpus_per_group
			.get(&best_group)
			.expect("Schedule::CDF -> Best group to select vCPU")
			.iter()
			.copied()
			.min_by_key(|&vcpu_id| workloads[vcpu_id])
			.expect("Schedule::CDF -> At least one vCPU in 'best group'.")
		};

		task.task_set_current_vcpu_id(selected);
		debug!(
		    "[VM {}] Selected vCPU ({}) - CDF ({:?})",
		    vm.clone().vm_get_id(),
		    selected,
		    thread::current().id(),
		);
		selected
		// let task_cdf = Self::compute_cdf(&recent);
		// let task_percentiles: Vec<usize> = Self::extract_percentiles(&task_cdf, &[0.25, 0.5, 0.75]);

		// let mut group_valid_counts: HashMap<usize, usize> = HashMap::new();
		// let tolerance = tol;

		// for &group_id in vm.vcpus_per_group.keys() {
		//     let group_hotness = vm.group_task_hotness(group_id).await;

		//     if group_hotness.is_empty() {
		// 	group_valid_counts.insert(group_id, 3);
		// 	continue;
		//     }

		//     let group_cdf = Self::compute_cdf(&group_hotness);
		//     let group_percentiles: Vec<usize> = Self::extract_percentiles(&group_cdf, &[0.25, 0.5, 0.75]);

		//     let mut valid_count = 0;		    
		//     for i in 0..task_percentiles.len() {
			
		// 	// if group_percentiles[i] == 0.0 {
		// 	//     if task_percentiles[i] == 0.0 { valid_count += 1; }
		// 	//     continue;
		// 	// }
			
		// 	let dist = (group_percentiles[i] as f64) * *tolerance;
		// 	let lower = (group_percentiles[i] as f64) + dist;
		// 	let upper = (group_percentiles[i] as f64) - dist;
		// 	if (task_percentiles[i] as f64) <= lower || (task_percentiles[i] as f64) >= upper {
		// 	    valid_count += 1;
		// 	}
		//     }

		//     group_valid_counts.insert(group_id, valid_count);
		// }


		// let max_valid = group_valid_counts.values()
		//     .copied()
		//     .max()
		//     .unwrap_or(0);
		// let candidate_groups: Vec<usize> = group_valid_counts
		//     .iter()
		//     .filter(|&(_, &count)| count == max_valid)
		//     .map(|(&gid, _)| gid)
		//     .collect();

		// let tasks_per_group = vm.tasks_per_group.lock().await;
		// // As done in Jaccard, the tie-breaking will be the group with least amount of tasks
		// let best_group = candidate_groups
		//     .into_iter()
		//     .min_by_key(|gid| tasks_per_group.get(gid).map(|v| v.len()).unwrap_or(0))
		//     .expect("Schedule::CDF -> Best group wasn't found.");
		// drop(tasks_per_group);
		
		// let workloads = vm.vcpus_workload.lock().await;
		// let selected = vm.vcpus_per_group
		//     .get(&best_group)
		//     .expect("Schedule::CDF -> Best group to select vCPU")
		//     .iter()
		//     .min_by_key(|&&vcpu_id| workloads[vcpu_id])
		//     .copied()
		//     .expect("Schedule::CDF -> At least one vCPU in 'best group'.");
		// drop(workloads);
		    
		// task.task_set_current_vcpu_id(selected);

		// debug!(
		//     "[VM {}] Selected vCPU ({}) - CDF ({:?})",
		//     vm.clone().vm_get_id(),
		//     selected,
		//     thread::current().id(),
		// );

		// selected
		
	    },
	    Scheduler::GraphBased { } => {

		// let recent = task.get_hotness_recent();
		// let global = task.get_hotness_global();

		// if !Self::hotness_drift_significant(global, &recent, 0.2) {
		//     let selected = task.task_get_last_vcpu_id();

		//     debug!(
		// 	"[VM {}] Selected vCPU ({}) - Graph ({:?})",
		// 	vm.clone().vm_get_id(),
		// 	selected,
		// 	thread::current().id(),
		//     );
		    
		//     task.task_set_current_vcpu_id(selected);
		//     return selected;
		// }

		// let task_id = task.ts_id;

		// let mut dag = vm.task_conflict_dag.lock().await;
		// let mut tasks_map = HashMap::new();

		// let running_tasks = vm.vm_tasks_running.lock().await;

		// // Adding task to DAG and getting weights
		// // weights = frequence(a) * frequence(b) if key(a) == key(b)
		// for t in running_tasks.iter() {
		//     if t.ts_id == task_id {
		// 	continue;
		//     }

		//     let weight = Self::conflict_weight(&task, t);
		//     if weight > 0.0 {
		// 	tasks_map.insert(t.ts_id as usize, weight);
		// 	dag.entry(t.ts_id.into()).or_default().insert(task_id.into(), weight);
		//     }
		// }
		// dag.insert(task_id.into(), tasks_map);

		// let mut group_conflicts: HashMap<usize, f64> = HashMap::new();

		// // For each grup, we get the amount of accum. conflict with the tasks
		// // currently running in that group.
		// for (&group_id, vcpus) in &vm.vcpu_groups {
		//     let mut total_conflict = 0.0;
		//     for t in running_tasks.iter() {
		// 	if t.ts_id == task_id { continue }
		// 	let vcpu_id = t.task_get_current_vcpu_id();

		// 	// Identifies if vCPU is in current group
		// 	if vcpus.contains(&vcpu_id) {

		// 	    // If conflict found, sum it up
		// 	    if let Some(conflicts) = dag.get(&task_id.into()) {
		// 		total_conflict += conflicts.get(&t.ts_id.into()).copied().unwrap_or(0.0);
		// 	    }
			    
		// 	}
		//     }
		    
		//     group_conflicts.insert(group_id, total_conflict);
		// }
		// drop(running_tasks);

		// // Choosing all groups with the least amount of accum. conflict
		// let min_conflict = group_conflicts.values().copied().fold(f64::INFINITY, f64::min);
		// let candidate_groups: Vec<usize> = group_conflicts.iter()
		//     .filter_map(|(&gid, &conflict)|
		// 		if conflict == min_conflict {
		// 		    Some(gid)
		// 		}
		// 		else {
		// 		    None
		// 		}
		//     ).collect();

		// // If necessary, breaking the tie selecting the group with the least amount of workload
		// let mut group_loads: Vec<(usize, usize)> = Vec::with_capacity(candidate_groups.len());
		// for gid in candidate_groups.into_iter() {
		//     let vcpus: HashSet<usize> = vm.vcpu_groups.get(&gid)
		// 	.expect("group id must exist")
		// 	.iter()
		// 	.copied()
		// 	.collect();

		//     let counts_map = vm.get_task_count_per_vcpu_group(&vcpus).await;
		//     let load: usize = counts_map.values().sum();
		//     group_loads.push((gid, load));
			
		// }
		
		// let (selected_group_id, _) = group_loads
		//     .into_iter()
		//     .min_by_key(|(_, load)| *load)
		//     .expect("Schedule::GraphBased -> Expected at least one candidate group");


		// let selected_vcpus_slice = &vm.vcpu_groups[&selected_group_id];
		// let group_vcpus: HashSet<usize> = selected_vcpus_slice.iter().copied().collect();
		// let vcpu_task_count = vm.get_task_count_per_vcpu_group(&group_vcpus).await;

		// let selected = selected_vcpus_slice
		//     .iter()
		//     .min_by_key(|vcpu_id| {
		// 	vcpu_task_count.get(vcpu_id).copied().unwrap_or(0)
		//     })
		//     .copied()
		//     .expect("Scheduler::GraphBased -> Expected to find a vCPU id");

		// task.task_set_current_vcpu_id(selected);
		
		// debug!(
		//     "[VM {}] Selected vCPU ({}) - Graph ({:?})",
		//     vm.clone().vm_get_id(),
		//     selected,
		//     thread::current().id(),
		// );

		// selected
		0
	    }
	    Scheduler::Balanced { } => {
		let tasks_per_group = vm.tasks_per_group.lock().await;
		let (&best_group_id, _) = tasks_per_group
		    .iter()
		    .min_by_key(|(_, tasks)| tasks.len())
		    .expect("Schedule::Balanced => No NUMA-groups found.");
		drop(tasks_per_group);

		let mut best_vcpu = None;
		let mut min_workload = u64::MAX;
		
		let vcpus_per_group = &vm.vcpus_per_group;
		let vcpus_workload = vm.vcpus_workload.lock().await;
		if let Some(vcpus) = vcpus_per_group.get(&best_group_id) {
		    for &vcpu_id in vcpus {
			if let Some(&load) = vcpus_workload.get(vcpu_id) {
			    if load < min_workload {
				min_workload = load;
				best_vcpu = Some(vcpu_id);
			    }
			}
		    }
		}

		drop(vcpus_per_group);
		drop(vcpus_workload);
		
		let selected = best_vcpu.expect("Schedule::Balanced => No vCPU selected. ERROR");
		debug!(
		    "[VM {}] Selected vCPU ({}) - Balanced ({:?})",
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
    #[inline]
    fn percentiles3_from_hotness(h: &HashMap<usize, u64>) -> [usize; 3] {
	if h.is_empty() {
	    return [0; 3]
	}

	let mut items: Vec<(usize, u64)> = Vec::with_capacity(h.len());
	items.extend(h.iter().map(|(&idx, &cnt)| (idx, cnt)));
	items.sort_unstable_by_key(|&(idx, _)| idx);

	let total: u128 = items.iter().map(|&(_, c)| c as u128).sum();
	if total == 0 {
	    let first = items[0].0;
	    return [first, first, first];
	}

	let t25 = (total as f64 * 0.25).ceil() as u128;
	let t50 = (total as f64 * 0.50).ceil() as u128;
	let t75 = (total as f64 * 0.75).ceil() as u128;

	let mut cumul: u128 = 0;
	let mut p25 = None;
	let mut p50 = None;
	let mut p75 = None;

	for (idx, cnt) in items {
	    cumul += cnt as u128;
	    if p25.is_none() && cumul >= t25 { p25 = Some(idx); }
	    if p50.is_none() && cumul >= t50 { p50 = Some(idx); }
	    if p75.is_none() && cumul >= t75 { p75 = Some(idx); }
	    if p25.is_some() && p50.is_some() && p75.is_some() { break; }
	}

	[
	    p25.unwrap_or(0),
	    p50.unwrap_or(0),
	    p75.unwrap_or(0),
	]
	
    }

    #[inline]
    fn percentile_valid_count(task_p: [usize; 3], group_p: [usize; 3], tol: f64) -> usize {
	let mut count = 0usize;
	for i in 0..3 {
	    let g = group_p[i] as f64;
	    let t = task_p[i] as f64;
	    let band = tol * g.max(1.0);
	    if (t - g).abs() > band {
		count += 1;
	    }
	}

	count
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
	cdf_tol: f64,
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
		Arc::new(Mutex::new(Scheduler::CDF { tol: cdf_tol } ))
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
	let mut tasks_per_group: HashMap<usize, Vec<usize>> = HashMap::new();
	for idx in 0..num_l2_groups {
	    tasks_per_group.insert(idx, Vec::new());
	}
	    

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
		ram.clone(),
		l2_index.try_into().unwrap()
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

	let mut vcpus_workload: Vec<u64> = Vec::with_capacity(num_vcpus);

	for vcpu_i in 0..num_vcpus {
	    vcpus_workload.push(0);
	}
	
	let mut group_hotness_agg: HashMap<usize, HashMap<usize, u64>> = HashMap::new();
	for idx in 0..num_l2_groups {
	    group_hotness_agg.insert(idx, HashMap::new());
	}

	Ok(Self {
	       vm_id			: id,
	       vm_vcpus			: vcpus,
	       l2_caches		: l2_groups,
	       l3_cache			: l3_shared,
	       vm_total_task_count	: num_tasks,
	       contention_time_nanos,
	       schedule_time,
	       vcpu_senders		: Arc::new(senders),
	       vm_tasks_finished	: Arc::new(Mutex::new(VecDeque::new())),
               vm_tasks_ready		: Arc::new(Mutex::new(VecDeque::new())),
	       vm_tasks_running		: Arc::new(Mutex::new(VecDeque::new())),
	       scheduler		: vm_scheduler,
	       mapper			: vm_mapper,
	       quantum			: 300,
	       preemptive,
	       ram,
	       tasks_per_group		: Arc::new(Mutex::new(tasks_per_group)),
               vcpus_per_group		: vcpu_l2_groups,
	       vcpus_workload           : Arc::new(Mutex::new(vcpus_workload)),
	       group_hotness_agg        : Arc::new(Mutex::new(group_hotness_agg)),
	       task_conflict_dag	: Arc::new(Mutex::new(HashMap::new())),
	})
    }

    pub async fn completion_loop(
	self: Arc<VM>,
	mut completion_receiver: mpsc::Receiver<(Task, u64, u8)>
    ) {
	while let Some((mut task, execution_time, selected_vcpu)) = completion_receiver.recv().await {
	    let vcpu_group_id = self.vm_vcpus[selected_vcpu as usize].lock().await.vcpu_group_id;

	    let start = Instant::now();
	    
	    {
		let recent_hotness = task.get_hotness_recent().clone();
		let mut agg = self.group_hotness_agg.lock().await;
		if let Some(group_map) = agg.get_mut(&(vcpu_group_id as usize)) {
		    for (idx, cnt) in recent_hotness.iter() {
			if let Some(value) = group_map.get_mut(idx) {
			    if *value > *cnt {
				*value -= *cnt;
			    } else {
				group_map.remove(idx);
			    }
			}
		    }
		}
		
		
	    }

	    // Removing task from running map	    
	    let mut running = self.vm_tasks_running.lock().await;
	    running.remove(task.task_ts_id() as usize);
	    drop(running);
	    
	    // Updating vCPU workload
	    let mut vcpus_workload = self.vcpus_workload.lock().await;
	    vcpus_workload[selected_vcpu as usize] -= task.task_current_exectime();
	    drop(vcpus_workload);

	    // Removing task from group
	    let mut tpg = self.tasks_per_group.lock().await;
	    if let Some(vec) = tpg.get_mut(&(vcpu_group_id as usize)) {
		if let Some(pos) = vec.iter().rposition(|&x| x == task.task_ts_id() as usize) {
		    vec.swap_remove(pos);
		}
	    }
	    
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
	completion_sender: mpsc::Sender<(Task, u64, u8)>
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
		    task.start_time = Some(Instant::now());
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
		let vcpu_group_id = self.vm_vcpus[selected_vcpu as usize].lock().await.vcpu_group_id;
		
		self.vm_tasks_running.lock()
		    .await
		    .push_back(task.clone());
		
		self.tasks_per_group.lock()
		    .await
		    .entry(vcpu_group_id as usize)
		    .or_default()
		    .push(task.task_ts_id() as usize);
		
		let mut vcpu_wl = self.vcpus_workload.lock().await;
		vcpu_wl[selected_vcpu as usize] += task.task_current_exectime();
		drop(vcpu_wl);

		{
		    let recent_hotness = task.get_hotness_recent().clone();
		    let mut agg = self.group_hotness_agg.lock().await;
		    if let Some(group_map) = agg.get_mut(&(vcpu_group_id as usize)) {
			for (idx, cnt) in recent_hotness.iter() {
			    *group_map.entry(*idx).or_insert(0) += *cnt;
			}
		    } else {
			// Sanity check
			agg.insert(vcpu_group_id as usize, recent_hotness);
		    }
		}
		
		self.contention_time_nanos.fetch_add(start.elapsed().as_nanos() as usize, Ordering::Relaxed);
		
		
		
		let (response_transmitter, response_receiver) = oneshot::channel();
		if self.vcpu_senders[selected_vcpu].send((task.clone(), response_transmitter)).await.is_ok() {
		    let completion_sender = completion_sender.clone();
		    tokio::spawn(async move {
			if let Ok(returned_task) = response_receiver.await {
			    let _ = completion_sender.send(
				(returned_task, execution_time, selected_vcpu.try_into().unwrap())
			    ).await;
			}
		    });
		    
		    // If failed to send task, re-add task to be sent elsewhere
		    // and update vCPU workload and group membership
		} else {
		    let start = Instant::now();
		    
		    {
			let recent_hotness = task.get_hotness_recent().clone();
			let mut agg = self.group_hotness_agg.lock().await;
			if let Some(group_map) = agg.get_mut(&(vcpu_group_id as usize)) {
			    for (idx, cnt) in recent_hotness.iter() {
				if let Some(value) = group_map.get_mut(idx) {
				    if *value > *cnt {
					*value -= *cnt;
				    } else {
					group_map.remove(idx);
				    }
				}
			    }
			}
			    
			    
		    }
		    
		    
		    let mut vcpu_wl = self.vcpus_workload.lock().await;
		    vcpu_wl[selected_vcpu as usize] -= task.task_current_exectime();
		    drop(vcpu_wl);

		    let mut tpg = self.tasks_per_group.lock().await;
		    if let Some(vec) = tpg.get_mut(&(vcpu_group_id as usize)) {
			if let Some(pos) = vec.iter().rposition(|&x| x == task.task_ts_id() as usize) {
			    vec.swap_remove(pos);
			}
		    }
		    drop(tpg);

		    
		    self.vm_tasks_running.lock()
			.await
			.remove(task.task_ts_id() as usize);
		    
		    self.vm_tasks_ready.lock().await.push_front(task);

		    
		    self.contention_time_nanos.fetch_add(start.elapsed().as_nanos() as usize, Ordering::Relaxed);
		}
	    } else {
		tokio::time::sleep(Duration::from_millis(100)).await;
	    }
		
	}
	
	debug!("[VM {}] Finished processing.", self.vm_get_id());
    }

    pub async fn group_task_hotness(&self, group_id: usize) -> HashMap<usize, u64> {
	// Return a cloned snapshot of the aggregated hotness for the group.
	let agg = self.group_hotness_agg.lock().await;
	agg.get(&group_id).cloned().unwrap_or_default()
    }


    // TODO: this might not being used currently
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
