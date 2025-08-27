//==================================================================================================
// Configuration
//==================================================================================================
#![deny(clippy::all)]

//==================================================================================================
// Modules
//==================================================================================================
mod args;

//==================================================================================================
// Imports
//==================================================================================================
use anyhow::Result;
use args::Args;
use task_lib::Task;
use vm_lib::VM;
use mem_lib::Addr;
use ram_lib::RAM;
use log::debug;
use tokio::{
    sync::{
	mpsc,
	Mutex,
    },
    runtime::Builder,
    time::{
	sleep,
	Duration,
    },
};
use std::{
    time::Instant,
    sync::{
	atomic::{
	    AtomicUsize,
	    Ordering,
	},
	Arc,
    },
    thread,
};

use rand::{
    Rng,
    SeedableRng,
    rngs::StdRng,
    distributions::{
	Distribution,
	WeightedIndex,
    },
};
use rand_distr::Zipf;


fn main() {
    env_logger::init();

    let args: Args = Args::parse(std::env::args().collect()).unwrap();
    let num_vms = args.number_of_vms();
    let num_vcpus = args.number_of_vcpus();
    let total = args.number_of_tasks();
    let scheduler = args.scheduler();
    let mapper = args.mapper();
    let cdf_tol = args.tol();
    let seed = args.seed();
   
    let runtime = Builder::new_current_thread()
	.enable_all()
	.build()
	.unwrap();
    
    runtime.block_on(async  {
	let mut rng = StdRng::seed_from_u64(seed as u64);
	let total_tasks = Arc::new(AtomicUsize::new(total));
	let mut ram = Arc::new(Mutex::new(RAM::new(1048576).expect("Failed to Create RAM")));
	let mut vm_raw = VM::new(
	    0,
	    num_vcpus,
	    &scheduler,
	    &mapper,
	    Arc::clone(&total_tasks),
	    total,
	    cdf_tol,
	    Arc::clone(&ram)
	).await.expect("Failed to create VM");	    
	debug!("Initiating... {:?}", thread::current().id());
	let zipf = Zipf::new(100_000, 1.0).unwrap();	
	// let zipf = Zipf::new(100_000, 0.8).unwrap();
	let base = 0x0000_4000_0000;
	let page_size = 4096;
	for i in 0..total {
	    // let workload_size = rng.gen_range(5000..=6000);
	    let workload_size = 3000;
	    let addresses: Vec<Addr> = (0..workload_size)
		.map(|_| {
		    let page_index = zipf.sample(&mut rng) as u64;
		    Addr::new(base + page_index * page_size)
		})
		.collect();
	    vm_raw.vm_task_ready_add(
		Task::new(i as u8,
			  Arc::clone(&ram),
			  "foo",
			  workload_size,
			  addresses).await
	    ).await;
	}
	let start = Instant::now();
	let mut vm = Arc::new(vm_raw);

	let (completion_transmitter, completion_receiver) = mpsc::channel(100);
	let vm_schedule: Arc<VM> = Arc::clone(&vm);
	let vm_completion: Arc<VM> = Arc::clone(&vm);
	
	let schedule_thread = std::thread::spawn(move || { 
	    let rt = tokio::runtime::Builder::new_current_thread()
		.enable_all()
		.build()
		.unwrap();

	    rt.block_on(async move {
		vm_schedule.schedule_loop(completion_transmitter).await;
	    });
	});

	let completion_thread = std::thread::spawn(move || { 
	    let rt = tokio::runtime::Builder::new_current_thread()
		.enable_all()
		.build()
		.unwrap();

	    rt.block_on(async move {
		vm_completion.completion_loop(completion_receiver).await;
	    });
	});
	
	schedule_thread.join().unwrap();
	completion_thread.join().unwrap();
	let elapsed = start.elapsed();
	let contention = vm.contention_time_nanos.load(Ordering::Relaxed);
	let schedule_time = vm.schedule_time.load(Ordering::Relaxed);
	println!("{:?},{:?},{:?},{:.4}", elapsed, contention, schedule_time, (contention as f64 / elapsed.as_nanos() as f64) * 100.0);
	let finished = vm.vm_tasks_finished.lock().await;
	for task in finished.iter() {
	    println!("Task {} - Response time: {:?}, Waiting time: {:?}, Execution Time: {:?}",
		     task.task_ts_id(),
		     task.first_response_time.unwrap_or_default(),
		     task.accumulated_wait_time,
		     task.accumulated_exec_time,
	    );
	}

	vm.print_cache_stats().await;
	
    });
}
