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
use log::debug;
use tokio::{
    sync::mpsc,
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
};


fn main() {
    env_logger::init();

    let args: Args = Args::parse(std::env::args().collect()).unwrap();
    let num_vms = args.number_of_vms();
    let num_vcpus = args.number_of_vcpus();
    let total = args.number_of_tasks();
    let scheduler = args.scheduler();
    let mapper = args.mapper();
    let seed = args.seed();
   
    let runtime = Builder::new_current_thread()
	.enable_all()
	.build()
	.unwrap();
    
    // TODO: Mudar total para ser parametrizável
    runtime.block_on(async  {
	let mut rng = StdRng::seed_from_u64(seed as u64);
	let total_tasks = Arc::new(AtomicUsize::new(total));
	let mut vm_raw = VM::new(0, num_vcpus, &scheduler, &mapper, Arc::clone(&total_tasks), total)
	    .await
	    .expect("Failed to create  VM");
	    
	debug!("Initiating... {:?}", thread::current().id());

	for i in 0..total {
	    // TODO: Os workloads das tarefas (idealmente) devem seguir uma distribuição
	    vm_raw.vm_task_ready_add(Task::new(i as u8, "foo", rng.gen_range(600..=1200))).await;
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
	println!("{:?},{:?},{:.4}", elapsed, contention, (contention as f64 / elapsed.as_nanos() as f64) * 100.0);
    });
}
