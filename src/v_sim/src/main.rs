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

fn main() {
    let start = Instant::now();
    env_logger::init();

    let args: Args = Args::parse(std::env::args().collect()).unwrap();
    let num_vms = args.number_of_vms();
    let num_vcpus = args.number_of_vcpus();
    let scheduler = args.scheduler();
    let mapper = args.mapper();
   
    let runtime = Builder::new_current_thread()
	.enable_all()
	.build()
	.unwrap();

    let total: usize = 10;
    runtime.block_on(async  {
	let total_tasks = Arc::new(AtomicUsize::new(total));
	let mut vm_raw = VM::new(0, num_vcpus, &scheduler, &mapper, Arc::clone(&total_tasks))
	    .await
	    .expect("Failed to create  VM");
	    
	debug!("Initiating... {:?}", thread::current().id());

	for i in 0..total {
	    vm_raw.vm_task_ready_add(Task::new(i as u8, "foo", 1000)).await;
	}
	let mut vm = Arc::new(vm_raw);

	let (completion_transmitter, completion_receiver) = mpsc::channel(100);
	let vm_schedule: Arc<VM> = Arc::clone(&vm);
	let vm_completion: Arc<VM> = Arc::clone(&vm);
	
	let schedule_tread = std::thread::spawn(move || { 
	    let rt = tokio::runtime::Builder::new_current_thread()
		.enable_all()
		.build()
		.unwrap();

	    rt.block_on(async move {
		vm_schedule.schedule_loop(completion_transmitter).await;
	    });
	});

	let completion_tread = std::thread::spawn(move || { 
	    let rt = tokio::runtime::Builder::new_current_thread()
		.enable_all()
		.build()
		.unwrap();

	    rt.block_on(async move {
		vm_completion.completion_loop(completion_receiver).await;
	    });
	});
	
	schedule_tread.join().unwrap();
	completion_tread.join().unwrap();
    });
    println!("Elapsed {:?}", start.elapsed());
}
