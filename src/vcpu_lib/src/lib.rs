//==================================================================================================
// Configuration
//==================================================================================================
#![deny(clippy::all)]


//==================================================================================================
// Imports
//==================================================================================================
use ::anyhow::Result;
use task_lib::Task;
use cache_lib::Cache;
use mmu_lib::{MMU, TranslateResult};
use ram_lib::RAM;
use tokio::{
    time::{
	sleep,
	Duration,
    },
    sync::{
	mpsc::Receiver,
	oneshot::Sender,
	Mutex,
    }
};

use std::{
    sync::{
	Arc,
	// Mutex,
    },
    time::{Instant},
    thread,
};
use log::debug;

//==================================================================================================
// Structures
//==================================================================================================
/// Represents a Virtual Machine's virutal CPU
#[derive(Clone)]
pub struct VCPU
{
    /// `vCPU` unique identifier
    vcpu_id		: u8,
    /// Which `VM` is this `vCPU` from
    vcpu_vm_id		: u8,
    /// `vCPU`'s running `Task`
    running_task	: Option<Task>,    
    /// `vCPU`'s L1 Cache
    pub cache_l1	: Cache,
    /// `vCPU`'s L2 Cache
    pub cache_l2	: Arc<Mutex<Cache>>,
    /// `vCPU`'s L3 Cache
    pub cache_l3	: Arc<Mutex<Cache>>,
    /// `vCPU`'s MMU
    pub mmu		: MMU,
    /// `vCPU`'s NUMA-group id
    pub vcpu_group_id   : u8,
}

//==================================================================================================
// Implementations
//==================================================================================================
impl VCPU {

    pub fn new(
	id: u8,
	vm_id: u8,
	cache_l1: Cache,
	cache_l2: Arc<Mutex<Cache>>,
	cache_l3: Arc<Mutex<Cache>>,
	ram     : Arc<Mutex<RAM>>,
	group_id: u8
    ) -> Result<Self>
    {
	debug!(
	    "[VM {}] vCPU {}: Creating vCPU ({:?})",
	    vm_id,
	    id,
	    thread::current().id(),
	);

	let mmu = MMU::new(ram)?;
	Ok(Self {
	    vcpu_id: id,
	    vcpu_vm_id: vm_id,
	    running_task: None,
	    cache_l1,
	    cache_l2,
	    cache_l3,
	    mmu,
	    vcpu_group_id: group_id
	})
    }

    
    pub async fn vcpu_loop(
	vcpu: Arc<Mutex<Self>>,
	mut receiver: Receiver<(Task, Sender<Task>)>
    ) -> Result<(), anyhow::Error> {
	while let Some((mut task, responder)) = receiver.recv().await {
	    if task.first_response_time.is_none() {
		let now = Instant::now();
		task.first_response_time = Some(now.duration_since(task.arrival_time));
		task.accumulated_wait_time += now.duration_since(task.arrival_time);
	    }
	    if let Some(left_time) = task.last_left_time.take() {
		    task.accumulated_wait_time += Instant::now().duration_since(left_time);
	    } 
	    task.task_set_current_vcpu_id(usize::MAX);
	    
	    let mut vcpu = vcpu.lock().await;
	    debug!(
		"[VM {}] vCPU {}: Starting to process Task ({}) ({:?})",
		vcpu.vcpu_vm_id,
		vcpu.vcpu_id,
		task.task_ts_id(),
		thread::current().id(),		    
	    );

	    
	    let mem_indexer = task.task_mem_indexer() as usize;
	    let task_exectime = task.task_current_exectime() as usize;
	    let task_cr3 = task.task_get_cr3();
	    let task_current_addresses = {
		let addresses = task.task_addresses(mem_indexer, mem_indexer + task_exectime)?;
		addresses.to_vec()
	    };
	    
	    for mut address in task_current_addresses {
		let virt_address = address.address_virtual_address();
		
		// Latency to access RAM (not using TLB)
		sleep(Duration::from_micros(30_000)).await;
		let phys_address = loop {

		    match vcpu.mmu.translate(task_cr3, virt_address).await {
			TranslateResult::Hit(p) => {
			    address.address_set_phys_address(p);
			    break p
			},
			TranslateResult::Fault(cause) => {
			    vcpu.mmu.handle_page_fault(task_cr3, virt_address, cause).await;
			}
		    }
		};
	
		// Latency to access L1
		sleep(Duration::from_nanos(2_000)).await;
		let (block_l1, index_l1) = vcpu.cache_l1.cache_lookup(phys_address);
		if let Some(block) = block_l1 {
		    task.update_hotness(index_l1.unwrap() as usize);
		    continue;
		}

		// Latency to access L2
		sleep(Duration::from_nanos(4_000)).await;
		let (block_l2, index_l2) = {
		    let mut l2 = vcpu.cache_l2.lock().await;
		    l2.cache_lookup(phys_address)
		};
		if let Some(block) = block_l2 {
		    task.update_hotness(index_l2.unwrap() as usize);
		    vcpu.cache_l1.cache_update(phys_address, block.clone());
		    continue
		}

		// Latency to access L3
		sleep(Duration::from_nanos(13_000)).await;
		let (block_l3, index_l3) = {
		    let mut l3 = vcpu.cache_l3.lock().await;
		    l3.cache_lookup(phys_address)
		};
		if let Some(block) = block_l3 {
		    {
			let mut l2 = vcpu.cache_l2.lock().await;
			l2.cache_update(phys_address, block.clone());
		    }
		    task.update_hotness(index_l3.unwrap() as usize);
		    vcpu.cache_l1.cache_update(phys_address, block.clone());
		    continue
		}

		// Setting block_size to be "num_words" for simplicity.
		// This shall be fixed id in the future
		let block_size = vcpu.cache_l1.cache_num_words();
		
		let block_ram = vcpu.mmu.fetch_from_ram(phys_address, block_size).await;
		{
		    let mut l3 = vcpu.cache_l3.lock().await;
		    l3.cache_update(phys_address, block_ram.clone());
		}
		{
		    let mut l2 = vcpu.cache_l2.lock().await;
		    l2.cache_update(phys_address, block_ram.clone());
		}
		vcpu.cache_l1.cache_update(phys_address, block_ram.clone());
		{
		    let index_l1 = vcpu.cache_l1.index_from_address(phys_address);
		    task.update_hotness(index_l1 as usize);
		}
	
	    }
	    task.task_set_last_vcpu_id(vcpu.vcpu_id.into());
	    task.last_left_time = Some(Instant::now());
	    
	    debug!(
		"[VM {}] vCPU {}: Finished processing Task ({}) ({:?})",
		vcpu.vcpu_vm_id,
		vcpu.vcpu_id,
		task.task_ts_id(),
		thread::current().id(),
	    );

	    // Whenever vCPU finishes the "processing" of a Task, we must send it back to the VM
	    let _ = responder.send(task);
	    
	}
	let vcpu = vcpu.lock().await;


	debug!("[VM {}] vCPU {}: Channel closed.",
	       vcpu.vcpu_vm_id,
	       vcpu.vcpu_id,
	);

	Ok(())

    }

    pub fn vcpu_id(&self) -> u8 {
	self.vcpu_id
    }
}
