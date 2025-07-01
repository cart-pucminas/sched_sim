//==================================================================================================
// Configuration
//==================================================================================================
#![deny(clippy::all)]


//==================================================================================================
// Imports
//==================================================================================================
use ::anyhow::Result;
use task_lib::Task;
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
}

//==================================================================================================
// Implementations
//==================================================================================================
impl VCPU {

    pub fn new(id: u8, vm_id: u8) -> Result<Self>
    {
	debug!(
	    "[VM {}] vCPU {}: Creating vCPU ({:?})",
	    vm_id,
	    id,
	    thread::current().id(),
	);
	
	Ok(Self {
	    vcpu_id: id,
	    vcpu_vm_id: vm_id,
	    running_task: None
	})
    }

    pub async fn vcpu_loop(vcpu: Arc<Mutex<VCPU>>, mut receiver: Receiver<(Task, Sender<Task>)>) {
	while let Some((mut task, responder)) = receiver.recv().await {
	    let vcpu = vcpu.lock().await;
	    debug!(
		"[VM {}] vCPU {}: Starting to process Task ({}) ({:?})",
		vcpu.vcpu_vm_id,
		vcpu.vcpu_id,
		task.task_ts_id(),
		thread::current().id(),		    
	    );

	    
	    sleep(Duration::from_millis(task.task_current_workload())).await;
	    
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

    }

    pub fn vcpu_id(self) -> u8 {
	self.vcpu_id
    }
    

}
