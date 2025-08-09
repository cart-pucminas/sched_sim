//==================================================================================================
// Configuration
//==================================================================================================
#![deny(clippy::all)]

//==================================================================================================
// Imports
//==================================================================================================
use mem_lib::Addr;
use anyhow::Result;
use std::{
    collections::{
	HashMap,
	VecDeque,
    },
    sync::{
	Arc,
	Mutex,
    },
};
use log::debug;
use std::thread;

//==================================================================================================
// Structures
//==================================================================================================
const FRAME_SIZE: usize = 4096;
#[derive(Clone)]
pub struct Frame {
    id: u64,
    data: Vec<u8>,
}

#[derive(Clone)]
pub struct PageTableEntry {
    pub present: bool,
    pub frame_number: u64,
}
type PageTable = std::collections::HashMap<u16, PageTableEntry>;

#[derive(Clone)]
pub struct RAM {
    pub frames: HashMap<u64, Frame>,
    frames_queue: VecDeque<u64>,
    pub pml4_tables: HashMap<u64, PageTable>,
    next_frame_id: u64,
    max_frames: usize,
}

//==================================================================================================
// Implementations
//==================================================================================================
impl RAM {
    pub fn new (max_frames: usize) -> Result<Self> {
	Ok(Self {
	    frames: HashMap::new(),
	    frames_queue: VecDeque::new(),
	    pml4_tables: HashMap::new(),
	    next_frame_id: 0,
	    max_frames,
	})
    }

    pub fn ram_alloc_frame(&mut self) -> u64 {
	if self.frames.len() >= self.max_frames {
	    if let Some(evicted) = self.frames_queue.pop_front() {
		debug!("[RAM] Evited frame {}", evicted);
		self.frames.remove(&evicted);
	    }
	}

	let id = self.next_frame_id;
	self.next_frame_id += 1;

	self.frames.insert(id, Frame { id, data: vec![0; FRAME_SIZE] });
	self.frames_queue.push_back(id);
	id
    }

    pub fn ram_create_plm4_for_task(&mut self, cr3: u64) {
	self.pml4_tables.insert(cr3, HashMap::new());
    }

}
