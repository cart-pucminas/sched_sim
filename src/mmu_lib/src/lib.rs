//==================================================================================================
// Configuration
//==================================================================================================
#![deny(clippy::all)]

//==================================================================================================
// Imports
//==================================================================================================
use ram_lib::{RAM, PageTableEntry};
use anyhow::Result;
use std::collections::HashMap;

use log::debug;
use std::{
    sync::{Arc},
    thread,
};
use tokio::{
    sync::{Mutex},
    time::{
	sleep,
	Duration,
    }
};

//==================================================================================================
// Structures
//==================================================================================================
#[derive(Clone)]
pub struct MMU {
    ram: Arc<Mutex<RAM>>,
}

#[derive(Debug)]
pub enum PageFaultCause {
    MissingPML4,
    MissingPDPT,
    MissingPD,
    MissingPT,
    MissingFrame,
}

#[derive(Debug)]
pub enum TranslateResult {
    Hit(u64),
    Fault(PageFaultCause),
}

//==================================================================================================
// Implementations
//==================================================================================================
impl MMU {
    pub fn new(ram: Arc<Mutex<RAM>>) -> Result<Self> {
	debug!("[MMU] Creating MMU");
	Ok(Self {ram})
    }

    pub async fn translate(&self, cr3: u64, vaddr: u64) -> TranslateResult {
	let pml4_idx = ((vaddr >> 39) & 0x1FF) as u16;
	let pdpt_idx = ((vaddr >> 30) & 0x1FF) as u16;
	let pd_idx   = ((vaddr >> 21) & 0x1FF) as u16;
	let pt_idx   = ((vaddr >> 12) & 0x1FF) as u16;
	let offset   = vaddr & 0xFFF;

	let ram = self.ram.lock().await;
	// 1 - PML4 -> Get PDPT's frame_number
	let pdpt_frame = {
	    let pml4 = match ram.pml4_tables.get(&cr3) {
		Some(p) => p,
		None => return TranslateResult::Fault(PageFaultCause::MissingPML4),
	    };
	    match pml4.get(&pml4_idx) {
		Some(e) if e.present => e.frame_number,
		_ => return TranslateResult::Fault(PageFaultCause::MissingPDPT),
	    }
	};

	// 2 - PDPT -> Get PD's frame_number
	let pd_frame = {
	    let pdpt = match ram.pml4_tables.get(&pdpt_frame) {
		Some(p) => p,
		None => return TranslateResult::Fault(PageFaultCause::MissingPDPT),
	    };
	    match pdpt.get(&pdpt_idx) {
		Some(e) if e.present => e.frame_number,
		_ => return TranslateResult::Fault(PageFaultCause::MissingPD),
	    }

	};
	
	// 3 - PD -> Get PT's frame number
	let pt_frame = {
	    let pd = match ram.pml4_tables.get(&pd_frame) {
		Some(p) => p,
		None => return TranslateResult::Fault(PageFaultCause::MissingPD),
	    };
	    match pd.get(&pd_idx) {
		Some (e) if e.present => e.frame_number,
		_ => return TranslateResult::Fault(PageFaultCause::MissingPT),
	    }

	};
	
	// 4 - PT
	let page_frame = {
	    let pt = match ram.pml4_tables.get(&pt_frame) {
		Some(p) => p,
		None => return TranslateResult::Fault(PageFaultCause::MissingPT),
	    };
	    match pt.get(&pt_idx) { 
		Some(e) if e.present => e.frame_number,
		_ => return TranslateResult::Fault(PageFaultCause::MissingFrame),
	    }

	};
	
	let paddr = (page_frame << 12) | offset;
	TranslateResult::Hit(paddr)
    }

    pub async fn fetch_from_ram(&mut self, phys_address: u64, block_size: usize) -> Vec<u64> {
	sleep(Duration::from_micros(30_000)).await;

	let frame_id = phys_address >> 12;

	let ram_guard = self.ram.lock().await;
	let frame = ram_guard.frames.get(&frame_id);
	// Since data is not being used (only which blocks are accessed)
	// I'm filling it with blank
	let block_data = vec![0; block_size];
	drop(ram_guard);

	block_data
    }
    
    pub async fn handle_page_fault(&self, cr3: u64, vaddr: u64, cause: PageFaultCause) {	
	let mut ram = self.ram.lock().await;
	let new_frame = ram.ram_alloc_frame();

	match cause {
	    PageFaultCause::MissingPDPT => {
		let pml4 = ram.pml4_tables.get_mut(&cr3).unwrap();
		let pml4_index = ((vaddr >> 39) & 0x1FF) as u16;
		pml4.insert(
		    pml4_index,
		    PageTableEntry { present: true, frame_number: new_frame },
		);

		ram.pml4_tables.insert(new_frame, HashMap::new());
	    },
	    PageFaultCause::MissingPD => {
		let pdpt_frame = {
		    let pml4 = ram.pml4_tables.get_mut(&cr3).unwrap();
		    let pml4_index = ((vaddr >> 39) & 0x1FF) as u16;
		    pml4.get_mut(&pml4_index).unwrap().frame_number
		};
		
		let pdpt = ram.pml4_tables.entry(pdpt_frame).or_insert(HashMap::new());
		let pdpt_index = ((vaddr >> 30) & 0x1FF) as u16;
		pdpt.insert(
		    pdpt_index,
		    PageTableEntry { present: true, frame_number: new_frame },
		);
		
		ram.pml4_tables.insert(new_frame, HashMap::new());
	    },
	    PageFaultCause::MissingPT => {
		let pdpt_frame = {
		    let pml4 = ram.pml4_tables.get_mut(&cr3).unwrap();
		    let pml4_index = ((vaddr >> 39) & 0x1FF) as u16;
		    pml4.get_mut(&pml4_index).unwrap().frame_number
		};

		let pd_frame = {
		    let pdpt = ram.pml4_tables.entry(pdpt_frame).or_insert(HashMap::new());
		    let pdpt_index = ((vaddr >> 30) & 0x1FF) as u16;
		    pdpt.get_mut(&pdpt_index).unwrap().frame_number
		};
		
		let pd = ram.pml4_tables.entry(pd_frame).or_insert(HashMap::new());
		let pd_index = ((vaddr >> 21) & 0x1FF) as u16;
		
		pd.insert(
		    pd_index,
		    PageTableEntry { present: true, frame_number: new_frame },
		);
		ram.pml4_tables.insert(new_frame, HashMap::new());
		
	    },
	    PageFaultCause::MissingFrame => {
		let pdpt_frame = {
		    let pml4 = ram.pml4_tables.get_mut(&cr3).unwrap();
		    let pml4_index = ((vaddr >> 39) & 0x1FF) as u16;
		    pml4.get_mut(&pml4_index).unwrap().frame_number
		};

		let pd_frame = {
		    let pdpt = ram.pml4_tables.entry(pdpt_frame).or_insert(HashMap::new());
		    let pdpt_index = ((vaddr >> 30) & 0x1FF) as u16;
		    pdpt.get_mut(&pdpt_index).unwrap().frame_number
		};
		
		let pt_frame = {
		    let pd = ram.pml4_tables.entry(pd_frame).or_insert(HashMap::new());
		    let pd_index = ((vaddr >> 21) & 0x1FF) as u16;
		    pd.get_mut(&pd_index).unwrap().frame_number
		};

		let pt = ram.pml4_tables.entry(pt_frame).or_insert(HashMap::new());
		let pt_index = ((vaddr >> 12) & 0x1FF) as u16;
		
		pt.insert(
		    pt_index,
		    PageTableEntry { present: true, frame_number: new_frame },
		);
		
	    },

	    _ => unreachable!("All Page Fault Causes are covered."),
	}
    }
}
