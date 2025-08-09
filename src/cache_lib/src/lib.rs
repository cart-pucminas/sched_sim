//==================================================================================================
// Configuration
//==================================================================================================
#![deny(clippy::all)]
//==================================================================================================
// Imports
//==================================================================================================
use ::anyhow::Result;
use log::debug;
use std::collections::VecDeque;

//==================================================================================================
// Constants
//==================================================================================================
type WORD_TYPE = u64;
const WORD_SIZE: usize = std::mem::size_of::<WORD_TYPE>(); 

//==================================================================================================
// Enum
//==================================================================================================
#[derive(Clone)]
enum Cache_Level {
    First	= 1,
    Second	= 2,
    Third       = 3,
}

#[derive(Clone)]
enum Way_Replacement {
    Lru,
    Fifo,
}
//==================================================================================================
// Structures
//==================================================================================================
/// Represents a Cache Line (or Block) (i.e., the representation of a line in a cache way)
#[derive(Clone)]
struct Cache_Line {
    line_valid	: bool,
    line_tag	: u64,

    line_words	: Vec<WORD_TYPE>,
}

/// Represents a Cache instance.
#[derive(Clone)]
pub struct Cache {
    c_id			: u8,
    cache_level			: Cache_Level,
    cache_num_sets		: usize,
    cache_associativity_level	: usize,
    cache_num_words             : usize,

    // Represents the total size of our cache (in Bytes)
    cache_size			: usize,

    // Way replacement structures
    cache_way_replacement       : Way_Replacement,
    cache_replacement_state     : Vec<VecDeque<usize>>,

    // Cache here is represented as a contiguous sequence of cache lines.
    cache_lines			: Vec<Cache_Line>,

    index_bits			: u32,
    word_offset_bits		: u32,
    byte_offset_bits		: u32,
    block_offset_bits		: u32,
    
    cache_accesses		: usize,
    cache_hits			: usize,
    cache_misses		: usize,
}

//==================================================================================================
// Implementations
//==================================================================================================
impl Way_Replacement {
    fn replace_way(&self, usage_queue: &mut VecDeque<usize>) -> usize {
	match self {
	    Way_Replacement::Lru => {
		let victim = usage_queue.pop_back().unwrap();
		victim
	    }
	    Way_Replacement::Fifo => {
		let victim = usage_queue.pop_front().unwrap();
		victim
	    }
	}
    }

    fn update_usage(&self, usage_queue: &mut VecDeque<usize>, way: usize) {
	match self {
	    Way_Replacement::Lru => {
		if let Some(pos) = usage_queue.iter().position(|&w| w == way) {
		    usage_queue.remove(pos);
		}
		usage_queue.push_front(way);
	    }
	    Way_Replacement::Fifo => {
		if !usage_queue.contains(&way) {
		    usage_queue.push_back(way);
		}
	    }
	}
    }
    
}

impl Cache_Line {
    pub fn new (
	line_valid	: bool,
	line_tag	: u64,
	num_words	: usize,
    ) -> Result<Self> {
	
	let mut line_words : Vec<WORD_TYPE> = Vec::with_capacity(num_words);
	for word in 0..num_words {
	    line_words.push(0);
	}
	
	Ok(Self{
	    line_valid,
	    line_tag,
	    line_words,
	})	    
    }
    
}

impl Cache {
    pub fn new (
	cache_id		: u8,
	desired_cache_level	: &str,
	cache_size		: usize,
	associativity_level     : usize,
	num_words               : usize,
	desired_replacement     : &str,
    ) -> Result<Self> {
	debug!(
	    "Creating new Cache ({})",
	    cache_id,
	);

	// Sanity Check
	if !cache_size.is_power_of_two() {
	    return Err(anyhow::anyhow!(
		"Cache size ({}) must be power of 2",
		cache_size
	    ));
	}
	if !associativity_level.is_power_of_two() {
	    return Err(anyhow::anyhow!(
		"Associativity level ({}) must be power of 2",
		associativity_level
	    ));
	}
	if !num_words.is_power_of_two() {
	    return Err(anyhow::anyhow!(
		"Total number of words per block ({}) must be power of 2",
		num_words
	    ));
	}

	let cache_way_replacement = match desired_replacement {
	    "Lru" => {
		Way_Replacement::Lru
	    },
	    "Fifo" => {
		Way_Replacement::Fifo
	    },
	    &_ => {
		return Err(anyhow::anyhow!("invalid replacement strategy"));
	    }
	};
	
	let cache_level = match desired_cache_level {
	    "L1" => {
		Cache_Level::First
	    },
	    "L2" => {
		Cache_Level::Second
	    },
	    "L3" => {
		Cache_Level::Third
	    },
	    &_ => {
		return Err(anyhow::anyhow!("invalid cache level"));
	    }
	};

	let num_sets = cache_size / (associativity_level * (num_words * WORD_SIZE));
	let num_lines = num_sets * associativity_level;
	let mut cache_lines: Vec<Cache_Line> = Vec::with_capacity(num_lines);

	
	for line in 0..num_lines {
	    cache_lines.push(Cache_Line::new(false, 0, num_words)?);
	}

	// Address bits
	let index_bits = num_sets.ilog2();
	let word_offset_bits = num_words.ilog2();
	let byte_offset_bits = WORD_SIZE.ilog2();
	let block_offset_bits = word_offset_bits + byte_offset_bits;
	
	let mut cache_replacement_state: Vec<VecDeque<usize>> = Vec::with_capacity(num_sets);
	for _ in 0..num_sets {
	    let mut q = VecDeque::new();
	    for way in 0..associativity_level {
		q.push_back(way);
	    }
	    cache_replacement_state.push(q);
	}
	
	Ok(Self {
	    c_id			: cache_id,
	    cache_level,
	    cache_num_sets		: num_sets,
	    cache_associativity_level	: associativity_level,
	    cache_num_words		: num_words,
	    cache_size,
	    cache_lines,
	    
	    cache_way_replacement,
	    cache_replacement_state,

	    index_bits,
	    word_offset_bits,
	    byte_offset_bits,
	    block_offset_bits,
	    
	    cache_accesses		: 0,
	    cache_hits			: 0,
	    cache_misses		: 0,
	})
    }
    
    fn get_addr_bits(&self, addr: u64) -> (u64, u64, u64, u64) {			    

	let byte_offset = addr & ((1 << self.byte_offset_bits) - 1);
	let word_offset = (addr >> self.byte_offset_bits) & ((1 << self.word_offset_bits) - 1);
	let index = (addr >> self.block_offset_bits) & ((1 << self.index_bits) - 1);
	let tag = addr >> (self.block_offset_bits + self.index_bits);

	(byte_offset, word_offset, index, tag)
    }

    pub fn index_from_address(&self, phys_address: u64) -> usize {
	let block_number = phys_address as usize / (self.cache_num_words * WORD_SIZE);
	let num_sets = (self.cache_size / (self.cache_num_words * WORD_SIZE)) / self.cache_associativity_level;
	block_number % num_sets
    }
    
    pub fn cache_lookup(&mut self, addr: u64) -> (Option<Vec<WORD_TYPE>>, Option<u64>) {
	self.cache_accesses += 1;

	let (_, _, index, tag) = self.get_addr_bits(addr);
	
	// TODO: Maybe, for the sched. strategy, it's better to use ways than sets
	
	// Checking for hit
	let set_start = (index as usize) * self.cache_associativity_level;
	for way in 0..self.cache_associativity_level {
	    let line_index = set_start + way;
	    let line = &self.cache_lines[line_index];
	    if line.line_valid && line.line_tag == tag {
		self.cache_hits += 1;
		self.cache_way_replacement.update_usage(&mut self.cache_replacement_state[index as usize] , way);
		return (Some(line.line_words.clone()), Some(index));
	    }
	}

	self.cache_misses += 1;	
	(None, None)
    }

    pub fn cache_extract_sub_block(
	self,
	addr: u64,
	super_block: &[WORD_TYPE],
	target_block_size: usize
    ) -> Vec<WORD_TYPE> {
	let word_offset = (addr >> self.byte_offset_bits) as usize % super_block.len();
	let block_start = (word_offset / target_block_size) * target_block_size;
	super_block[block_start..block_start + target_block_size].to_vec()
    }
    
    pub fn cache_update(&mut self, addr: u64, words: Vec<WORD_TYPE>) {
	let (_, _, index, tag) = self.get_addr_bits(addr);

	let set = self.cache_replacement_state.get_mut(index as usize)
	    .unwrap_or_else(|| panic!("Index out of range: {} (addr = {:#x})", index, addr));
	let way = self.cache_way_replacement.replace_way(set);
	let line_index = (index as usize) * self.cache_associativity_level + way;
	let line = self.cache_lines.get_mut(line_index)
	    .unwrap_or_else(|| panic!(
		"line index out of range: {} (index={}, way={}, addr={:#x})",
		line_index, index, way, addr
	    ));
	self.cache_lines[line_index] = Cache_Line {
	    line_valid: true,
	    line_tag: tag,
	    line_words: words,		
	};

	self.cache_way_replacement.update_usage(&mut self.cache_replacement_state[index as usize], way);
    }
    
    pub fn cache_accesses(&self) -> usize {
	self.cache_accesses
    }
    
    pub fn cache_hits(&self) -> usize {
	self.cache_hits
    }
    
    pub fn cache_misses(&self) -> usize {
	self.cache_misses
    }

    pub fn cache_associativity_level(&self) -> usize {
	self.cache_associativity_level
    }

    pub fn cache_num_words(&self) -> usize {
	self.cache_num_words
    }
	
    
}
