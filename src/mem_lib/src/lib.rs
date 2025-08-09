//==================================================================================================
// Configuration
//==================================================================================================
#![deny(clippy::all)]

//==================================================================================================
// Imports
//==================================================================================================
use ::anyhow::Result;
use log::debug;

//==================================================================================================
// Structures
//==================================================================================================
/// Represents an Address Instance.
#[derive(Clone)]
pub struct Addr
{
    virtual_address	: u64,
    phys_address	: Option<u64>,
}

//==================================================================================================
// Implementations
//==================================================================================================
impl Addr {
    pub fn new (
	virtual_address: u64
    ) -> Self {
	// debug!(
	//     "Creating new Mem. Address ({})",
	//     virtual_address,
	// );


	Self {
	    virtual_address,
	    phys_address: None,
	}
    }

    pub fn address_virtual_address(&self) -> u64 {
	self.virtual_address
    }

    pub fn address_phys_address(&self) -> Option<u64> {
	self.phys_address
    }

    pub fn address_set_phys_address(&mut self, phys_address: u64) {
	self.phys_address = Some(phys_address)
    }
}
