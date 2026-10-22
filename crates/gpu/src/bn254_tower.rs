//! Bounded BN254 extension-field arithmetic diagnostic.
//!
//! This is an explicit preparation step for a future pairing backend. Results
//! do not authorize point decoding, SRS publication or protocol acceptance.
//! Arithmetic lives in CUDA C++; Rust only owns buffers and checks the ABI.
use anyhow::{ensure, Result};
use std::ffi::{c_char, c_void};

pub const MAX_BN254_TOWER_ITEMS: usize = 64;
pub const BN254_TOWER_NON_CANONICAL: u32 = 1;
pub const BN254_TOWER_ZERO_INVERSE: u32 = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bn254TowerField { Fq2, Fq6, Fq12 }

impl Bn254TowerField {
    pub const fn element_bytes(self) -> usize {
        match self { Self::Fq2 => 64, Self::Fq6 => 192, Self::Fq12 => 384 }
    }
    fn degree(self) -> u32 {
        match self { Self::Fq2 => 2, Self::Fq6 => 6, Self::Fq12 => 12 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bn254TowerOperation {
    Add, Subtract, Multiply, Square, Inverse, Frobenius(u32),
}

#[derive(Debug, PartialEq, Eq)]
pub struct Bn254TowerProbe {
    /// AoS canonical LE32 Fq coefficients in recursive c0,c1[,c2] order.
    /// Rejected items are zeroed; inspect statuses before using any item.
    pub output: Vec<u8>,
    /// 0=success, 1=noncanonical input, 2=zero has no multiplicative inverse.
    pub statuses: Vec<u32>,
}

extern "C" {
    fn dvgpu_bn254_tower_inspect(owner: *mut c_void, degree: u32, operation: u32, power: u32,
        left: *const u8, left_bytes: usize, right: *const u8, right_bytes: usize,
        output: *mut u8, output_bytes: usize, statuses: *mut u32, statuses_count: usize,
        error: *mut c_char, error_capacity: usize) -> i32;
}

impl crate::IcicleBackend {
    /// Runs at most 64 extension-field items synchronously. Binary operations
    /// require equally sized right input; unary operations require empty right.
    /// Noncanonical fields are rejected per item, without modular
    /// reduction. This inspector never changes protocol/backend selection.
    pub fn inspect_bn254_tower(&self, field: Bn254TowerField, operation: Bn254TowerOperation,
        left: &[u8], right: &[u8]) -> Result<Bn254TowerProbe> {
        let width = field.element_bytes();
        ensure!(left.len() % width == 0, "invalid BN254 tower element length");
        let count = left.len() / width;
        ensure!(count <= MAX_BN254_TOWER_ITEMS, "BN254 tower batch exceeds 64 items");
        let (code, power, binary) = match operation {
            Bn254TowerOperation::Add => (1, 0, true),
            Bn254TowerOperation::Subtract => (2, 0, true),
            Bn254TowerOperation::Multiply => (3, 0, true),
            Bn254TowerOperation::Square => (4, 0, false),
            Bn254TowerOperation::Inverse => (5, 0, false),
            Bn254TowerOperation::Frobenius(power) => {
                ensure!(power <= 12, "BN254 tower Frobenius power exceeds 12");
                (6, power, false)
            }
        };
        if binary {
            ensure!(right.len() == left.len(), "BN254 tower binary input length mismatch");
        } else {
            ensure!(right.is_empty(), "BN254 tower unary operation has extra input");
        }
        let mut result = Bn254TowerProbe { output: vec![0; left.len()], statuses: vec![0; count] };
        let mut error = [0u8; 256];
        // SAFETY: native borrows exact-length initialized host buffers only for
        // this synchronous call. Native stages both outputs until stream sync
        // and resource cleanup succeed; no Rust pointer is retained afterwards.
        let status = unsafe { dvgpu_bn254_tower_inspect(self.handle.as_ptr(),field.degree(),code,power,
            left.as_ptr(),left.len(),right.as_ptr(),right.len(),result.output.as_mut_ptr(),result.output.len(),
            result.statuses.as_mut_ptr(),result.statuses.len(),error.as_mut_ptr().cast(),error.len()) };
        crate::icicle::check_status(status,&error)?;
        for &item in &result.statuses {
            ensure!(item == 0 || item == BN254_TOWER_NON_CANONICAL ||
                (item == BN254_TOWER_ZERO_INVERSE && operation == Bn254TowerOperation::Inverse),
                "invalid BN254 tower item status");
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests;
