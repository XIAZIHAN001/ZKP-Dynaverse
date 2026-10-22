//! Deterministic fan-in-two circuit used by the CPU reference proof.

use crate::dynamo::DynamoPermutation;
use crate::types::{Instance, Witness};
use anyhow::{ensure, Result};
use ark_bn254::Fr;
use ark_ff::Zero;

#[derive(Clone, Debug)]
pub struct ToyCircuit {
    n: usize,
    m: usize,
    permutation: DynamoPermutation,
}

impl ToyCircuit {
    pub fn new(n: usize) -> Result<Self> {
        ensure!(n > 0 && n.is_power_of_two(), "n must be a nonzero power of two");
        let m = n.checked_mul(8).ok_or_else(|| anyhow::anyhow!("circuit size overflow"))?;
        let mut sigma: Vec<usize> = (0..m).collect();
        for i in 0..n {
            sigma.swap(i, 6 * n + i);
            sigma.swap(2 * n + i, 3 * n + i);
        }
        let permutation = DynamoPermutation::try_new(sigma)?;
        Ok(Self { n, m, permutation })
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn m(&self) -> usize {
        self.m
    }

    pub fn permutation(&self) -> DynamoPermutation {
        // The clone copies sigma and its inverse, 2m entries.
        crate::workcount::record_scan(2 * self.m);
        self.permutation.clone()
    }

    pub fn permutation_ref(&self) -> &DynamoPermutation { &self.permutation }

    pub fn valid_instance(&self) -> (Instance, Witness) {
        let mut public_inputs = vec![Fr::zero(); self.n];
        let mut wires = vec![Fr::zero(); 6 * self.n];
        for i in 0..self.n {
            let left = if i == 0 {
                Fr::from(1u64)
            } else {
                Fr::from((i + 1) as u64)
            };
            let right = if i == 0 {
                -Fr::from(1u64)
            } else {
                Fr::from((2 * i + 3) as u64)
            };
            let sum = left + right;
            let multiplier = Fr::from((i + 7) as u64);
            public_inputs[i] = left;
            wires[i] = left;
            wires[self.n + i] = right;
            wires[2 * self.n + i] = sum;
            wires[3 * self.n + i] = sum;
            wires[4 * self.n + i] = multiplier;
            wires[5 * self.n + i] = sum * multiplier;
        }
        (Instance { public_inputs }, Witness { wires })
    }

    pub fn validate(&self, instance: &Instance, witness: &Witness) -> Result<()> {
        ensure!(
            instance.public_inputs.len() == self.n,
            "public input length differs from circuit"
        );
        ensure!(witness.wires.len() == 6 * self.n, "witness length differs from circuit");
        let w = &witness.wires;
        crate::workcount::record_gates(self.n);
        for i in 0..self.n {
            ensure!(w[i] == instance.public_inputs[i], "public copy constraint fails at {i}");
            ensure!(w[2 * self.n + i] == w[3 * self.n + i], "wire copy constraint fails at {i}");
            ensure!(w[i] + w[self.n + i] == w[2 * self.n + i], "addition gate fails at {i}");
            ensure!(
                w[3 * self.n + i] * w[4 * self.n + i] == w[5 * self.n + i],
                "multiplication gate fails at {i}"
            );
        }
        Ok(())
    }
}
