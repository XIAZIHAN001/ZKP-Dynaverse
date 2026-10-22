//! Per-equation CPU pairing checks. Distinct equations are never aggregated.
use crate::trace::{PairingCounts, VerificationPhase, VerificationTrace};
use crate::types::VerificationMode;
use ark_bn254::{Bn254, G1Affine, G1Projective, G2Affine};
use ark_ec::{pairing::{Pairing, PairingOutput}, CurveGroup};
use ark_ff::Zero;
use std::collections::HashMap;
use std::time::Instant;

// The mutable borrow ties the checker to one recorder and prevents another
// checker from writing that recorder concurrently. No key or proof is owned.
pub(crate) struct PairingChecks<'a> {
    trace: &'a mut VerificationTrace,
    phase: VerificationPhase,
    mode: VerificationMode,
    // Select BN254's prepared type from the Pairing trait. All cached owners
    // remain local to this checker, so mutable external keys cannot stale them.
    prepared_g2: HashMap<G2Affine, <Bn254 as Pairing>::G2Prepared>,
}

impl<'a> PairingChecks<'a> {
    pub(crate) fn new(mode: VerificationMode, trace: &'a mut VerificationTrace) -> Self {
        Self { trace, phase: VerificationPhase::Structure, mode, prepared_g2: HashMap::new() }
    }

    pub(crate) fn begin(&mut self, phase: VerificationPhase) -> Option<Instant> {
        self.phase = phase;
        self.trace.start()
    }

    pub(crate) fn end(&mut self, started: Option<Instant>) {
        if let Some(started) = started {
            self.trace.measurement_mut(self.phase).elapsed += started.elapsed();
        }
    }

    pub(crate) fn equal(&mut self, left: (G1Affine, G2Affine), right: &[(G1Affine, G2Affine)]) -> bool {
        if self.trace.enabled {
            let terms = 1 + right.len();
            let calls = if self.mode == VerificationMode::Reference { terms } else { 1 };
            self.trace.measurement_mut(self.phase).counts.add(PairingCounts {
                equations: 1, pairing_terms: terms, miller_loops: calls,
                final_exponentiations: calls,
            });
        }
        if self.mode == VerificationMode::Product { return self.product_equal(left, right); }
        let actual = self.reference_pairing(left);
        let mut expected = PairingOutput::<Bn254>::zero();
        for &term in right { expected += self.reference_pairing(term); }
        actual == expected
    }

    /// Only private rebuild tables supply these fixed public preparations.
    /// One call is one equation; nothing is merged across roots or wires.
    pub(crate) fn equal_prepared(&mut self,left:G1Affine,right:G1Affine,
        generator:&<Bn254 as Pairing>::G2Prepared,
        divisor:&<Bn254 as Pairing>::G2Prepared)->bool {
        debug_assert_eq!(self.mode,VerificationMode::Product);
        let started=self.trace.start();
        // arkworks' Miller interface consumes its inputs, so retain the
        // immutable table and explicitly account these bounded deep clones.
        let prepared=[generator.clone(),divisor.clone()];
        if let Some(started)=started {
            let m=self.trace.measurement_mut(self.phase);
            m.g2_prepare_time+=started.elapsed();m.g2_prepare_hits+=2;
            m.counts.add(PairingCounts {equations:1,pairing_terms:2,
                miller_loops:1,final_exponentiations:1});
        }
        let g1s=[left,(-G1Projective::from(right)).into_affine()];
        let started=self.trace.start();
        let miller=Bn254::multi_miller_loop(g1s,prepared);
        if let Some(started)=started {self.trace.measurement_mut(self.phase).miller_time+=started.elapsed();}
        let started=self.trace.start();
        let result=Bn254::final_exponentiation(miller);
        if let Some(started)=started {self.trace.measurement_mut(self.phase).final_exponentiation_time+=started.elapsed();}
        match result {Some(value)=>value.is_zero(),None=>false}
    }

    fn product_equal(&mut self, left: (G1Affine, G2Affine), right: &[(G1Affine, G2Affine)]) -> bool {
        let mut g1s = Vec::with_capacity(1 + right.len());
        let mut g2s = Vec::with_capacity(1 + right.len());
        g1s.push(left.0);
        g2s.push(self.prepare_g2(left.1));
        for &(g1, g2) in right {
            // Bilinearity moves a right-hand pairing to the left by negating
            // its G1 point. This product contains exactly one equation.
            g1s.push((-G1Projective::from(g1)).into_affine());
            g2s.push(self.prepare_g2(g2));
        }
        let started = self.trace.start();
        let miller = Bn254::multi_miller_loop(g1s, g2s);
        if let Some(started) = started {
            self.trace.measurement_mut(self.phase).miller_time += started.elapsed();
        }
        let started = self.trace.start();
        let result = Bn254::final_exponentiation(miller);
        if let Some(started) = started {
            self.trace.measurement_mut(self.phase).final_exponentiation_time += started.elapsed();
        }
        // PairingOutput uses additive notation: zero is the target field's
        // multiplicative identity. A missing final exponentiation fails closed.
        match result { Some(value) => value.is_zero(), None => false }
    }

    fn prepare_g2(&mut self, point: G2Affine) -> <Bn254 as Pairing>::G2Prepared {
        let started=self.trace.start();
        let prepared=if let Some(prepared)=self.prepared_g2.get(&point) {
            let value=prepared.clone();
            if self.trace.enabled {self.trace.measurement_mut(self.phase).g2_prepare_hits+=1;}
            value
        } else {
            let value: <Bn254 as Pairing>::G2Prepared=point.into();
            self.prepared_g2.insert(point,value.clone());
            if self.trace.enabled {
                let len=self.prepared_g2.len();
                let measurement=self.trace.measurement_mut(self.phase);
                measurement.g2_prepare_builds+=1;
                measurement.prepared_peak_entries=measurement.prepared_peak_entries.max(len);
            }
            value
        };
        if let Some(started)=started {
            self.trace.measurement_mut(self.phase).g2_prepare_time+=started.elapsed();
        }
        prepared
    }

    fn reference_pairing(&mut self, term: (G1Affine, G2Affine)) -> PairingOutput<Bn254> {
        if !self.trace.enabled { return Bn254::pairing(term.0, term.1); }
        // Preserve single-pair reference arithmetic. G2 preparation is now
        // disjoint; Miller still includes G1 preparation inside arkworks.
        let started=Instant::now();
        let prepared: <Bn254 as Pairing>::G2Prepared=term.1.into();
        self.trace.measurement_mut(self.phase).g2_prepare_time+=started.elapsed();
        self.trace.measurement_mut(self.phase).g2_prepare_builds+=1;
        let started = Instant::now();
        let miller = Bn254::multi_miller_loop([term.0], [prepared]);
        self.trace.measurement_mut(self.phase).miller_time += started.elapsed();
        let started = Instant::now();
        let result = Bn254::final_exponentiation(miller).expect("BN254 pairing final exponentiation");
        self.trace.measurement_mut(self.phase).final_exponentiation_time += started.elapsed();
        result
    }
}
