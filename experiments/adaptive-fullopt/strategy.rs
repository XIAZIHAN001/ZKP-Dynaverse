//! External policy only: existing protocol and native algorithms are unchanged.
#[derive(Clone, Copy, Debug)]
pub struct Strategy {
    pub window:usize,
    pub packet_cap:Option<usize>,
    pub undo_bytes:u64,
    pub limit_bytes:u64,
    pub cache_roots:usize,
    pub calibrate_affine:bool,
}

impl Strategy {
    pub fn for_size(n:usize)->Result<Self,&'static str> {
        if !n.is_power_of_two() || !(64..=65536).contains(&n) {
            return Err("policy supports power-of-two n in64..=65536; larger setup domains need separate admission");
        }
        let (window,packet_cap,calibrate_affine)=match n {
            64..=512 => (6,Some(4096),false),
            1024..=4096 => (8,Some(4096),true),
            _ => (16,None,true),
        };
        // Native journals save each writable element once, rather than every
        // intermediate write. Keep a power-of-two arena above the exact bound.
        let undo_bytes=(1472*n as u64).next_power_of_two().max(64<<20);
        Ok(Self{window,packet_cap,undo_bytes,limit_bytes:2<<30,
            cache_roots:64,calibrate_affine})
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn small_hotspot_retains_geometry_below_horizon() {
        let s=Strategy::for_size(256).unwrap();
        assert!(2<s.window/2,"two-wire hotspot must stay below rebuild horizon");
    }
    #[test]
    fn large_transform_uses_tiles_and_undo_contains_unique_writes() {
        let s=Strategy::for_size(65536).unwrap();
        assert!(s.packet_cap.is_none());
        assert!(s.undo_bytes>=96_468_992);
        assert!(s.calibrate_affine);
    }
    #[test]
    fn small_owner_does_not_reserve_large_undo_and_coalesces_reads() {
        let s=Strategy::for_size(256).unwrap();
        assert!(s.undo_bytes<=64<<20);
        assert!(s.cache_roots>=64);
    }
    #[test]
    fn unsupported_domain_is_rejected_before_allocation() {
        for n in [0,63,255,131072,usize::MAX] {assert!(Strategy::for_size(n).is_err());}
    }
}
