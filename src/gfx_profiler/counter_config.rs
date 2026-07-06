//! Hardware performance counter event definitions and multi-pass scheduling.
//!
//! GFX1100 exposes per-block performance counters with configurable event selection.
//! Each counter block has a limited number of hardware counters, and `RELEASE_MEM`
//! can only read back counter 0/1 per block, so we need multiple profiling passes.

/// A hardware performance counter event.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CounterEvent {
    pub name: &'static str,
    pub block: CounterBlock,
    pub event_id: u32,
    /// Register address for SELECT (e.g., 0xD040 for SQ_PERFCOUNTER0_SELECT)
    pub select_reg: u32,
    /// Register address for value LO (e.g., 0xD100 for SQ_PERFCOUNTER0_LO)
    pub value_lo_reg: u32,
}

impl CounterEvent {
    /// Return a copy of this event targeting the specified counter index (0 or 1).
    /// Counter 0 uses SELECT0/LO0, counter 1 uses SELECT1/LO1.
    pub fn with_counter(&self, idx: usize) -> CounterEvent {
        let mut evt = *self;
        if idx == 1 {
            evt.select_reg += 4;
            match self.block {
                CounterBlock::SQ => evt.value_lo_reg = SQ_PERFCOUNTER1_LO,
                CounterBlock::GRBM => evt.value_lo_reg = GRBM_PERFCOUNTER1_LO,
                CounterBlock::TCC => evt.value_lo_reg = TCC_PERFCOUNTER1_LO,
            }
        }
        evt
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CounterBlock {
    SQ,      // Per-CU shader engine counters
    GRBM,    // Global pipeline utilization (non-instanced)
    TCC,     // L2 cache (per-channel, 16 on GFX1100)
}

/// A single profiling pass: which counters to configure per block.
/// Each block is limited to 2 counters per pass (RELEASE_MEM reads counter 0/1 only).
#[derive(Clone, Debug)]
pub struct ProfilePass {
    pub sq_events: [Option<CounterEvent>; 2],
    pub grbm_events: [Option<CounterEvent>; 2],
    pub tcc_events: [Option<CounterEvent>; 2],
}

// ── GFX1100 Register Addresses ──────────────────────────────────────────

// SQ Performance Counter registers (per-CU, instanced via GRBM_GFX_INDEX)
pub const SQ_PERFCOUNTER0_SELECT: u32 = 0xD040;
pub const SQ_PERFCOUNTER1_SELECT: u32 = 0xD044;
pub const SQ_PERFCOUNTER0_LO: u32 = 0xD100;
pub const SQ_PERFCOUNTER1_LO: u32 = 0xD108;
pub const SQ_PERFCOUNTER_CTRL: u32 = 0xD030;
// Bit 6: CS (compute shader) enable, Bit 8: CNTR_MODE (0=snapshot)

// GRBM Performance Counters (global, non-instanced)
pub const GRBM_PERFCOUNTER0_SELECT: u32 = 0xD000;
pub const GRBM_PERFCOUNTER1_SELECT: u32 = 0xD004;
pub const GRBM_PERFCOUNTER0_LO: u32 = 0xD008;
pub const GRBM_PERFCOUNTER1_LO: u32 = 0xD010;

// TCC Performance Counters (per L2 channel, instanced via GRBM_GFX_INDEX)
pub const TCC_PERFCOUNTER0_SELECT: u32 = 0xD200;
pub const TCC_PERFCOUNTER1_SELECT: u32 = 0xD204;
pub const TCC_PERFCOUNTER0_LO: u32 = 0xD240;
pub const TCC_PERFCOUNTER1_LO: u32 = 0xD248;

// ── Standard Event Catalog ──────────────────────────────────────────────

// SQ events (per-CU)
pub const EVT_SQ_WAVES: CounterEvent = CounterEvent {
    name: "SQ_WAVES", block: CounterBlock::SQ, event_id: 0x01,
    select_reg: SQ_PERFCOUNTER0_SELECT, value_lo_reg: SQ_PERFCOUNTER0_LO,
};
pub const EVT_SQ_INSTS: CounterEvent = CounterEvent {
    name: "SQ_INSTS", block: CounterBlock::SQ, event_id: 0x07,
    select_reg: SQ_PERFCOUNTER0_SELECT, value_lo_reg: SQ_PERFCOUNTER0_LO,
};
pub const EVT_SQ_INSTS_VALU: CounterEvent = CounterEvent {
    name: "SQ_INSTS_VALU", block: CounterBlock::SQ, event_id: 0x0A,
    select_reg: SQ_PERFCOUNTER0_SELECT, value_lo_reg: SQ_PERFCOUNTER0_LO,
};
pub const EVT_SQ_INSTS_SALU: CounterEvent = CounterEvent {
    name: "SQ_INSTS_SALU", block: CounterBlock::SQ, event_id: 0x0B,
    select_reg: SQ_PERFCOUNTER0_SELECT, value_lo_reg: SQ_PERFCOUNTER0_LO,
};
pub const EVT_SQ_INSTS_SMEM: CounterEvent = CounterEvent {
    name: "SQ_INSTS_SMEM", block: CounterBlock::SQ, event_id: 0x0C,
    select_reg: SQ_PERFCOUNTER0_SELECT, value_lo_reg: SQ_PERFCOUNTER0_LO,
};
pub const EVT_SQ_INSTS_FLAT: CounterEvent = CounterEvent {
    name: "SQ_INSTS_FLAT", block: CounterBlock::SQ, event_id: 0x0D,
    select_reg: SQ_PERFCOUNTER0_SELECT, value_lo_reg: SQ_PERFCOUNTER0_LO,
};
pub const EVT_SQ_INSTS_LDS: CounterEvent = CounterEvent {
    name: "SQ_INSTS_LDS", block: CounterBlock::SQ, event_id: 0x0E,
    select_reg: SQ_PERFCOUNTER0_SELECT, value_lo_reg: SQ_PERFCOUNTER0_LO,
};
pub const EVT_SQ_INSTS_VMEM: CounterEvent = CounterEvent {
    name: "SQ_INSTS_VMEM", block: CounterBlock::SQ, event_id: 0x12,
    select_reg: SQ_PERFCOUNTER0_SELECT, value_lo_reg: SQ_PERFCOUNTER0_LO,
};
pub const EVT_SQ_INSTS_MFMA: CounterEvent = CounterEvent {
    name: "SQ_INSTS_MFMA", block: CounterBlock::SQ, event_id: 0x17,
    select_reg: SQ_PERFCOUNTER0_SELECT, value_lo_reg: SQ_PERFCOUNTER0_LO,
};
pub const EVT_SQ_THREAD_CYCLES: CounterEvent = CounterEvent {
    name: "SQ_THREAD_CYCLES", block: CounterBlock::SQ, event_id: 0x1A,
    select_reg: SQ_PERFCOUNTER0_SELECT, value_lo_reg: SQ_PERFCOUNTER0_LO,
};
pub const EVT_SQ_WAIT_INSTS: CounterEvent = CounterEvent {
    name: "SQ_WAIT_INSTS", block: CounterBlock::SQ, event_id: 0x1B,
    select_reg: SQ_PERFCOUNTER0_SELECT, value_lo_reg: SQ_PERFCOUNTER0_LO,
};
pub const EVT_SQ_ACTIVE_INSTS: CounterEvent = CounterEvent {
    name: "SQ_ACTIVE_INSTS", block: CounterBlock::SQ, event_id: 0x20,
    select_reg: SQ_PERFCOUNTER0_SELECT, value_lo_reg: SQ_PERFCOUNTER0_LO,
};

// GRBM events (global)
pub const EVT_GRBM_COUNT: CounterEvent = CounterEvent {
    name: "GRBM_COUNT", block: CounterBlock::GRBM, event_id: 0x01,
    select_reg: GRBM_PERFCOUNTER0_SELECT, value_lo_reg: GRBM_PERFCOUNTER0_LO,
};
pub const EVT_GRBM_GUI_ACTIVE: CounterEvent = CounterEvent {
    name: "GRBM_GUI_ACTIVE", block: CounterBlock::GRBM, event_id: 0x02,
    select_reg: GRBM_PERFCOUNTER0_SELECT, value_lo_reg: GRBM_PERFCOUNTER0_LO,
};

// TCC events (L2 cache)
pub const EVT_TCC_REQ: CounterEvent = CounterEvent {
    name: "TCC_REQ", block: CounterBlock::TCC, event_id: 0x01,
    select_reg: TCC_PERFCOUNTER0_SELECT, value_lo_reg: TCC_PERFCOUNTER0_LO,
};
pub const EVT_TCC_HIT: CounterEvent = CounterEvent {
    name: "TCC_HIT", block: CounterBlock::TCC, event_id: 0x10,
    select_reg: TCC_PERFCOUNTER0_SELECT, value_lo_reg: TCC_PERFCOUNTER0_LO,
};
pub const EVT_TCC_MISS: CounterEvent = CounterEvent {
    name: "TCC_MISS", block: CounterBlock::TCC, event_id: 0x11,
    select_reg: TCC_PERFCOUNTER0_SELECT, value_lo_reg: TCC_PERFCOUNTER0_LO,
};
pub const EVT_TCC_MC_WRITEREQ: CounterEvent = CounterEvent {
    name: "TCC_MC_WRITEREQ", block: CounterBlock::TCC, event_id: 0x12,
    select_reg: TCC_PERFCOUNTER0_SELECT, value_lo_reg: TCC_PERFCOUNTER0_LO,
};
pub const EVT_TCC_MC_READREQ: CounterEvent = CounterEvent {
    name: "TCC_MC_READREQ", block: CounterBlock::TCC, event_id: 0x13,
    select_reg: TCC_PERFCOUNTER0_SELECT, value_lo_reg: TCC_PERFCOUNTER0_LO,
};

/// Standard event set for a comprehensive profile.
pub fn standard_events() -> Vec<CounterEvent> {
    vec![
        // SQ: instruction mix + cycles + occupancy
        EVT_SQ_WAVES, EVT_SQ_INSTS, EVT_SQ_INSTS_VALU, EVT_SQ_INSTS_SALU,
        EVT_SQ_INSTS_SMEM, EVT_SQ_INSTS_FLAT, EVT_SQ_INSTS_LDS,
        EVT_SQ_INSTS_VMEM, EVT_SQ_INSTS_MFMA, EVT_SQ_THREAD_CYCLES,
        EVT_SQ_WAIT_INSTS, EVT_SQ_ACTIVE_INSTS,
        // GRBM: pipeline utilization
        EVT_GRBM_COUNT, EVT_GRBM_GUI_ACTIVE,
        // TCC: L2 cache
        EVT_TCC_REQ, EVT_TCC_HIT, EVT_TCC_MISS, EVT_TCC_MC_WRITEREQ, EVT_TCC_MC_READREQ,
    ]
}

/// Schedule events into profiling passes.
///
/// Each pass can hold at most 2 events per counter block (RELEASE_MEM reads counter 0/1).
/// The first event in a pair uses counter 0 (SELECT0), the second uses counter 1 (SELECT1).
/// Returns a Vec of passes that, when executed sequentially, collect all requested events.
pub fn schedule_passes(events: &[CounterEvent]) -> Vec<ProfilePass> {
    let sq_queue: Vec<CounterEvent> = events.iter()
        .filter(|e| e.block == CounterBlock::SQ).copied().collect();
    let grbm_queue: Vec<CounterEvent> = events.iter()
        .filter(|e| e.block == CounterBlock::GRBM).copied().collect();
    let tcc_queue: Vec<CounterEvent> = events.iter()
        .filter(|e| e.block == CounterBlock::TCC).copied().collect();

    let mut passes = Vec::new();

    let mut sq_idx = 0usize;
    let mut grbm_idx = 0usize;
    let mut tcc_idx = 0usize;

    loop {
        let sq0 = if sq_idx < sq_queue.len() {
            let e = Some(sq_queue[sq_idx].with_counter(0));
            sq_idx += 1;
            e
        } else { None };
        let sq1 = if sq_idx < sq_queue.len() {
            let e = Some(sq_queue[sq_idx].with_counter(1));
            sq_idx += 1;
            e
        } else { None };

        let grbm0 = if grbm_idx < grbm_queue.len() {
            let e = Some(grbm_queue[grbm_idx].with_counter(0));
            grbm_idx += 1;
            e
        } else { None };
        let grbm1 = if grbm_idx < grbm_queue.len() {
            let e = Some(grbm_queue[grbm_idx].with_counter(1));
            grbm_idx += 1;
            e
        } else { None };

        let tcc0 = if tcc_idx < tcc_queue.len() {
            let e = Some(tcc_queue[tcc_idx].with_counter(0));
            tcc_idx += 1;
            e
        } else { None };
        let tcc1 = if tcc_idx < tcc_queue.len() {
            let e = Some(tcc_queue[tcc_idx].with_counter(1));
            tcc_idx += 1;
            e
        } else { None };

        if sq0.is_none() && sq1.is_none() && grbm0.is_none() &&
           grbm1.is_none() && tcc0.is_none() && tcc1.is_none() {
            break;
        }

        passes.push(ProfilePass {
            sq_events: [sq0, sq1],
            grbm_events: [grbm0, grbm1],
            tcc_events: [tcc0, tcc1],
        });
    }

    passes
}

impl ProfilePass {
    /// Number of counter slots that will produce readback values in this pass.
    pub fn num_counters(&self) -> usize {
        let sq = self.sq_events.iter().filter(|e| e.is_some()).count();
        let grbm = self.grbm_events.iter().filter(|e| e.is_some()).count();
        let tcc = self.tcc_events.iter().filter(|e| e.is_some()).count();
        sq + grbm + tcc
    }

    /// All events in this pass, with their block type.
    pub fn all_events(&self) -> Vec<(CounterBlock, &CounterEvent)> {
        let mut out = Vec::new();
        for e in self.sq_events.iter().flatten()   { out.push((CounterBlock::SQ, e)); }
        for e in self.grbm_events.iter().flatten() { out.push((CounterBlock::GRBM, e)); }
        for e in self.tcc_events.iter().flatten()  { out.push((CounterBlock::TCC, e)); }
        out
    }
}
