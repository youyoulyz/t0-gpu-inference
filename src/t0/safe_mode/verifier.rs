//! Level 2: Binary Verification — ELF/ISA validation before GPU dispatch.
//!
//! Validates compiled ELF binaries using llvm-mc disassembly to catch:
//! - Invalid instruction encodings
//! - Missing s_endpgm
//! - Problematic instruction patterns
//! - Code size limits
//!
//! # Safety Levels
//! - `fast`: basic structural checks only (~1ms)
//! - `full`: structural + llvm-mc disassembly (~10ms)
//! - `paranoid`: full + pattern matching for known hang patterns (~50ms)

use std::process::Command;

/// Verification severity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VerifyLevel {
    /// Basic structural checks: ELF magic, code size, endpgm presence.
    Fast,
    /// Fast + llvm-mc disassembly verification.
    Full,
    /// Full + pattern matching for known hang patterns.
    Paranoid,
}

/// Verification result with warnings and errors.
#[derive(Debug)]
pub struct VerifyReport {
    pub level: VerifyLevel,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
    pub disasm: Option<String>,
}

impl VerifyReport {
    pub fn is_ok(&self) -> bool { self.errors.is_empty() }

    pub fn report(&self) {
        for w in &self.warnings {
            eprintln!("[Verify WARN] {}", w);
        }
        for e in &self.errors {
            eprintln!("[Verify ERROR] {}", e);
        }
    }
}

/// Maximum ELF code size in bytes (64KB — instruction cache limit).
const MAX_CODE_SIZE: usize = 65536;

/// Verify a compiled ELF binary before GPU dispatch.
///
/// # Arguments
/// * `elf` — The compiled ELF binary (HSA code object)
/// * `level` — Verification thoroughness
///
/// # Returns
/// `VerifyReport` with errors/warnings. If `is_ok()` is false, do NOT dispatch.
pub fn verify_elf(elf: &[u8], level: VerifyLevel) -> VerifyReport {
    let mut report = VerifyReport {
        level,
        warnings: Vec::new(),
        errors: Vec::new(),
        disasm: None,
    };

    // ── Level 1: Structural checks ──
    verify_structure(elf, &mut report);

    if level == VerifyLevel::Fast || !report.errors.is_empty() {
        return report;
    }

    // ── Level 2: llvm-mc disassembly ──
    verify_with_llvm_mc(elf, &mut report);

    if level == VerifyLevel::Full || !report.errors.is_empty() {
        return report;
    }

    // ── Level 3: Pattern matching ──
    verify_patterns(&mut report);

    report
}

/// Basic structural checks on the ELF binary.
fn verify_structure(elf: &[u8], report: &mut VerifyReport) {
    // ELF magic
    if elf.len() < 4 {
        report.errors.push("ELF too small (< 4 bytes)".to_string());
        return;
    }
    if elf[0] != 0x7F || elf[1] != b'E' || elf[2] != b'L' || elf[3] != b'F' {
        report.errors.push(format!(
            "Invalid ELF magic: expected 7F 45 4C 46, got {:02X} {:02X} {:02X} {:02X}",
            elf[0], elf[1], elf[2], elf[3]
        ));
        return;
    }

    // Check for reasonable size
    if elf.len() > 10 * 1024 * 1024 {
        report.warnings.push(format!(
            "ELF is very large: {} bytes (>10MB)", elf.len()
        ));
    }

    // Look for .text section presence by scanning for common instruction patterns
    // GFX1100 s_endpgm encoding: 0xBF800000
    let endpgm_bytes = 0xBF800000u32.to_le_bytes();
    let mut found_endpgm = false;
    for i in 0..elf.len().saturating_sub(3) {
        if elf[i..i + 4] == endpgm_bytes {
            found_endpgm = true;
            break;
        }
    }
    if !found_endpgm {
        report.errors.push(
            "No s_endpgm instruction found in ELF — GPU will hang on dispatch!".to_string()
        );
    }
}

/// Use llvm-mc to disassemble and verify the .text section.
fn verify_with_llvm_mc(elf: &[u8], report: &mut VerifyReport) {
    // Extract .text section bytes (simplified: scan for instruction patterns)
    // In practice, we'd parse the ELF section headers properly.
    // For now, try to disassemble the whole file and look for errors.

    // Check if llvm-mc is available
    let has_llvm_mc = Command::new("which")
        .arg("llvm-mc")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);

    if !has_llvm_mc {
        report.warnings.push(
            "llvm-mc not found — skipping disassembly verification".to_string()
        );
        return;
    }

    // Write ELF to temp file for objdump
    let tmp_path = "/tmp/t0_verify_elf.o";
    if std::fs::write(tmp_path, elf).is_err() {
        report.warnings.push("Failed to write temp ELF for verification".to_string());
        return;
    }

    // Try objdump disassembly
    let output = Command::new("llvm-objdump")
        .args(["-d", "--mcpu=gfx1100", tmp_path])
        .output();

    match output {
        Ok(out) => {
            let disasm = String::from_utf8_lossy(&out.stdout).to_string();
            let stderr = String::from_utf8_lossy(&out.stderr).to_string();

            if !out.status.success() {
                report.errors.push(format!(
                    "llvm-objdump failed (exit {}): {}",
                    out.status.code().unwrap_or(-1),
                    stderr
                ));
                return;
            }

            // Check for disassembly errors in output
            if disasm.contains("(bad)") || disasm.contains("unknown") {
                report.errors.push(
                    "Disassembly contains '(bad)' or 'unknown' — likely invalid encoding".to_string()
                );
            }

            // Check for s_endpgm in disassembly
            if !disasm.contains("s_endpgm") {
                report.errors.push(
                    "No s_endpgm found in disassembly — GPU will hang!".to_string()
                );
            }

            // Count instructions
            let insn_count = disasm.lines()
                .filter(|l| l.contains("\t") && !l.starts_with(" "))
                .count();
            if insn_count > 4096 {
                report.warnings.push(format!(
                    "Very large kernel: {} instructions (may cause i-cache issues)", insn_count
                ));
            }

            report.disasm = Some(disasm);
        }
        Err(e) => {
            report.warnings.push(format!("Failed to run llvm-objdump: {}", e));
        }
    }

    let _ = std::fs::remove_file(tmp_path);
}

/// Pattern-matching checks on the disassembly for known hang patterns.
fn verify_patterns(report: &mut VerifyReport) {
    let disasm = match &report.disasm {
        Some(d) => d.clone(),
        None => {
            report.warnings.push(
                "No disassembly available for pattern checks".to_string()
            );
            return;
        }
    };

    // Check for s_endpgm at the very end
    let last_insn = disasm.lines()
        .rev()
        .find(|l| l.contains("\t") && !l.starts_with(" "))
        .unwrap_or("");
    if !last_insn.contains("s_endpgm") {
        report.warnings.push(
            "Last instruction is not s_endpgm — instructions may execute past kernel end".to_string()
        );
    }

    // Check for infinite loop patterns: branch to self
    for line in disasm.lines() {
        if line.contains("s_branch") || line.contains("s_cbranch") {
            // Extract target and check if it's the same address
            // This is a simplified check
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                let target = parts.last().unwrap_or(&"");
                // Check for .LBB0_N type labels pointing to same block
                if target.starts_with(".L") && line.contains(target) {
                    report.errors.push(format!(
                        "Possible infinite loop detected: {}", line
                    ));
                }
            }
        }
    }

    // Check for v_add_co without prior vcc clear in loop context
    // (simplified: just warn if we see v_add_co)
    let add_co_count = disasm.matches("v_add_co_u32").count();
    if add_co_count > 0 {
        report.warnings.push(format!(
            "Found {} v_add_co_u32 instructions — verify VCC is cleared before loops",
            add_co_count
        ));
    }
}

/// Verify a T0 IR Op sequence (pre-compilation) for known issues.
/// This is a lightweight check that doesn't require ELF compilation.
pub fn verify_ops_lightweight(ops: &[super::super::ir::Op]) -> VerifyReport {
    let mut report = VerifyReport {
        level: VerifyLevel::Fast,
        warnings: Vec::new(),
        errors: Vec::new(),
        disasm: None,
    };

    use super::super::ir::Op;

    // Check for s_endpgm
    let has_endpgm = ops.iter().any(|op| matches!(op, Op::Endpgm));
    if !has_endpgm {
        report.errors.push("No s_endpgm in Op sequence — GPU will hang".to_string());
    }

    // Check for dead code after endpgm
    let mut seen_endpgm = false;
    for (i, op) in ops.iter().enumerate() {
        if seen_endpgm && !matches!(op, Op::Label(_)) {
            report.warnings.push(format!(
                "Op[{}]: Instruction after s_endpgm is unreachable", i
            ));
        }
        if matches!(op, Op::Endpgm) {
            seen_endpgm = true;
        }
    }

    // Check EXEC mask balance
    let mut exec_depth: i32 = 0;
    for (i, op) in ops.iter().enumerate() {
        match op {
            Op::SaveExec { .. } => exec_depth += 1,
            Op::RestoreExec { .. } => exec_depth -= 1,
            _ => {}
        }
        if exec_depth < 0 {
            report.errors.push(format!(
                "Op[{}]: RestoreExec without matching SaveExec (depth={})", i, exec_depth
            ));
        }
    }
    if exec_depth > 0 {
        report.errors.push(format!(
            "Unbalanced SaveExec: {} more SaveExec than RestoreExec", exec_depth
        ));
    }

    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verify_elf_magic() {
        let bad_elf = vec![0x00, 0x00, 0x00, 0x00];
        let report = verify_elf(&bad_elf, VerifyLevel::Fast);
        assert!(!report.is_ok());
        assert!(report.errors.iter().any(|e| e.contains("magic")));
    }

    #[test]
    fn test_verify_elf_no_endpgm() {
        // Valid ELF header but no s_endpgm
        let mut elf = vec![0x7F, b'E', b'L', b'F'];
        elf.resize(64, 0);
        let report = verify_elf(&elf, VerifyLevel::Fast);
        assert!(!report.is_ok());
        assert!(report.errors.iter().any(|e| e.contains("s_endpgm")));
    }

    #[test]
    fn test_verify_elf_with_endpgm() {
        let mut elf = vec![0x7F, b'E', b'L', b'F'];
        elf.resize(64, 0);
        // Append s_endpgm encoding
        elf.extend_from_slice(&0xBF800000u32.to_le_bytes());
        let report = verify_elf(&elf, VerifyLevel::Fast);
        // Should pass structural check (no endpgm error)
        assert!(!report.errors.iter().any(|e| e.contains("s_endpgm")));
    }

    #[test]
    fn test_verify_ops_lightweight_no_endpgm() {
        use super::super::super::ir::*;
        let ops = vec![
            Op::VAddF32 { dst: VReg(0), src0: Operand::VReg(VReg(1)), src1: Operand::VReg(VReg(2)) },
        ];
        let report = verify_ops_lightweight(&ops);
        assert!(!report.is_ok());
        assert!(report.errors.iter().any(|e| e.contains("s_endpgm")));
    }

    #[test]
    fn test_verify_ops_unbalanced_exec() {
        use super::super::super::ir::*;
        let ops = vec![
            Op::SaveExec { dst: SReg(10) },
            Op::SaveExec { dst: SReg(11) },
            Op::RestoreExec { src: SReg(10) },
            // Missing RestoreExec for SReg(11)
            Op::Endpgm,
        ];
        let report = verify_ops_lightweight(&ops);
        assert!(!report.is_ok());
        assert!(report.errors.iter().any(|e| e.contains("Unbalanced")));
    }
}
