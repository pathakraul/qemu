/*
 * QEMU RISC-V Smmpt table walker
 * RISC-V Supervisor Domains Access Protection (SMMPT)
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#ifndef RISCV_SMMPT_H
#define RISCV_SMMPT_H

#include "cpu.h"
#include "exec/page-protection.h"

/*
 * smmpt_check_access() -- MPT table-walk entry point.
 *
 * @env:         hart state; env->mptmode/sdid/mptppn are the active
 *               mmpt CSR fields
 * @addr:        supervisor physical address to check
 * @prot:        PAGE_READ/WRITE/EXEC granted by matching MPTE
 * @access_type: MMU_DATA_LOAD or MMU_DATA_STORE or MMU_INST_FETCH
 */
bool smmpt_check_access(CPURISCVState *env, hwaddr addr,
                        int *prot, MMUAccessType access_type);

#endif /* RISCV_SMMPT_H */
