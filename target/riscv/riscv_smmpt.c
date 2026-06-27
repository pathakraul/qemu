/*
 * QEMU RISC-V Smmpt table walker
 * RISC-V Supervisor Domains Access Protection (SMMPT)
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "riscv_smmpt.h"
#include "cpu_bits.h"
#include "tcg/pmp.h"
#include "system/memory.h"

#define MPTE_V              BIT_ULL(0)   /* valid */
#define MPTE_L              BIT_ULL(1)   /* leaf */
#define MPTE_N              BIT_ULL(2)   /* NAPOT (leaf only) */

static inline bool mpte_valid(uint64_t m)
{
    return m & MPTE_V;
}

static inline bool mpte_leaf(uint64_t m)
{
    return m & MPTE_L;
}

static inline bool mpte_napot(uint64_t m)
{
    return m & MPTE_N;
}

/*
 * Non-leaf MPTE PPN field (RV32 [31:10], RV64 [53:10]).
 */
#define MPTE_NL_PPN_SHIFT        10
#define MPTE_NL_PPN_MASK_32      MAKE_64BIT_MASK(10, 22)
#define MPTE_NL_RSV_MASK_32      MAKE_64BIT_MASK(2, 8)
#define MPTE_NL_PPN_MASK_64      MAKE_64BIT_MASK(10, 44)
#define MPTE_NL_RSV_MASK_64      (MAKE_64BIT_MASK(2, 8) | MAKE_64BIT_MASK(54, 10))

static inline uint64_t mpte_nl_ppn(uint64_t mpte, bool rv32)
{
    uint64_t mask = rv32 ? MPTE_NL_PPN_MASK_32 : MPTE_NL_PPN_MASK_64;
    return (mpte & mask) >> MPTE_NL_PPN_SHIFT;
}

static inline bool mpte_nl_reserved(uint64_t mpte, bool rv32)
{
    return mpte & (rv32 ? MPTE_NL_RSV_MASK_32 : MPTE_NL_RSV_MASK_64);
}

/*
 * Non-NAPOT leaf XWR tuple extraction.
 * Tuple i occupies bits [8+3i+2 : 8+3i]. Uniform across all schemes.
 * 8 tuples in Smmpt34 and 16 tuples in Smmpt43/52/64.
 */
static inline uint32_t mpte_xwr_tuple(uint64_t mpte, unsigned pi)
{
    return (mpte >> (8 + pi * 3)) & 0x7;
}

#define MPTE_LEAF_RSV_LOW        MAKE_64BIT_MASK(3, 5)   /* [7:3] */
#define MPTE_LEAF_RSV_HIGH64     MAKE_64BIT_MASK(56, 8)  /* [63:56] RV64 only */

static inline bool mpte_nonnapot_reserved(uint64_t mpte, bool rv32)
{
    return (mpte & MPTE_LEAF_RSV_LOW) || (!rv32 && (mpte & MPTE_LEAF_RSV_HIGH64));
}

/*
 * NAPOT leaf fields
 */
#define MPTE_NAPOT_XWR_SHIFT     8
#define MPTE_NAPOT_XWR_MASK      MAKE_64BIT_MASK(8, 3)
#define MPTE_NAPOT_ZERO          BIT_ULL(11)
#define MPTE_NAPOT_G_SHIFT       12
#define MPTE_NAPOT_G_MASK        MAKE_64BIT_MASK(12, 4)
#define MPTE_NAPOT_RSV_LOW       MAKE_64BIT_MASK(3, 5)

static inline uint32_t mpte_napot_xwr(uint64_t mpte)
{
    return (mpte & MPTE_NAPOT_XWR_MASK) >> MPTE_NAPOT_XWR_SHIFT;
}

static inline uint32_t mpte_napot_g(uint64_t mpte)
{
    return (mpte & MPTE_NAPOT_G_MASK) >> MPTE_NAPOT_G_SHIFT;
}

static inline bool mpte_napot_reserved(uint64_t mpte, bool rv32)
{
    uint64_t rsv = mpte & (MPTE_NAPOT_RSV_LOW | MPTE_NAPOT_ZERO);
    rsv |= mpte & (rv32 ? MAKE_64BIT_MASK(16, 16) : MAKE_64BIT_MASK(16, 48));
    return rsv != 0;
}

typedef enum {
    SMMPT_34,
    SMMPT_43,
    SMMPT_52,
    SMMPT_64
} SmmptScheme;

#define SMMPT_PGSHIFT       12

/*
 * pn[level] extraction.
 */
static uint64_t smmpt_pn(SmmptScheme s, hwaddr addr, int level)
{
    if (s == SMMPT_34) {
        return (level == 0) ? extract64(addr, 15, 10)
                            : extract64(addr, 25,  9);
    }
    if (s == SMMPT_64 && level == 4) {
        return extract64(addr, 52, 12);
    }
    return extract64(addr, 16 + level * 9, 9);
}

/*
 * pi: tuple index for non-NAPOT leaf.
 * level>0: top NUMPGINRANGE bits of pn[level-1]
 * level=0: top NUMPGINRANGE bits of range offset
 */
static uint64_t smmpt_pi(SmmptScheme s, hwaddr addr, int level)
{
    if (s == SMMPT_34) {
        /* NUMPGINRANGE = 3 */
        return (level == 0) ? extract64(addr, 12, 3)
                            : extract64(addr, 22, 3);
    }
    /* Smmpt43/52/64: NUMPGINRANGE = 4 */
    return (level == 0) ? extract64(addr, 12, 4)
                        : extract64(addr, 9 * level + 12, 4);
}

static const struct {
    int levels;
    int mptesize;
    int pa_bits;            /* SPA width; bits at/above must be 0 */
    uint32_t napot_g_legal;
} smmpt_geo[] = {
    [SMMPT_34] = {2, 4, 34, 6},
    [SMMPT_43] = {3, 8, 43, 4},
    [SMMPT_52] = {4, 8, 52, 4},
    [SMMPT_64] = {5, 8, 64, 4},
};

static SmmptScheme scheme_from_mode(uint32_t mode, bool rv32)
{
    if (rv32) {
        return SMMPT_34;
    }
    switch (mode) {
    case MMPT_MODE_SMMPT43: return SMMPT_43;
    case MMPT_MODE_SMMPT52: return SMMPT_52;
    case MMPT_MODE_SMMPT64: return SMMPT_64;
    default: g_assert_not_reached();
    }
}

/* 
 * Table walk (steps 1-7 in the smmtt spec)
 */

bool smmpt_check_access(CPURISCVState *env, hwaddr addr,
                        int *prot, MMUAccessType access_type)
{
    bool rv32 = (riscv_cpu_mxl(env) == MXL_RV32);
    SmmptScheme scheme = scheme_from_mode(env->mptmode, rv32);
    CPUState *cs = env_cpu(env);
    MemTxAttrs attrs = MEMTXATTRS_UNSPECIFIED;
    MemTxResult res;
    hwaddr base = (hwaddr)env->mptppn << SMMPT_PGSHIFT;
    int pmp_prot;
    uint32_t xwr = 0;
    bool allowed = false;

    /* Small Self Test (remove after verifying) which access
     * the firmware region PA */
    static bool selftest_done = false;
    if (!selftest_done) {
        selftest_done = true;
        int p;
        bool fw   = smmpt_check_access(env, 0x80010000, &p, MMU_DATA_LOAD);
        bool kern = smmpt_check_access(env, 0x80200000, &p, MMU_INST_FETCH);
        printf("QEMU: SELFTEST fw(0x80010000)=%s  kernel(0x80200000)=%s\n",
               fw   ? "PERMIT(BUG!)" : "DENY(ok)",
               kern ? "PERMIT(ok)"   : "DENY(BUG!)");
    }
    /* ---- end self-test ---- */
    /*
     * The SPA which is out of range for any mode may cause aliasing
     * since higher bits are not considered and the lower which are under
     * range may alias to a lower in-range permissions.
     */
    int pa_bits = smmpt_geo[scheme].pa_bits;
    if (pa_bits < 64 && (addr >> pa_bits) != 0) {
        allowed = false;
        goto out;
    }

    /* Step 1: Calculate a = mmpt.PPN * PAGESIZE, i = LEVELS-1 */
    for (int i = smmpt_geo[scheme].levels - 1; i >= 0; i--) {
        uint64_t pn      = smmpt_pn(scheme, addr, i);
        hwaddr   mpte_pa = base + pn * smmpt_geo[scheme].mptesize;
        uint64_t mpte;

        /*
         * Step 2: PMP check all implicit MPT accesses as Effective M-mode.
         */
        if (get_physical_address_pmp(env, &pmp_prot, mpte_pa,
                                     smmpt_geo[scheme].mptesize,
                                     MMU_DATA_LOAD, PRV_M) != TRANSLATE_SUCCESS) {
            allowed = false;
            goto out;
        }

        /* Load MPTE */
        if (smmpt_geo[scheme].mptesize == 4) {
            mpte = address_space_ldl_le(cs->as, mpte_pa, attrs, &res);
        }
        else {
            mpte = address_space_ldq_le(cs->as, mpte_pa, attrs, &res);
        }

        if (res != MEMTX_OK) {
            allowed = false;
            goto out;
        }

        /* Step 3: Is MPTE valid or is any reserved bits accessed.*/
        if (!mpte_valid(mpte)) {
            allowed = false;
            goto out;
        }

        if (!mpte_leaf(mpte) && mpte_napot(mpte)) {
            allowed = false;
            goto out;
        }

        if (!mpte_leaf(mpte)) {
            /* Step 4: Is it non-leaf and check for reserved bits */
            if (mpte_nl_reserved(mpte, rv32)) {
                allowed = false;
                goto out;
            }

            if (i == 0) {
                allowed = false;
                goto out;
            }

            base = mpte_nl_ppn(mpte, rv32) << SMMPT_PGSHIFT;
            continue;
        }

        if (mpte_napot(mpte)) {
            /* Step 6: Is it a NAPOT leaf */
            if (mpte_napot_reserved(mpte, rv32)) {
                allowed = false;
                goto out;
            }

            if (mpte_napot_g(mpte) != smmpt_geo[scheme].napot_g_legal) {
                allowed = false;
                goto out;
            }

            xwr = mpte_napot_xwr(mpte);

        } else {
            /* Step 5: Is it a non-NAPOT leaf */
            if (mpte_nonnapot_reserved(mpte, rv32)) {
                allowed = false;
                goto out;
            }

            xwr = mpte_xwr_tuple(mpte, (unsigned)smmpt_pi(scheme, addr, i));
        }

        /* Step 7: check XWR against access_type */
        switch (xwr) {
            case 0:
                allowed = false;
                break; /* No access */
            case 1:
                *prot = PAGE_READ;
                allowed = (access_type == MMU_DATA_LOAD);
                break;
            case 2:
                allowed = false;
                break; /* reserved */
            case 3:
                *prot = PAGE_READ | PAGE_WRITE;
                allowed = (access_type == MMU_DATA_LOAD || access_type == MMU_DATA_STORE);
                break;
            case 4:  *prot = PAGE_EXEC;
                allowed = (access_type == MMU_INST_FETCH);
                break;
            case 5:
                *prot = PAGE_READ | PAGE_EXEC;
                allowed = (access_type == MMU_DATA_LOAD || 
                access_type == MMU_INST_FETCH);
                break;
            case 6:
                allowed = false;
                break; /* reserved */
            case 7:
                *prot = PAGE_READ | PAGE_WRITE | PAGE_EXEC;
                allowed = true;
                break;
            default: g_assert_not_reached();
        }

        goto out;
    }

    /* Fell out of the loop without reaching a leaf: deny. */
    allowed = false;

out:
    //printf("QEMU: smmpt_check_access: addr=0x%lx mode=%u sdid=%u mptppn=0x%lx "
    //       "xwr=%x accesstype=%d -> %s\n",
    //       (unsigned long)addr, env->mptmode, env->sdid,
    //       (unsigned long)env->mptppn, xwr, access_type,
    //       allowed ? "PERMIT" : "DENY");
    return allowed;
}
