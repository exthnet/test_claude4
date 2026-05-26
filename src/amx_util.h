#ifndef AMX_UTIL_H
#define AMX_UTIL_H

#include <stdio.h>
#include <stdlib.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <stdint.h>
#include <string.h>

// AMX permission request flags (Linux kernel >= 5.16)
#ifndef ARCH_GET_XCOMP_PERM
#define ARCH_GET_XCOMP_PERM 0x1022
#endif
#ifndef ARCH_REQ_XCOMP_PERM
#define ARCH_REQ_XCOMP_PERM 0x1023
#endif
#ifndef XFEATURE_XTILECFG
#define XFEATURE_XTILECFG 17
#endif
#ifndef XFEATURE_XTILEDATA
#define XFEATURE_XTILEDATA 18
#endif

static inline int amx_request_permission(void) {
    long rc = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
    if (rc != 0) {
        fprintf(stderr, "ERROR: arch_prctl(ARCH_REQ_XCOMP_PERM, XTILEDATA) failed: rc=%ld errno-like\n", rc);
        return -1;
    }
    return 0;
}

// AMX tile configuration. 8 tiles, each up to 16 rows x 64 bytes (1024B).
typedef struct __attribute__((packed, aligned(64))) {
    uint8_t  palette;
    uint8_t  start_row;
    uint8_t  reserved0[14];
    uint16_t colsb[16];   // bytes per row for each tile (0..15)
    uint8_t  rows[16];
} amx_tilecfg_t;

#endif
