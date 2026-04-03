#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <openssl/sha.h>

int main() {
    int iterations = 10000000;
    unsigned char input[32];
    unsigned char hash[SHA256_DIGEST_LENGTH];

    memset(input, 0x41, 32);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int i = 0; i < iterations; i++) {
        SHA256(input, 32, hash);
        // Feed output back as input for next round
        memcpy(input, hash, 32);
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    double hps = iterations / elapsed;

    printf("Iterations: %d\n", iterations);
    printf("Time: %.3f seconds\n", elapsed);
    printf("Speed: %.0f hashes/sec\n", hps);
    printf("Last hash: ");
    for (int i = 0; i < 8; i++) printf("%02x", hash[i]);
    printf("...\n");

    return 0;
}
