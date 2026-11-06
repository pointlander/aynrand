/* Copyright 2026 The Ayn Rand Authors. All rights reserved.
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE file.
 *
 * ap1 decompressor. Byte-for-byte inverse of pack/compress.go.
 *   gcc -Os -s -o decompress pack/decoder/decompress.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static int get(FILE *f) {
	int c = fgetc(f);
	if (c < 0) {
		fprintf(stderr, "truncated\n");
		exit(1);
	}
	return c;
}

int main(int argc, char **argv) {
	FILE *in, *out;
	unsigned char mag[4];
	unsigned char *buf;
	uint64_t size = 0, n = 0;
	int flags = 0, nbit = 0;
	int i;

	if (argc != 3) {
		fprintf(stderr, "usage: decompress input output\n");
		return 1;
	}
	in = fopen(argv[1], "rb");
	if (!in) {
		perror(argv[1]);
		return 1;
	}
	if (fread(mag, 1, 4, in) != 4 || memcmp(mag, "ap1\n", 4) != 0) {
		fprintf(stderr, "bad magic\n");
		return 1;
	}
	for (i = 0; i < 8; i++)
		size |= (uint64_t)get(in) << (8 * i);
	buf = malloc(size ? size : 1);
	if (!buf) {
		fprintf(stderr, "out of memory\n");
		return 1;
	}
	while (n < size) {
		unsigned len, dist;
		if (nbit == 0) {
			flags = get(in);
			nbit = 8;
		}
		if (flags & 1) {
			buf[n++] = (unsigned char)get(in);
		} else {
			dist = (unsigned)get(in) | ((unsigned)get(in) << 8);
			dist += 1;
			len = (unsigned)get(in) + 3;
			if (dist > n || n + len > size) {
				fprintf(stderr, "bad match\n");
				return 1;
			}
			for (i = 0; i < (int)len; i++)
				buf[n + i] = buf[n - dist + i];
			n += len;
		}
		flags >>= 1;
		nbit--;
	}
	out = fopen(argv[2], "wb");
	if (!out) {
		perror(argv[2]);
		return 1;
	}
	if (fwrite(buf, 1, size, out) != size) {
		fprintf(stderr, "write failed\n");
		return 1;
	}
	fclose(out);
	fclose(in);
	free(buf);
	return 0;
}
