// Copyright 2026 The Ayn Rand Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Command pack writes a lossless LZ77 encoding of a file.
// The stream is magic "ap1\n", the original size as a little-endian uint64,
// then tokens packed eight to a flag byte, least significant bit first.
// A 1 bit is followed by one literal byte. A 0 bit is followed by a
// little-endian uint16 distance minus one and one byte of length minus
// three. Distance is 1..65535 and length is 3..258.
package main

import (
	"encoding/binary"
	"fmt"
	"os"
)

const (
	apMagic    = "ap1\n"
	apWindow   = 65535
	apMinMatch = 3
	apMaxMatch = apMinMatch + 255
	apChain    = 2048
)

func main() {
	in := "100.txt.utf-8"
	out := in + ".ap1"
	if len(os.Args) > 1 {
		in = os.Args[1]
	}
	if len(os.Args) > 2 {
		out = os.Args[2]
	}
	src, err := os.ReadFile(in)
	if err != nil {
		fmt.Fprintf(os.Stderr, "read %s: %v\n", in, err)
		os.Exit(1)
	}
	packed := Compress(src)
	if err := os.WriteFile(out, packed, 0644); err != nil {
		fmt.Fprintf(os.Stderr, "write %s: %v\n", out, err)
		os.Exit(1)
	}
	fmt.Printf("%s %d bytes -> %s %d bytes\n", in, len(src), out, len(packed))
}

// Compress encodes src with a 64 KiB-window LZ77 search.
func Compress(src []byte) []byte {
	n := len(src)
	head := make(map[uint32]int, n/8)
	prev := make([]int, n)
	for i := range prev {
		prev[i] = -1
	}
	key := func(i int) uint32 {
		return uint32(src[i]) | uint32(src[i+1])<<8 | uint32(src[i+2])<<16
	}
	insert := func(i int) {
		if i+2 >= n {
			return
		}
		k := key(i)
		if p, ok := head[k]; ok {
			prev[i] = p
		} else {
			prev[i] = -1
		}
		head[k] = i
	}
	find := func(i int) (dist, length int) {
		if i+apMinMatch > n {
			return 0, 0
		}
		k := key(i)
		p, ok := head[k]
		if !ok {
			return 0, 0
		}
		limit := apMaxMatch
		if limit > n-i {
			limit = n - i
		}
		for steps := 0; p >= 0 && steps < apChain; p, steps = prev[p], steps+1 {
			d := i - p
			if d <= 0 || d > apWindow {
				break
			}
			l := 0
			for l < limit && src[p+l] == src[i+l] {
				l++
			}
			if l > length {
				dist, length = d, l
				if l == limit {
					break
				}
			}
		}
		if length < apMinMatch {
			return 0, 0
		}
		return dist, length
	}

	out := make([]byte, 0, n/2)
	var hdr [12]byte
	copy(hdr[:4], apMagic)
	binary.LittleEndian.PutUint64(hdr[4:], uint64(n))
	out = append(out, hdr[:]...)

	flagAt := -1
	nbit := 0
	bit := func(literal bool) {
		if nbit == 0 {
			flagAt = len(out)
			out = append(out, 0)
		}
		if literal {
			out[flagAt] |= 1 << nbit
		}
		nbit++
		if nbit == 8 {
			nbit = 0
		}
	}
	emitLit := func(b byte) {
		bit(true)
		out = append(out, b)
	}
	emitMatch := func(dist, length int) {
		bit(false)
		var buf [3]byte
		binary.LittleEndian.PutUint16(buf[0:2], uint16(dist-1))
		buf[2] = byte(length - apMinMatch)
		out = append(out, buf[:]...)
	}

	for i := 0; i < n; {
		dist, length := find(i)
		if length >= apMinMatch && i+1 < n {
			if _, length2 := find(i + 1); length2 > length {
				emitLit(src[i])
				insert(i)
				i++
				continue
			}
		}
		if length >= apMinMatch {
			emitMatch(dist, length)
			for j := 0; j < length; j++ {
				insert(i + j)
			}
			i += length
			continue
		}
		emitLit(src[i])
		insert(i)
		i++
	}
	return out
}
