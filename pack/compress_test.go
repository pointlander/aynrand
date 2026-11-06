package main

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

func TestCompressRoundTrip(t *testing.T) {
	samples := [][]byte{
		nil,
		[]byte("a"),
		[]byte("aaaaaaa"),
		[]byte("abcdefghijklmnopqrstuvwxyz"),
		bytes.Repeat([]byte("To be, or not to be, that is the question.\n"), 50),
	}
	for _, src := range samples {
		if got := decodeAP1(t, Compress(src)); !bytes.Equal(got, src) {
			t.Fatalf("round trip failed for %q", src)
		}
	}
}

func TestShakespeareRoundTrip(t *testing.T) {
	src, err := os.ReadFile(filepath.Join("..", "100.txt.utf-8"))
	if err != nil {
		t.Fatal(err)
	}
	packed := Compress(src)
	dir := t.TempDir()
	bin := filepath.Join(dir, "decompress")
	cmd := exec.Command("gcc", "-Os", "-s", "-o", bin, "decoder/decompress.c")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("gcc: %v\n%s", err, out)
	}
	in := filepath.Join(dir, "in.ap1")
	out := filepath.Join(dir, "out.txt")
	if err := os.WriteFile(in, packed, 0644); err != nil {
		t.Fatal(err)
	}
	cmd = exec.Command(bin, in, out)
	if msg, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("decompress: %v\n%s", err, msg)
	}
	got, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, src) {
		t.Fatalf("decompressed %d bytes, original %d", len(got), len(src))
	}
	info, err := os.Stat(bin)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("original %d compressed %d decompressor %d", len(src), len(packed), info.Size())
}

func decodeAP1(t *testing.T, packed []byte) []byte {
	t.Helper()
	if len(packed) < 12 || string(packed[:4]) != apMagic {
		t.Fatalf("bad header %q", packed)
	}
	size := int(uint64(packed[4]) | uint64(packed[5])<<8 | uint64(packed[6])<<16 | uint64(packed[7])<<24 |
		uint64(packed[8])<<32 | uint64(packed[9])<<40 | uint64(packed[10])<<48 | uint64(packed[11])<<56)
	buf := make([]byte, size)
	p := 12
	n := 0
	nbit := 0
	flags := 0
	get := func() byte {
		t.Helper()
		if p >= len(packed) {
			t.Fatal("truncated")
		}
		b := packed[p]
		p++
		return b
	}
	for n < size {
		if nbit == 0 {
			flags = int(get())
			nbit = 8
		}
		if flags&1 != 0 {
			buf[n] = get()
			n++
		} else {
			dist := int(get()) | int(get())<<8
			dist++
			length := int(get()) + apMinMatch
			if dist > n || n+length > size {
				t.Fatalf("bad match dist %d len %d at %d", dist, length, n)
			}
			for i := 0; i < length; i++ {
				buf[n+i] = buf[n-dist+i]
			}
			n += length
		}
		flags >>= 1
		nbit--
	}
	return buf
}
