# aynrand

Searches on weighted graphs that keep a change when the Shannon entropy of PageRank falls. The random number generator is seeded with 1, so a run repeats. Most modes loop until you stop them.

```bash
go run . -g
```

## Modes

`-g` builds a 33-node undirected graph with symmetric integer weights from 1 to 33. Each step draws two nodes from the current PageRank and tries a directed ±1 on that link, keeping the weight at least 1. PageRank sees the inverse square of each weight, scaled by the gravitational constant.

`-e` places the same 33 nodes in three dimensions. PageRank uses e to the minus Euclidean distance. A random pair is moved so the distance between them changes by 1.

`-s` is the Snodgrass model. Link weights are 256-bit `big.Float` values. Each step draws two nodes from the live PageRank and adds a normal sample scaled by the standard deviation of the weights, clamped at 0.1. Coordinates are rebuilt by nearest-neighbor trilateration, with node 1 at the origin. A new distance waits a number of steps equal to the current distance before it replaces the live weight. PageRank uses the inverse square of each weight.

`-lm` reads `100.txt.utf-8` from the working directory and walks a directed graphical language model over `a`–`z` and space. Edge weights are bigram counts from that text. Monte Carlo tree search samples the next symbol from the outgoing link weights and keeps the directed ±1000 update whose chain has the lower PageRank entropy. It prints the symbol for each node.

`-elm` splits `100.txt.utf-8` into 64 KiB chunks. Each chunk becomes an order-2 Markov graph whose nodes are successive symbol pairs. It prints the PageRank entropy of each chunk and exits.

PageRank uses damping 0.85.

## Text

`-lm` and `-elm` expect `100.txt.utf-8` in the current directory. Letters are folded to lowercase, punctuation is dropped, and each run of whitespace becomes one space.

## Compressor

`pack` writes a lossless LZ77 encoding. The decompressor is the small C program in `pack/decoder`.

```bash
go run ./pack 100.txt.utf-8 100.txt.utf-8.ap1
gcc -Os -s -o decompress pack/decoder/decompress.c
./decompress 100.txt.utf-8.ap1 restored.txt
```

With no arguments, `go run ./pack` reads `100.txt.utf-8` and writes `100.txt.utf-8.ap1`. `.gitignore` skips that archive and the `decompress` binary.

## License

BSD. See [LICENSE](LICENSE).
