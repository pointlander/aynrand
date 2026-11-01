// Copyright 2026 The Ayn Rand Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// A 33-node undirected graph starts with symmetric integer weights drawn from
// 1..33. Each step picks a random pair and tries +1 on that link. The change
// is kept when it does not raise both endpoints' PageRank (one staying flat
// or falling is enough). When +1 does raise both, the link is set one below
// its value at the start of the step, and that decrease is kept when it
// likewise fails to raise both. Otherwise the weight is put back.
package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

const (
	nodeCount = 33
	minWeight = 1
	maxWeight = 33
	damping   = 0.85
	tolerance = 1e-15
	maxIter   = 10000
)

func main() {
	seed := time.Now().UnixNano()
	rng := rand.New(rand.NewSource(seed))

	weights := randomSymmetricWeights(rng)
	ranks := pageRank(weights)

	fmt.Printf("seed %d\n", seed)
	fmt.Printf("33-node undirected graph, symmetric weights uniform on %d..%d, damping %g\n", minWeight, maxWeight, damping)
	fmt.Println("initial")
	printRanks(ranks)

	for step := 1; ; step++ {
		a, b := randomPair(rng)
		before := weights[a][b]
		beforeRanks := ranks

		setWeight(weights, a, b, before+1)
		plus := pageRank(weights)
		plusBoth := raisedBoth(plus, beforeRanks, a, b)
		if !plusBoth {
			setWeight(weights, a, b, before)
		} else {
			fmt.Println("++++++++++++++++++++++++++++++++++")
			fmt.Printf("step %d nodes %d,%d +1 weight %d -> %d both_increased %t\n", step, a+1, b+1, before, before+1, plusBoth)
			printRanks(plus)
			ranks = plus
			continue
		}

		if before-1 < 1 {
			//setWeight(weights, a, b, before)
			//fmt.Printf("step %d nodes %d,%d weight %d restored\n", step, a+1, b+1, before)
			//printRanks(beforeRanks)
			continue
		}

		setWeight(weights, a, b, before-1)
		minus := pageRank(weights)
		minusBoth := raisedBoth(minus, beforeRanks, a, b)
		if !minusBoth {
			setWeight(weights, a, b, before)
			continue
		} else {
			fmt.Println("----------------------------------")
			fmt.Printf("step %d nodes %d,%d -1 weight %d -> %d both_increased %t\n", step, a+1, b+1, before, before-1, minusBoth)
			printRanks(minus)
			ranks = minus
		}

		//fmt.Printf("step %d nodes %d,%d weight %d restored\n", step, a+1, b+1, before)
		//printRanks(beforeRanks)
	}
}

func randomSymmetricWeights(rng *rand.Rand) [][]int {
	weights := make([][]int, nodeCount)
	for i := range weights {
		weights[i] = make([]int, nodeCount)
	}
	for i := 0; i < nodeCount; i++ {
		for j := i + 1; j < nodeCount; j++ {
			weights[i][j] = rng.Intn(maxWeight-minWeight+1) + minWeight
			weights[j][i] = weights[i][j]
		}
	}
	return weights
}

func randomPair(rng *rand.Rand) (int, int) {
	a := rng.Intn(nodeCount)
	b := rng.Intn(nodeCount - 1)
	if b >= a {
		b++
	}
	if a > b {
		a, b = b, a
	}
	return a, b
}

func setWeight(weights [][]int, a, b, value int) {
	weights[a][b] = value
	weights[b][a] = value
}

func raisedBoth(next, prev []float64, a, b int) bool {
	aa := 0.0
	for _, value := range next {
		aa += value * math.Log(value)
	}
	aa = -aa
	bb := 0.0
	for _, value := range prev {
		bb += value * math.Log(value)
	}
	bb = -bb
	return aa < bb
	//return next[a] > prev[a] && next[b] > prev[b]
}

func printRanks(ranks []float64) {
	sum := 0.0
	for i, rank := range ranks {
		if i > 0 {
			fmt.Print(" ")
		}
		fmt.Printf("%d:%.8f", i+1, rank)
		sum += rank
	}
	fmt.Printf(" sum:%.8f\n", sum)
}

// pageRank is the damped random walk on an undirected weighted graph.
// Transition chance along an edge is proportional to its weight. A node with
// no remaining weight spreads its rank evenly, which is the usual dangling-node rule.
func pageRank(weights [][]int) []float64 {
	n := len(weights)
	out := make([]float64, n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			out[i] += float64(weights[i][j])
		}
	}

	ranks := make([]float64, n)
	next := make([]float64, n)
	uniform := 1.0 / float64(n)
	for i := range ranks {
		ranks[i] = uniform
	}
	base := (1 - damping) * uniform

	for iter := 0; iter < maxIter; iter++ {
		for i := range next {
			next[i] = base
		}
		dangling := 0.0
		for j := 0; j < n; j++ {
			if out[j] == 0 {
				dangling += ranks[j]
				continue
			}
			scale := damping * ranks[j] / out[j]
			row := weights[j]
			for i := 0; i < n; i++ {
				if row[i] != 0 {
					next[i] += scale * float64(row[i])
				}
			}
		}
		if dangling != 0 {
			add := damping * dangling * uniform
			for i := range next {
				next[i] += add
			}
		}

		delta := 0.0
		for i := 0; i < n; i++ {
			diff := math.Abs(next[i] - ranks[i])
			if diff > delta {
				delta = diff
			}
			ranks[i] = next[i]
		}
		if delta < tolerance {
			break
		}
	}
	return ranks
}
