// Copyright 2026 The Ayn Rand Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// -g perturbs one integer link weight by ±1.
// -e gives every node an (x, y, z) coordinate. PageRank uses e to the minus
// Euclidean distance as the link weight. A random pair is moved so that
// distance changes by 1. Either model keeps a proposal when the Shannon
// entropy of PageRank decreases.
package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
)

const (
	minWeight = 1
	maxWeight = 33
	damping   = 0.85
	tolerance = 1e-15
	maxIter   = 10000
)

var nodeCount = 33

var (
	// FlagGraphical graphical model
	FlagGraphical = flag.Bool("g", false, "graphical model")
	// FlagEuclidean euclidean graphical model
	FlagEuclidean = flag.Bool("e", false, "euclidean graphical model")
)

func main() {
	flag.Parse()
	if *FlagEuclidean {
		nodeCount *= 10
		euclidean()
		return
	}
	if *FlagGraphical {
		graphical()
		return
	}
}

// graphical model
func graphical() {
	rng := rand.New(rand.NewSource(1))

	weights := randomSymmetricWeights(rng)
	ranks := pageRank(toFloat(weights))

	fmt.Printf("33-node undirected graph, symmetric weights uniform on %d..%d, damping %g\n", minWeight, maxWeight, damping)
	fmt.Println("initial")
	printRanks(ranks)

	for step := 1; ; step++ {
		a, b := randomPair(rng)
		before := weights[a][b]
		beforeRanks := ranks

		add := func() bool {
			setWeight(weights, a, b, before+1)
			plus := pageRank(toFloat(weights))
			plusBoth := raisedBoth(plus, beforeRanks, a, b)
			if plusBoth {
				fmt.Println("++++++++++++++++++++++++++++++++++")
				fmt.Printf("step %d nodes %d,%d +1 weight %d -> %d both_increased %t\n", step, a+1, b+1, before, before+1, plusBoth)
				printRanks(plus)
				ranks = plus
				return true
			}
			setWeight(weights, a, b, before)
			return false
		}

		subtract := func() bool {
			if before-1 < 1 {
				return false
			}

			setWeight(weights, a, b, before-1)
			minus := pageRank(toFloat(weights))
			minusBoth := raisedBoth(minus, beforeRanks, a, b)
			if minusBoth {
				fmt.Println("----------------------------------")
				fmt.Printf("step %d nodes %d,%d -1 weight %d -> %d both_increased %t\n", step, a+1, b+1, before, before-1, minusBoth)
				printRanks(minus)
				ranks = minus
				return true
			}
			setWeight(weights, a, b, before)
			return false
		}

		if rng.Intn(2) == 0 {
			if !add() {
				subtract()
			}
		} else {
			if !subtract() {
				add()
			}
		}
	}
}

type point struct {
	x, y, z float64
}

// euclidean is the graphical model with each node placed in R^3.
// PageRank and its entropy see link weight e^{-distance}. Moving a pair
// closer or farther by 1 is the stand-in for a ±1 distance edit; every other
// distance changes with them because the points share one embedding.
func euclidean() {
	rng := rand.New(rand.NewSource(1))

	points := make([]point, nodeCount)
	span := float64(maxWeight - minWeight)
	for i := range points {
		points[i] = point{
			x: float64(minWeight) + rng.Float64()*span,
			y: float64(minWeight) + rng.Float64()*span,
			z: float64(minWeight) + rng.Float64()*span,
		}
	}
	ranks := pageRank(linkWeights(points))

	fmt.Printf("33-node euclidean graph, coordinates uniform on [%d, %d), link weight is e^-distance, damping %g\n", minWeight, maxWeight, damping)
	printPoints(points)
	fmt.Println("initial")
	printRanks(ranks)

	for step := 1; ; step++ {
		a, b := randomPair(rng)
		pa, pb := points[a], points[b]
		beforeDist := distance(pa, pb)

		add := func() bool {
			shiftPair(points, a, b, +1)
			plus := pageRank(linkWeights(points))
			if raisedBoth(plus, ranks, a, b) {
				afterDist := distance(points[a], points[b])
				fmt.Println("++++++++++++++++++++++++++++++++++")
				fmt.Printf("step %d nodes %d,%d distance %.6f -> %.6f weight %.6e -> %.6e\n", step, a+1, b+1, beforeDist, afterDist, math.Exp(-beforeDist), math.Exp(-afterDist))
				printPoint(a, points[a])
				printPoint(b, points[b])
				printRanks(plus)
				ranks = plus
				return true
			}
			points[a], points[b] = pa, pb
			return false
		}

		subtract := func() bool {
			if beforeDist-1 < 1 {
				return false
			}
			shiftPair(points, a, b, -1)
			minus := pageRank(linkWeights(points))
			if raisedBoth(minus, ranks, a, b) {
				afterDist := distance(points[a], points[b])
				fmt.Println("----------------------------------")
				fmt.Printf("step %d nodes %d,%d distance %.6f -> %.6f weight %.6e -> %.6e\n", step, a+1, b+1, beforeDist, afterDist, math.Exp(-beforeDist), math.Exp(-afterDist))
				printPoint(a, points[a])
				printPoint(b, points[b])
				printRanks(minus)
				ranks = minus
				return true
			}
			points[a], points[b] = pa, pb
			return false
		}

		if rng.Intn(2) == 0 {
			if !add() {
				subtract()
			}
		} else {
			if !subtract() {
				add()
			}
		}
	}
}

func printPoints(points []point) {
	for i, p := range points {
		printPoint(i, p)
	}
}

func printPoint(i int, p point) {
	fmt.Printf("node %d x:%.6f y:%.6f z:%.6f\n", i+1, p.x, p.y, p.z)
}

func distance(p, q point) float64 {
	return math.Hypot(p.x-q.x, math.Hypot(p.y-q.y, p.z-q.z))
}

func linkWeights(points []point) [][]float64 {
	weights := make([][]float64, len(points))
	for i := range points {
		weights[i] = make([]float64, len(points))
		for j := range points {
			if i == j {
				continue
			}
			weights[i][j] = math.Exp(-distance(points[i], points[j]))
		}
	}
	return weights
}

// shiftPair moves a and b equally along the line between them so their
// distance changes by delta. Positive delta pushes them apart.
func shiftPair(points []point, a, b int, delta float64) {
	dx := points[b].x - points[a].x
	dy := points[b].y - points[a].y
	dz := points[b].z - points[a].z
	d := math.Hypot(dx, math.Hypot(dy, dz))
	ux, uy, uz := 1.0, 0.0, 0.0
	if d != 0 {
		ux, uy, uz = dx/d, dy/d, dz/d
	}
	half := delta / 2
	points[a].x -= ux * half
	points[a].y -= uy * half
	points[a].z -= uz * half
	points[b].x += ux * half
	points[b].y += uy * half
	points[b].z += uz * half
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

func toFloat(weights [][]int) [][]float64 {
	out := make([][]float64, len(weights))
	for i := range weights {
		out[i] = make([]float64, len(weights[i]))
		for j := range weights[i] {
			out[i][j] = float64(weights[i][j])
		}
	}
	return out
}

// pageRank is the damped random walk on an undirected weighted graph.
// Transition chance along an edge is proportional to its weight. A node with
// no remaining weight spreads its rank evenly, which is the usual dangling-node rule.
func pageRank(weights [][]float64) []float64 {
	n := len(weights)
	out := make([]float64, n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			out[i] += weights[i][j]
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
					next[i] += scale * row[i]
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
