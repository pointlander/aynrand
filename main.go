// Copyright 2026 The Ayn Rand Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// -g perturbs one integer link weight by ±1.
// -e gives every node an (x, y, z) coordinate. PageRank uses e to the minus
// Euclidean distance as the link weight. A random pair is moved so that
// distance changes by 1. Either model keeps a proposal when the Shannon
// entropy of PageRank decreases.
// -s is the integer-weight search. Each node also gets a 3D point from
// trilateration of those weights, with node 1 held at the origin.
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
	// FlagSnodgrass snodgrass graphical model
	FlagSnodgrass = flag.Bool("s", false, "snodgrass graphical model")
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
	if *FlagSnodgrass {
		snodgrass()
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

// snodgrass graphical model.
// Link weights stay the integer search state. Coordinates are rebuilt from
// those weights by trilateration whenever they are shown.
func snodgrass() {
	rng := rand.New(rand.NewSource(1))

	weights := randomSymmetricWeights(rng)
	ranks := pageRank(toFloat(weights))

	fmt.Printf("33-node undirected graph, symmetric weights uniform on %d..%d, damping %g\n", minWeight, maxWeight, damping)
	fmt.Println("initial")
	printRanks(ranks)
	reportCoordinates(weights)

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
				reportCoordinates(weights)
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
				reportCoordinates(weights)
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

func reportCoordinates(weights [][]int) {
	points, anchor := triangulate(toFloat(weights))
	if anchor < 2 {
		fmt.Println("coordinates by triangulation, node 1 at the origin, basis 1,2")
	} else {
		fmt.Printf("coordinates by triangulation, node 1 at the origin, basis 1,2,%d\n", anchor+1)
	}
	printPoints(points)
}

// triangulate embeds each node in R^3 from pairwise distances.
// Node 0 is the origin and node 1 lies on the positive x-axis. The third
// basis node is the one farthest from that axis, placed in the xy-plane
// with positive y. Every other node is trilaterated from those three, on
// the positive-z side of their plane.
func triangulate(weights [][]float64) ([]point, int) {
	n := len(weights)
	points := make([]point, n)
	if n < 2 {
		return points, 0
	}
	d01 := weights[0][1]
	if d01 == 0 {
		d01 = 1
	}
	points[1] = point{x: d01}
	if n < 3 {
		return points, 1
	}

	anchor := 2
	bestY := 0.0
	best := placeInPlane(weights[0][2], weights[1][2], d01)
	for k := 2; k < n; k++ {
		p := placeInPlane(weights[0][k], weights[1][k], d01)
		if p.y > bestY {
			bestY = p.y
			anchor = k
			best = p
		}
	}
	points[anchor] = best
	if bestY == 0 {
		for i := 2; i < n; i++ {
			points[i] = placeInPlane(weights[0][i], weights[1][i], d01)
		}
		return points, anchor
	}
	for i := 2; i < n; i++ {
		if i == anchor {
			continue
		}
		points[i] = trilaterate(points[0], points[1], points[anchor], weights[0][i], weights[1][i], weights[anchor][i])
	}
	return points, anchor
}

// placeInPlane lays a point in the xy-plane given its distances to the
// origin and to a point at (d01, 0, 0). y is zero when the two circles miss.
func placeInPlane(d0, d1, d01 float64) point {
	if d01 == 0 {
		return point{x: d0}
	}
	x := (d01*d01 + d0*d0 - d1*d1) / (2 * d01)
	y2 := d0*d0 - x*x
	y := 0.0
	if y2 > 0 {
		y = math.Sqrt(y2)
	}
	return point{x: x, y: y}
}

// trilaterate returns the point at the given distances from p1, p2, and p3.
// The z coordinate is taken non-negative in the plane's right-handed frame.
func trilaterate(p1, p2, p3 point, r1, r2, r3 float64) point {
	d := distance(p1, p2)
	if d == 0 {
		return point{x: p1.x + r1, y: p1.y, z: p1.z}
	}
	ex := point{(p2.x - p1.x) / d, (p2.y - p1.y) / d, (p2.z - p1.z) / d}
	vx, vy, vz := p3.x-p1.x, p3.y-p1.y, p3.z-p1.z
	i := ex.x*vx + ex.y*vy + ex.z*vz
	tx, ty, tz := vx-i*ex.x, vy-i*ex.y, vz-i*ex.z
	j := math.Hypot(tx, math.Hypot(ty, tz))
	if j == 0 {
		return point{x: p1.x + ex.x*r1, y: p1.y + ex.y*r1, z: p1.z + ex.z*r1}
	}
	ey := point{tx / j, ty / j, tz / j}
	ez := point{
		ex.y*ey.z - ex.z*ey.y,
		ex.z*ey.x - ex.x*ey.z,
		ex.x*ey.y - ex.y*ey.x,
	}
	x := (r1*r1 - r2*r2 + d*d) / (2 * d)
	y := (r1*r1-r3*r3+i*i+j*j)/(2*j) - (i/j)*x
	z := 0.0
	if z2 := r1*r1 - x*x - y*y; z2 > 0 {
		z = math.Sqrt(z2)
	}
	return point{
		x: p1.x + x*ex.x + y*ey.x + z*ez.x,
		y: p1.y + x*ex.y + y*ey.y + z*ez.y,
		z: p1.z + x*ex.z + y*ey.z + z*ez.z,
	}
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
