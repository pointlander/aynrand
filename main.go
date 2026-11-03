// Copyright 2026 The Ayn Rand Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// -g perturbs one integer link weight by ±1.
// -e gives every node an (x, y, z) coordinate. PageRank uses e to the minus
// Euclidean distance as the link weight. A random pair is moved so that
// distance changes by 1. Either model keeps a proposal when the Shannon
// entropy of PageRank decreases.
// -s uses float link weights. A proposal adds a standard normal sample to one
// link and clamps it at 0.1, then rebuilds the coordinates and replaces the
// other links with those Euclidean distances before PageRank. Node 1 is the origin.
package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"sort"
)

const (
	minWeight      = 1
	maxWeight      = 33
	damping        = 0.85
	snodgrassFloor = 0.1
	tolerance      = 1e-15
	maxIter        = 10000
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
// Weights are floats. A proposal adds one standard normal sample to a link,
// clamps the result at 0.1, rebuilds the embedding, and writes Euclidean
// distances back into every other link before PageRank.
func snodgrass() {
	rng := rand.New(rand.NewSource(1))

	weights := toFloat(randomSymmetricWeights(rng))
	ranks := pageRank(weights)

	fmt.Printf("33-node undirected graph, symmetric float weights uniform on %d..%d, proposal N(0,1) clamped at %g, damping %g\n", minWeight, maxWeight, snodgrassFloor, damping)
	fmt.Println("initial")
	printRanks(ranks)
	reportCoordinates(weights)

	for step := 1; ; step++ {
		a, b := randomPair(rng)
		before := weights[a][b]
		beforeRanks := ranks
		saved := cloneWeights(weights)

		sample := rng.NormFloat64()
		nextWeight := proposeSnodgrassWeight(before, sample)
		setFloatWeight(weights, a, b, nextWeight)
		points, xNode := syncOtherWeights(weights, a, b)
		next := pageRank(weights)
		if raisedBoth(next, beforeRanks, a, b) {
			fmt.Printf("step %d nodes %d,%d weight %.6f -> %.6f sample %.6f\n", step, a+1, b+1, before, nextWeight, sample)
			printRanks(next)
			printEmbedding(points, xNode)
			ranks = next
			continue
		}
		copyWeightsInto(weights, saved)
	}
}

// proposeSnodgrassWeight adds a standard normal sample and keeps the link at least 0.1.
func proposeSnodgrassWeight(before, sample float64) float64 {
	next := before + sample
	if next < snodgrassFloor {
		return snodgrassFloor
	}
	return next
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

func reportCoordinates(weights [][]float64) {
	points, xNode := triangulate(weights)
	printEmbedding(points, xNode)
}

func printEmbedding(points []point, xNode int) {
	if xNode <= 0 {
		fmt.Println("coordinates by nearest-point trilateration, node 1 at the origin")
	} else {
		fmt.Printf("coordinates by nearest-point trilateration, node 1 at the origin, node %d on the x-axis\n", xNode+1)
	}
	printPoints(points)
}

// syncOtherWeights embeds weights, then sets every edge except keepA-keepB
// to the Euclidean distance of that embedding. The kept edge is the one
// just edited. PageRank runs after this returns.
func syncOtherWeights(weights [][]float64, keepA, keepB int) ([]point, int) {
	if keepA > keepB {
		keepA, keepB = keepB, keepA
	}
	points, xNode := triangulate(weights)
	for i := range points {
		for j := i + 1; j < len(points); j++ {
			if i == keepA && j == keepB {
				continue
			}
			d := distance(points[i], points[j])
			weights[i][j] = d
			weights[j][i] = d
		}
	}
	return points, xNode
}

func cloneWeights(weights [][]float64) [][]float64 {
	out := make([][]float64, len(weights))
	for i := range weights {
		out[i] = append([]float64(nil), weights[i]...)
	}
	return out
}

func copyWeightsInto(dst, src [][]float64) {
	for i := range dst {
		copy(dst[i], src[i])
	}
}

func setFloatWeight(weights [][]float64, a, b int, value float64) {
	weights[a][b] = value
	weights[b][a] = value
}

// triangulate embeds each node from the link weights, using a local frame
// at every step. Node 0 is the origin. Its nearest neighbor lies on the
// positive x-axis, and the nearest remaining node that makes a triangle
// with those two lies in the xy-plane. Each later node is the unplaced node
// closest to the placed set, and it is trilaterated from its three nearest
// placed nodes. A fourth nearest node picks which side of that plane it
// lands on.
func triangulate(weights [][]float64) ([]point, int) {
	n := len(weights)
	points := make([]point, n)
	if n < 2 {
		return points, 0
	}

	xNode := nearestIndex(weights[0], 0)
	d01 := weights[0][xNode]
	if d01 == 0 {
		d01 = 1
	}
	points[xNode] = point{x: d01}
	if n < 3 {
		return points, xNode
	}

	placed := make([]bool, n)
	placed[0] = true
	placed[xNode] = true

	third := -1
	order := nodesByNearness(weights, placed)
	for _, k := range order {
		p := placeInPlane(weights[0][k], weights[xNode][k], d01)
		if p.y > 0 {
			points[k] = p
			third = k
			break
		}
	}
	if third < 0 {
		for i := 1; i < n; i++ {
			if i == xNode {
				continue
			}
			points[i] = placeInPlane(weights[0][i], weights[xNode][i], d01)
		}
		return points, xNode
	}
	placed[third] = true

	for placedCount := 3; placedCount < n; placedCount++ {
		i := closestUnplaced(weights, placed)
		refs := localReferences(i, placed, points, weights)
		p1, p2 := refs[0], refs[1]
		if len(refs) < 3 {
			points[i] = trilaterate(points[p1], points[p2], points[p2], weights[i][p1], weights[i][p2], weights[i][p2], +1)
			placed[i] = true
			continue
		}
		p3 := refs[2]
		plus := trilaterate(points[p1], points[p2], points[p3], weights[i][p1], weights[i][p2], weights[i][p3], +1)
		if len(refs) < 4 {
			points[i] = plus
		} else {
			minus := trilaterate(points[p1], points[p2], points[p3], weights[i][p1], weights[i][p2], weights[i][p3], -1)
			fourth := refs[3]
			want := weights[i][fourth]
			errPlus := math.Abs(distance(plus, points[fourth]) - want)
			errMinus := math.Abs(distance(minus, points[fourth]) - want)
			if errPlus <= errMinus {
				points[i] = plus
			} else {
				points[i] = minus
			}
		}
		placed[i] = true
	}
	return points, xNode
}

// nearestIndex returns the index other than skip with the smallest weight.
// Ties take the smaller index.
func nearestIndex(row []float64, skip int) int {
	best := -1
	for j := range row {
		if j == skip {
			continue
		}
		if best < 0 || row[j] < row[best] || (row[j] == row[best] && j < best) {
			best = j
		}
	}
	return best
}

// nodesByNearness lists unplaced nodes by distance to the closest placed node.
func nodesByNearness(weights [][]float64, placed []bool) []int {
	type nearNode struct {
		i int
		d float64
	}
	nodes := make([]nearNode, 0, len(placed))
	for i := range placed {
		if placed[i] {
			continue
		}
		d := math.Inf(1)
		for j := range placed {
			if placed[j] && weights[i][j] < d {
				d = weights[i][j]
			}
		}
		nodes = append(nodes, nearNode{i: i, d: d})
	}
	sort.Slice(nodes, func(a, b int) bool {
		if nodes[a].d != nodes[b].d {
			return nodes[a].d < nodes[b].d
		}
		return nodes[a].i < nodes[b].i
	})
	order := make([]int, len(nodes))
	for i, node := range nodes {
		order[i] = node.i
	}
	return order
}

func closestUnplaced(weights [][]float64, placed []bool) int {
	order := nodesByNearness(weights, placed)
	return order[0]
}

// localReferences returns the two nearest placed nodes, then the nearest
// further node that is not collinear with them, then one more nearest node
// used to choose the side of that plane.
func localReferences(i int, placed []bool, points []point, weights [][]float64) []int {
	order := nearestPlaced(i, placed, weights, len(placed))
	if len(order) < 2 {
		return order
	}
	p1, p2 := order[0], order[1]
	refs := []int{p1, p2}
	for _, k := range order[2:] {
		if lineHeight(points[p1], points[p2], points[k]) >= 1e-6 {
			refs = append(refs, k)
			break
		}
	}
	chosen := map[int]bool{p1: true, p2: true}
	if len(refs) == 3 {
		chosen[refs[2]] = true
	}
	for _, k := range order {
		if !chosen[k] {
			refs = append(refs, k)
			break
		}
	}
	return refs
}

func lineHeight(a, b, c point) float64 {
	abx, aby, abz := b.x-a.x, b.y-a.y, b.z-a.z
	acx, acy, acz := c.x-a.x, c.y-a.y, c.z-a.z
	cx := aby*acz - abz*acy
	cy := abz*acx - abx*acz
	cz := abx*acy - aby*acx
	ab := math.Hypot(abx, math.Hypot(aby, abz))
	if ab == 0 {
		return 0
	}
	return math.Hypot(cx, math.Hypot(cy, cz)) / ab
}

// nearestPlaced returns up to k placed nodes closest to i.
func nearestPlaced(i int, placed []bool, weights [][]float64, k int) []int {
	idx := make([]int, 0, len(placed))
	for j := range placed {
		if placed[j] {
			idx = append(idx, j)
		}
	}
	sort.Slice(idx, func(a, b int) bool {
		da, db := weights[i][idx[a]], weights[i][idx[b]]
		if da != db {
			return da < db
		}
		return idx[a] < idx[b]
	})
	if len(idx) > k {
		idx = idx[:k]
	}
	return idx
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
// sign selects the side of their plane: +1 is the right-handed side.
func trilaterate(p1, p2, p3 point, r1, r2, r3, sign float64) point {
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
		z = sign * math.Sqrt(z2)
	} else if r1 > 0 {
		// The spheres miss. Stay on the nearest sphere instead of extrapolating.
		rad := math.Hypot(x, y)
		if rad > 0 {
			scale := r1 / rad
			x *= scale
			y *= scale
		}
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
