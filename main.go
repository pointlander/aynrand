// Copyright 2026 The Ayn Rand Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// -g perturbs one integer link weight by ±1.
// -e gives every node an (x, y, z) coordinate. PageRank uses e to the minus
// Euclidean distance as the link weight. A random pair is moved so that
// distance changes by 1. Either model keeps a proposal when the Shannon
// entropy of PageRank decreases.
// -s uses big.Float link weights. Each step samples two distinct nodes from
// the live PageRank. A proposal adds a normal sample scaled by the standard
// deviation of the weights to that link and clamps it at 0.1, then rebuilds
// the coordinates. New distances are buffered and take effect after a delay
// equal to the current distance, so a change moves at finite speed. PageRank
// uses the inverse square of each weight. Node 1 is the origin.
package main

import (
	"flag"
	"fmt"
	"math"
	"math/big"
	"math/rand"
	"sort"
)

const (
	minWeight      = 1
	maxWeight      = 33
	damping        = 0.85
	snodgrassFloor = 0.1
	tolerance      = "1e-15"
	maxIter        = 10000
	prec           = 256
)

var nodeCount = 33

var (
	bfZero    = mustBF("0")
	bfOne     = mustBF("1")
	bfTwo     = mustBF("2")
	bfDamping = mustBF("0.85")
	bfFloor   = mustBF("0.1")
	bfTol     = mustBF(tolerance)
	bfFlat    = mustBF("1e-6")
	bfG       = mustBF("6.6743e-11")
	ln2       = mustBF("0.693147180559945309417232121458176568075500134360255254120680009493393621969694715605863326996418687")
)

var (
	// FlagGraphical graphical model
	FlagGraphical = flag.Bool("g", false, "graphical model")
	// FlagGraphiclLM graphical language model
	FlagGraphicalLM = flag.Bool("lm", false, "graphical language model")
	// FlagEuclidean euclidean graphical model
	FlagEuclidean = flag.Bool("e", false, "euclidean graphical model")
	// FlagSnodgrass snodgrass graphical model
	FlagSnodgrass = flag.Bool("s", false, "snodgrass graphical model")
)

func main() {
	flag.Parse()
	if *FlagEuclidean {
		euclidean()
		return
	}
	if *FlagGraphical {
		graphical()
		return
	}
	if *FlagGraphicalLM {
		graphicallm()
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
		beforeX, beforeY := weights[a][b], weights[b][a]
		beforeRanks := ranks

		mutate := func(x, y int) bool {
			setWeight(weights, a, b, beforeX+x)
			setWeight(weights, b, a, beforeY+y)
			plus := pageRank(toFloat(weights))
			plusBoth := lt(plus, beforeRanks)
			if plusBoth {
				fmt.Printf("step %d nodes %d,%d %d %d\n", step, a+1, b+1, x, y)
				printRanks(plus)
				ranks = plus
				return true
			}
			setWeight(weights, a, b, beforeX)
			setWeight(weights, b, a, beforeY)
			return false
		}

		if choice := rng.Intn(4); choice == 0 {
			mutate(1, 1)
		} else if choice == 1 {
			mutate(1, -1)
		} else if choice == 2 {
			mutate(-1, 1)
		} else if choice == 3 {
			mutate(-1, -1)
		}
	}
}

// graphicall  language model
func graphicallm() {
	rng := rand.New(rand.NewSource(1))

	weights := randomSymmetricWeights(rng)
	ranks := pageRank(toFloat(weights))

	fmt.Printf("33-node undirected graph, symmetric weights uniform on %d..%d, damping %g\n", minWeight, maxWeight, damping)
	fmt.Println("initial")
	printRanks(ranks)

	a := 0
	for step := 1; ; step++ {
		min, next := mustBF(fmt.Sprintf("%f", math.MaxFloat64)), 0
		for b := 0; b < 33; b++ {
			if b == a {
				continue
			}
			beforeX, beforeY := weights[a][b], weights[b][a]
			beforeRanks := ranks

			mutate := func(x, y int) bool {
				setWeight(weights, a, b, beforeX+x)
				setWeight(weights, b, a, beforeY+y)
				plus := pageRank(toFloat(weights))
				plusBoth := lt(plus, beforeRanks)
				e := entropy(plus)
				if e.Cmp(min) < 0 {
					min, next = e, b
				}
				if plusBoth {
					//fmt.Printf("step %d nodes %d,%d %d %d\n", step, a+1, b+1, x, y)
					//printRanks(plus)
					ranks = plus
					return true
				}
				setWeight(weights, a, b, beforeX)
				setWeight(weights, b, a, beforeY)
				return false
			}

			if choice := rng.Intn(4); choice == 0 {
				mutate(1, 1)
			} else if choice == 1 {
				mutate(1, -1)
			} else if choice == 2 {
				mutate(-1, 1)
			} else if choice == 3 {
				mutate(-1, -1)
			}
		}
		a = next
		fmt.Println(a)
	}
}

type point struct {
	x, y, z *big.Float
}

func (p point) clone() point {
	return point{bfCopy(p.x), bfCopy(p.y), bfCopy(p.z)}
}

func originPoint() point {
	return point{newFloat(0), newFloat(0), newFloat(0)}
}

// snodgrass graphical model.
// Weights are big.Float values. Each step samples two distinct nodes from the
// live PageRank. A proposal adds one normal sample, scaled by the standard
// deviation of the accepted link weights, to a link, clamps the result at 0.1,
// and rebuilds the embedding. Live distances keep their current values until a
// delay of that many steps has elapsed, then take the accepted distance. A
// later accept refreshes the buffered distance without extending the delay.
// PageRank sees the inverse square of each weight.
func snodgrass() {
	rng := rand.New(rand.NewSource(1))

	weights := toFloat(randomSymmetricWeights(rng))
	ranks := pageRank(expMinusWeights(weights))
	goal := cloneWeights(weights)
	goalRanks := ranks
	due := make([][]int, len(weights))
	for i := range due {
		due[i] = make([]int, len(weights))
	}

	fmt.Printf("33-node undirected graph, symmetric big.Float weights uniform on %d..%d, nodes sampled from PageRank, proposal N(0, σ) scaled by the weight standard deviation, clamped at %g, distance updates arrive after a delay equal to the current distance, PageRank uses 1/weight^2, damping %g\n", minWeight, maxWeight, snodgrassFloor, damping)
	fmt.Println("initial")
	printRanks(ranks)
	reportCoordinates(weights)

	for step := 1; ; step++ {
		if applyBuffered(weights, goal, due, step) {
			ranks = pageRank(expMinusWeights(weights))
		}

		a, b := sampleRankPair(rng, ranks)
		before := bfCopy(goal[a][b])
		saved := cloneWeights(goal)

		sample := bfMul(weightStdDev(goal), newFloat(.01*rng.NormFloat64()))
		nextWeight := proposeSnodgrassWeight(before, sample)
		setFloatWeight(goal, a, b, nextWeight)
		points, xNode := syncOtherWeights(goal, a, b)
		next := pageRank(expMinusWeights(goal))
		if lt(next, goalRanks) {
			scheduleWeights(due, weights, goal, step)
			delay := 0
			if due[a][b] != 0 {
				delay = due[a][b] - step
			}
			fmt.Printf("step %d nodes %d,%d weight %s -> %s sample %s delay %d\n", step, a+1, b+1, before.Text('f', 6), nextWeight.Text('f', 6), sample.Text('f', 6), delay)
			printRanks(next)
			printEmbedding(points, xNode)
			goalRanks = next
			continue
		}
		copyWeightsInto(goal, saved)
	}
}

// travelDelay is how many steps a distance update waits. The wait equals the
// current distance rounded to the nearest integer, and it is at least 1 so the
// live weight does not change in the accepting step.
func travelDelay(distance *big.Float) int {
	v, _ := distance.Float64()
	if math.IsNaN(v) || v < 1 {
		return 1
	}
	d := int(math.Round(v))
	if d < 1 {
		return 1
	}
	return d
}

// scheduleWeights buffers goal distances that differ from the live weights.
// The arrival step is the current step plus the live distance. An edge that
// already has a buffered update keeps its arrival step and picks up the new
// goal value when that step comes.
func scheduleWeights(due [][]int, applied, goal [][]*big.Float, step int) {
	for i := 0; i < len(goal); i++ {
		for j := i + 1; j < len(goal); j++ {
			if applied[i][j].Cmp(goal[i][j]) == 0 {
				due[i][j] = 0
				continue
			}
			if due[i][j] == 0 {
				due[i][j] = step + travelDelay(applied[i][j])
			}
		}
	}
}

// applyBuffered writes goal distances whose travel time has elapsed into the
// live weights. due is 0 when an edge has nothing waiting.
func applyBuffered(applied, goal [][]*big.Float, due [][]int, step int) bool {
	arrived := false
	for i := 0; i < len(applied); i++ {
		for j := i + 1; j < len(applied); j++ {
			if due[i][j] == 0 || due[i][j] > step {
				continue
			}
			setFloatWeight(applied, i, j, goal[i][j])
			due[i][j] = 0
			arrived = true
		}
	}
	return arrived
}

// weightStdDev is the population standard deviation of the undirected link weights.
func weightStdDev(weights [][]*big.Float) *big.Float {
	n := 0
	sum := newFloat(0)
	for i := 0; i < len(weights); i++ {
		for j := i + 1; j < len(weights); j++ {
			sum = bfAdd(sum, weights[i][j])
			n++
		}
	}
	count := newInt(int64(n))
	mean := bfQuo(sum, count)
	acc := newFloat(0)
	for i := 0; i < len(weights); i++ {
		for j := i + 1; j < len(weights); j++ {
			d := bfSub(weights[i][j], mean)
			acc = bfAdd(acc, bfMul(d, d))
		}
	}
	return bfSqrt(bfQuo(acc, count))
}

// proposeSnodgrassWeight adds the scaled sample and keeps the link at least 0.1.
func proposeSnodgrassWeight(before, sample *big.Float) *big.Float {
	next := bfAdd(before, sample)
	if next.Cmp(bfFloor) < 0 {
		return bfCopy(bfFloor)
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
	span := newInt(int64(maxWeight - minWeight))
	floor := newInt(int64(minWeight))
	for i := range points {
		points[i] = point{
			x: bfAdd(floor, bfMul(newFloat(rng.Float64()), span)),
			y: bfAdd(floor, bfMul(newFloat(rng.Float64()), span)),
			z: bfAdd(floor, bfMul(newFloat(rng.Float64()), span)),
		}
	}
	ranks := pageRank(linkWeights(points))

	fmt.Printf("33-node euclidean graph, coordinates uniform on [%d, %d), link weight is e^-distance, damping %g\n", minWeight, maxWeight, damping)
	printPoints(points)
	fmt.Println("initial")
	printRanks(ranks)

	for step := 1; ; step++ {
		a, b := randomPair(rng)
		pa, pb := points[a].clone(), points[b].clone()
		beforeDist := distance(pa, pb)

		add := func() bool {
			shiftPair(points, a, b, bfOne)
			plus := pageRank(linkWeights(points))
			if lt(plus, ranks) {
				afterDist := distance(points[a], points[b])
				fmt.Println("++++++++++++++++++++++++++++++++++")
				fmt.Printf("step %d nodes %d,%d distance %s -> %s weight %s -> %s\n", step, a+1, b+1, beforeDist.Text('f', 6), afterDist.Text('f', 6), bfExp(bfNeg(beforeDist)).Text('e', 6), bfExp(bfNeg(afterDist)).Text('e', 6))
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
			if bfSub(beforeDist, bfOne).Cmp(bfOne) < 0 {
				return false
			}
			shiftPair(points, a, b, bfNeg(bfOne))
			minus := pageRank(linkWeights(points))
			if lt(minus, ranks) {
				afterDist := distance(points[a], points[b])
				fmt.Println("----------------------------------")
				fmt.Printf("step %d nodes %d,%d distance %s -> %s weight %s -> %s\n", step, a+1, b+1, beforeDist.Text('f', 6), afterDist.Text('f', 6), bfExp(bfNeg(beforeDist)).Text('e', 6), bfExp(bfNeg(afterDist)).Text('e', 6))
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
	fmt.Printf("node %d x:%s y:%s z:%s\n", i+1, p.x.Text('f', 6), p.y.Text('f', 6), p.z.Text('f', 6))
}

func distance(p, q point) *big.Float {
	return bfHypot3(bfSub(p.x, q.x), bfSub(p.y, q.y), bfSub(p.z, q.z))
}

func reportCoordinates(weights [][]*big.Float) {
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
func syncOtherWeights(weights [][]*big.Float, keepA, keepB int) ([]point, int) {
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
			weights[j][i] = bfCopy(d)
		}
	}
	return points, xNode
}

func cloneWeights(weights [][]*big.Float) [][]*big.Float {
	out := make([][]*big.Float, len(weights))
	for i := range weights {
		out[i] = make([]*big.Float, len(weights[i]))
		for j := range weights[i] {
			out[i][j] = bfCopy(weights[i][j])
		}
	}
	return out
}

func copyWeightsInto(dst, src [][]*big.Float) {
	for i := range dst {
		for j := range dst[i] {
			dst[i][j] = bfCopy(src[i][j])
		}
	}
}

func setFloatWeight(weights [][]*big.Float, a, b int, value *big.Float) {
	weights[a][b] = bfCopy(value)
	weights[b][a] = bfCopy(value)
}

// expMinusWeights returns the inverse square of each off-diagonal weight.
func expMinusWeights(weights [][]*big.Float) [][]*big.Float {
	out := make([][]*big.Float, len(weights))
	for i := range weights {
		out[i] = make([]*big.Float, len(weights[i]))
		for j := range weights[i] {
			if i == j || weights[i][j].Sign() == 0 {
				out[i][j] = newFloat(0)
				continue
			}
			//out[i][j] = bfExp(bfNeg(weights[i][j]))
			out[i][j] = bfQuo(bfG, bfMul(weights[i][j], weights[i][j]))
		}
	}
	return out
}

// triangulate embeds each node from the link weights, using a local frame
// at every step. Node 0 is the origin. Its nearest neighbor lies on the
// positive x-axis, and the nearest remaining node that makes a triangle
// with those two lies in the xy-plane. Each later node is the unplaced node
// closest to the placed set, and it is trilaterated from its three nearest
// placed nodes. A fourth nearest node picks which side of that plane it
// lands on.
func triangulate(weights [][]*big.Float) ([]point, int) {
	n := len(weights)
	points := make([]point, n)
	for i := range points {
		points[i] = originPoint()
	}
	if n < 2 {
		return points, 0
	}

	xNode := nearestIndex(weights[0], 0)
	d01 := bfCopy(weights[0][xNode])
	if d01.Sign() == 0 {
		d01 = bfCopy(bfOne)
	}
	points[xNode] = point{x: bfCopy(d01), y: newFloat(0), z: newFloat(0)}
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
		if p.y.Sign() > 0 {
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
			points[i] = trilaterate(points[p1], points[p2], points[p2], weights[i][p1], weights[i][p2], weights[i][p2], bfOne)
			placed[i] = true
			continue
		}
		p3 := refs[2]
		plus := trilaterate(points[p1], points[p2], points[p3], weights[i][p1], weights[i][p2], weights[i][p3], bfOne)
		if len(refs) < 4 {
			points[i] = plus
		} else {
			minus := trilaterate(points[p1], points[p2], points[p3], weights[i][p1], weights[i][p2], weights[i][p3], bfNeg(bfOne))
			fourth := refs[3]
			want := weights[i][fourth]
			errPlus := bfAbs(bfSub(distance(plus, points[fourth]), want))
			errMinus := bfAbs(bfSub(distance(minus, points[fourth]), want))
			if errPlus.Cmp(errMinus) <= 0 {
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
func nearestIndex(row []*big.Float, skip int) int {
	best := -1
	for j := range row {
		if j == skip {
			continue
		}
		if best < 0 || row[j].Cmp(row[best]) < 0 || (row[j].Cmp(row[best]) == 0 && j < best) {
			best = j
		}
	}
	return best
}

// nodesByNearness lists unplaced nodes by distance to the closest placed node.
func nodesByNearness(weights [][]*big.Float, placed []bool) []int {
	type nearNode struct {
		i int
		d *big.Float
	}
	nodes := make([]nearNode, 0, len(placed))
	for i := range placed {
		if placed[i] {
			continue
		}
		var d *big.Float
		for j := range placed {
			if !placed[j] {
				continue
			}
			if d == nil || weights[i][j].Cmp(d) < 0 {
				d = weights[i][j]
			}
		}
		nodes = append(nodes, nearNode{i: i, d: d})
	}
	sort.Slice(nodes, func(a, b int) bool {
		cmp := nodes[a].d.Cmp(nodes[b].d)
		if cmp != 0 {
			return cmp < 0
		}
		return nodes[a].i < nodes[b].i
	})
	order := make([]int, len(nodes))
	for i, node := range nodes {
		order[i] = node.i
	}
	return order
}

func closestUnplaced(weights [][]*big.Float, placed []bool) int {
	order := nodesByNearness(weights, placed)
	return order[0]
}

// localReferences returns the two nearest placed nodes, then the nearest
// further node that is not collinear with them, then one more nearest node
// used to choose the side of that plane.
func localReferences(i int, placed []bool, points []point, weights [][]*big.Float) []int {
	order := nearestPlaced(i, placed, weights, len(placed))
	if len(order) < 2 {
		return order
	}
	p1, p2 := order[0], order[1]
	refs := []int{p1, p2}
	for _, k := range order[2:] {
		if lineHeight(points[p1], points[p2], points[k]).Cmp(bfFlat) >= 0 {
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

func lineHeight(a, b, c point) *big.Float {
	abx, aby, abz := bfSub(b.x, a.x), bfSub(b.y, a.y), bfSub(b.z, a.z)
	acx, acy, acz := bfSub(c.x, a.x), bfSub(c.y, a.y), bfSub(c.z, a.z)
	cx := bfSub(bfMul(aby, acz), bfMul(abz, acy))
	cy := bfSub(bfMul(abz, acx), bfMul(abx, acz))
	cz := bfSub(bfMul(abx, acy), bfMul(aby, acx))
	ab := bfHypot3(abx, aby, abz)
	if ab.Sign() == 0 {
		return newFloat(0)
	}
	return bfQuo(bfHypot3(cx, cy, cz), ab)
}

// nearestPlaced returns up to k placed nodes closest to i.
func nearestPlaced(i int, placed []bool, weights [][]*big.Float, k int) []int {
	idx := make([]int, 0, len(placed))
	for j := range placed {
		if placed[j] {
			idx = append(idx, j)
		}
	}
	sort.Slice(idx, func(a, b int) bool {
		cmp := weights[i][idx[a]].Cmp(weights[i][idx[b]])
		if cmp != 0 {
			return cmp < 0
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
func placeInPlane(d0, d1, d01 *big.Float) point {
	if d01.Sign() == 0 {
		return point{x: bfCopy(d0), y: newFloat(0), z: newFloat(0)}
	}
	num := bfAdd(bfMul(d01, d01), bfSub(bfMul(d0, d0), bfMul(d1, d1)))
	x := bfQuo(num, bfMul(bfTwo, d01))
	y2 := bfSub(bfMul(d0, d0), bfMul(x, x))
	y := newFloat(0)
	if y2.Sign() > 0 {
		y = bfSqrt(y2)
	}
	return point{x: x, y: y, z: newFloat(0)}
}

// trilaterate returns the point at the given distances from p1, p2, and p3.
// sign selects the side of their plane: +1 is the right-handed side.
func trilaterate(p1, p2, p3 point, r1, r2, r3, sign *big.Float) point {
	d := distance(p1, p2)
	if d.Sign() == 0 {
		return point{x: bfAdd(p1.x, r1), y: bfCopy(p1.y), z: bfCopy(p1.z)}
	}
	ex := point{
		x: bfQuo(bfSub(p2.x, p1.x), d),
		y: bfQuo(bfSub(p2.y, p1.y), d),
		z: bfQuo(bfSub(p2.z, p1.z), d),
	}
	v := point{x: bfSub(p3.x, p1.x), y: bfSub(p3.y, p1.y), z: bfSub(p3.z, p1.z)}
	i := bfDot(ex, v)
	t := point{
		x: bfSub(v.x, bfMul(i, ex.x)),
		y: bfSub(v.y, bfMul(i, ex.y)),
		z: bfSub(v.z, bfMul(i, ex.z)),
	}
	j := bfHypot3(t.x, t.y, t.z)
	if j.Sign() == 0 {
		return point{
			x: bfAdd(p1.x, bfMul(ex.x, r1)),
			y: bfAdd(p1.y, bfMul(ex.y, r1)),
			z: bfAdd(p1.z, bfMul(ex.z, r1)),
		}
	}
	ey := point{x: bfQuo(t.x, j), y: bfQuo(t.y, j), z: bfQuo(t.z, j)}
	ez := point{
		x: bfSub(bfMul(ex.y, ey.z), bfMul(ex.z, ey.y)),
		y: bfSub(bfMul(ex.z, ey.x), bfMul(ex.x, ey.z)),
		z: bfSub(bfMul(ex.x, ey.y), bfMul(ex.y, ey.x)),
	}
	x := bfQuo(bfAdd(bfSub(bfMul(r1, r1), bfMul(r2, r2)), bfMul(d, d)), bfMul(bfTwo, d))
	num := bfAdd(bfSub(bfMul(r1, r1), bfMul(r3, r3)), bfAdd(bfMul(i, i), bfMul(j, j)))
	y := bfSub(bfQuo(num, bfMul(bfTwo, j)), bfMul(bfQuo(i, j), x))
	z := newFloat(0)
	z2 := bfSub(bfMul(r1, r1), bfAdd(bfMul(x, x), bfMul(y, y)))
	if z2.Sign() > 0 {
		z = bfMul(sign, bfSqrt(z2))
	} else if r1.Sign() > 0 {
		// The spheres miss. Stay on the nearest sphere instead of extrapolating.
		rad := bfHypot(x, y)
		if rad.Sign() > 0 {
			scale := bfQuo(r1, rad)
			x = bfMul(x, scale)
			y = bfMul(y, scale)
		}
	}
	return point{
		x: bfAdd(p1.x, bfAdd(bfMul(x, ex.x), bfAdd(bfMul(y, ey.x), bfMul(z, ez.x)))),
		y: bfAdd(p1.y, bfAdd(bfMul(x, ex.y), bfAdd(bfMul(y, ey.y), bfMul(z, ez.y)))),
		z: bfAdd(p1.z, bfAdd(bfMul(x, ex.z), bfAdd(bfMul(y, ey.z), bfMul(z, ez.z)))),
	}
}

func linkWeights(points []point) [][]*big.Float {
	weights := make([][]*big.Float, len(points))
	for i := range points {
		weights[i] = make([]*big.Float, len(points))
		for j := range points {
			if i == j {
				weights[i][j] = newFloat(0)
				continue
			}
			weights[i][j] = bfExp(bfNeg(distance(points[i], points[j])))
		}
	}
	return weights
}

// shiftPair moves a and b equally along the line between them so their
// distance changes by delta. Positive delta pushes them apart.
func shiftPair(points []point, a, b int, delta *big.Float) {
	dx := bfSub(points[b].x, points[a].x)
	dy := bfSub(points[b].y, points[a].y)
	dz := bfSub(points[b].z, points[a].z)
	d := bfHypot3(dx, dy, dz)
	ux, uy, uz := bfCopy(bfOne), newFloat(0), newFloat(0)
	if d.Sign() != 0 {
		ux, uy, uz = bfQuo(dx, d), bfQuo(dy, d), bfQuo(dz, d)
	}
	half := bfQuo(delta, bfTwo)
	points[a].x = bfSub(points[a].x, bfMul(ux, half))
	points[a].y = bfSub(points[a].y, bfMul(uy, half))
	points[a].z = bfSub(points[a].z, bfMul(uz, half))
	points[b].x = bfAdd(points[b].x, bfMul(ux, half))
	points[b].y = bfAdd(points[b].y, bfMul(uy, half))
	points[b].z = bfAdd(points[b].z, bfMul(uz, half))
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

// sampleRankPair draws two distinct nodes with probability proportional to PageRank.
// The second draw is taken from the remaining nodes.
func sampleRankPair(rng *rand.Rand, ranks []*big.Float) (int, int) {
	a := sampleRankNode(rng, ranks, -1)
	b := sampleRankNode(rng, ranks, a)
	if a > b {
		a, b = b, a
	}
	return a, b
}

func sampleRankNode(rng *rand.Rand, ranks []*big.Float, skip int) int {
	total := newFloat(0)
	for i, rank := range ranks {
		if i == skip {
			continue
		}
		total = bfAdd(total, rank)
	}
	mark := bfMul(total, newFloat(rng.Float64()))
	acc := newFloat(0)
	chosen := 0
	for i, rank := range ranks {
		if i == skip {
			continue
		}
		chosen = i
		acc = bfAdd(acc, rank)
		if acc.Cmp(mark) >= 0 {
			return i
		}
	}
	return chosen
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

func entropy(ranks []*big.Float) *big.Float {
	h := newFloat(0)
	for _, value := range ranks {
		h = bfAdd(h, bfMul(value, bfLog(value)))
	}
	return bfNeg(h)
}

func lt(next, prev []*big.Float) bool {
	return entropy(next).Cmp(entropy(prev)) < 0
}

func gt(next, prev []*big.Float) bool {
	return entropy(next).Cmp(entropy(prev)) > 0
}

func printRanks(ranks []*big.Float) {
	sum := newFloat(0)
	for i, rank := range ranks {
		if i > 0 {
			fmt.Print(" ")
		}
		fmt.Printf("%d:%s", i+1, rank.Text('f', 8))
		sum = bfAdd(sum, rank)
	}
	fmt.Printf(" sum:%s\n", sum.Text('f', 8))
}

func toFloat(weights [][]int) [][]*big.Float {
	out := make([][]*big.Float, len(weights))
	for i := range weights {
		out[i] = make([]*big.Float, len(weights[i]))
		for j := range weights[i] {
			out[i][j] = newInt(int64(weights[i][j]))
		}
	}
	return out
}

// pageRank is the damped random walk on an undirected weighted graph.
// Transition chance along an edge is proportional to its weight. A node with
// no remaining weight spreads its rank evenly, which is the usual dangling-node rule.
func pageRank(weights [][]*big.Float) []*big.Float {
	n := len(weights)
	out := make([]*big.Float, n)
	for i := 0; i < n; i++ {
		out[i] = newFloat(0)
		for j := 0; j < n; j++ {
			out[i] = bfAdd(out[i], bfAbs(weights[i][j]))
		}
	}

	uniform := bfQuo(bfOne, newInt(int64(n)))
	base := bfMul(bfSub(bfOne, bfDamping), uniform)
	ranks := make([]*big.Float, n)
	next := make([]*big.Float, n)
	for i := range ranks {
		ranks[i] = bfCopy(uniform)
	}

	for iter := 0; iter < maxIter; iter++ {
		for i := range next {
			next[i] = bfCopy(base)
		}
		dangling := newFloat(0)
		for j := 0; j < n; j++ {
			if out[j].Sign() == 0 {
				dangling = bfAdd(dangling, ranks[j])
				continue
			}
			scale := bfQuo(bfMul(bfDamping, ranks[j]), out[j])
			row := weights[j]
			for i := 0; i < n; i++ {
				if row[i].Sign() == 0 {
					continue
				}
				next[i] = bfAdd(next[i], bfMul(scale, bfAbs(row[i])))
			}
		}
		if dangling.Sign() != 0 {
			add := bfMul(bfDamping, bfMul(dangling, uniform))
			for i := range next {
				next[i] = bfAdd(next[i], add)
			}
		}

		delta := newFloat(0)
		for i := 0; i < n; i++ {
			diff := bfAbs(bfSub(next[i], ranks[i]))
			if diff.Cmp(delta) > 0 {
				delta = diff
			}
			ranks[i] = bfCopy(next[i])
		}
		if delta.Cmp(bfTol) < 0 {
			break
		}
	}
	return ranks
}

func mustBF(s string) *big.Float {
	f, ok := new(big.Float).SetPrec(prec).SetString(s)
	if !ok {
		panic("bad float " + s)
	}
	return f
}

func newFloat(v float64) *big.Float {
	return new(big.Float).SetPrec(prec).SetFloat64(v)
}

func newInt(v int64) *big.Float {
	return new(big.Float).SetPrec(prec).SetInt64(v)
}

func bfCopy(x *big.Float) *big.Float {
	return new(big.Float).SetPrec(prec).Copy(x)
}

func bfAdd(a, b *big.Float) *big.Float {
	return new(big.Float).SetPrec(prec).Add(a, b)
}

func bfSub(a, b *big.Float) *big.Float {
	return new(big.Float).SetPrec(prec).Sub(a, b)
}

func bfMul(a, b *big.Float) *big.Float {
	return new(big.Float).SetPrec(prec).Mul(a, b)
}

func bfQuo(a, b *big.Float) *big.Float {
	return new(big.Float).SetPrec(prec).Quo(a, b)
}

func bfSqrt(a *big.Float) *big.Float {
	return new(big.Float).SetPrec(prec).Sqrt(a)
}

func bfAbs(a *big.Float) *big.Float {
	return new(big.Float).SetPrec(prec).Abs(a)
}

func bfNeg(a *big.Float) *big.Float {
	return new(big.Float).SetPrec(prec).Neg(a)
}

func bfHypot(x, y *big.Float) *big.Float {
	return bfSqrt(bfAdd(bfMul(x, x), bfMul(y, y)))
}

func bfHypot3(x, y, z *big.Float) *big.Float {
	return bfSqrt(bfAdd(bfAdd(bfMul(x, x), bfMul(y, y)), bfMul(z, z)))
}

func bfDot(a, b point) *big.Float {
	return bfAdd(bfAdd(bfMul(a.x, b.x), bfMul(a.y, b.y)), bfMul(a.z, b.z))
}

// bfLog is the natural log. The argument must be positive.
func bfLog(x *big.Float) *big.Float {
	mant := new(big.Float).SetPrec(prec)
	exp := x.MantExp(mant)
	z := bfQuo(bfSub(mant, bfOne), bfAdd(mant, bfOne))
	z2 := bfMul(z, z)
	sum := bfCopy(z)
	term := bfCopy(z)
	for k := 1; k < int(prec); k++ {
		term = bfMul(term, z2)
		sum = bfAdd(sum, bfQuo(term, newInt(int64(2*k+1))))
		if term.Sign() != 0 && term.MantExp(nil) < -int(prec) {
			break
		}
	}
	return bfAdd(bfMul(bfTwo, sum), bfMul(newInt(int64(exp)), ln2))
}

// bfExp is the natural exponential.
func bfExp(x *big.Float) *big.Float {
	qf, _ := bfQuo(x, ln2).Float64()
	n := int64(math.Round(qf))
	r := bfSub(x, bfMul(newInt(n), ln2))
	sum := bfCopy(bfOne)
	term := bfCopy(bfOne)
	for k := int64(1); k < int64(prec)*2; k++ {
		term = bfQuo(bfMul(term, r), newInt(k))
		sum = bfAdd(sum, term)
		if term.Sign() != 0 && term.MantExp(nil) < -int(prec)-4 {
			break
		}
	}
	mant := new(big.Float).SetPrec(prec)
	e := sum.MantExp(mant)
	return new(big.Float).SetPrec(prec).SetMantExp(mant, e+int(n))
}
