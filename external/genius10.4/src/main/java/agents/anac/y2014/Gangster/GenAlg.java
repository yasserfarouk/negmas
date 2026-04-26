package agents.anac.y2014.Gangster;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.PriorityQueue;
import java.util.Random;

import genius.core.Bid;
import genius.core.bidding.BidDetails;
import genius.core.issue.IssueInteger;
import genius.core.issue.Value;
import genius.core.issue.ValueInteger;
import genius.core.utility.NonlinearUtilitySpace;

class GenAlg {

	NonlinearUtilitySpace utilitySpace;
	int numIssues;

	int initialGenerationSize;
	int numSurvivors;
	int numGenerations;
	int minDistance; // the minimum distance between any pair of elements in a
						// survivor set.

	PriorityQueue<BidDetails> generation;
	ArrayList<BidDetails> newGeneration;
	ArrayList<BidDetails> survivors;
	ArrayList<BidDetails> newSurvivors;

	ArrayList<ArrayList<ValueInteger>> genesTable1;
	ArrayList<ArrayList<ValueInteger>> genesTable2;
	boolean useTable1 = true;

	Random random = new Random();

	GenAlg(NonlinearUtilitySpace utilitySpace, int initialGenerationSize,
			int numSurvivors, int numGenerations, int minDistance) {

		this.utilitySpace = utilitySpace;
		this.numIssues = utilitySpace.getDomain().getIssues().size();

		this.initialGenerationSize = initialGenerationSize;
		this.numSurvivors = numSurvivors;
		this.numGenerations = numGenerations;
		this.minDistance = minDistance;

		generation = new PriorityQueue<BidDetails>(initialGenerationSize);
		newGeneration = new ArrayList(initialGenerationSize);
		survivors = new ArrayList<BidDetails>(numSurvivors);
		newSurvivors = new ArrayList<BidDetails>(numSurvivors);

		genesTable1 = new ArrayList<ArrayList<ValueInteger>>(numIssues + 1);
		genesTable2 = new ArrayList<ArrayList<ValueInteger>>(numIssues + 1);

		// Fill the table
		genesTable1.add(null);
		genesTable2.add(null);
		for (int i = 1; i <= numIssues; i++) {

			int highestVal = ((IssueInteger) utilitySpace.getDomain()
					.getIssues().get(i - 1)).getUpperBound();
			int lowestVal = ((IssueInteger) utilitySpace.getDomain()
					.getIssues().get(i - 1)).getLowerBound();

			ArrayList<ValueInteger> list1 = new ArrayList<ValueInteger>(
					highestVal + 1);
			ArrayList<ValueInteger> list2 = new ArrayList<ValueInteger>(
					highestVal + 1);

			for (int j = lowestVal; j <= highestVal; j++) {
				list1.add(new ValueInteger(j));
			}

			genesTable1.add(list1);
			genesTable2.add(list2);
		}

	}

	ArrayList<BidDetails> globalSearch() throws Exception {
		return go(null, -1);
	}

	ArrayList<BidDetails> localSearch(Bid latestBid, int maxDistance)
			throws Exception {
		return go(latestBid, maxDistance);
	}

	private ArrayList<BidDetails> go(Bid latestBid, int maxDistance)
			throws Exception {

		generation.clear();

		// generate initial generation and request their utilities.
		for (int i = 0; i < initialGenerationSize; i++) {
			generation.add(getSample(latestBid, maxDistance));
		}

		for (int k = 1; k < numGenerations; k++) {

			// get the survivors of the generation.
			fillSurvivorList(generation);

			// if the survivors are not diverse enough the algorithm has
			// converged and we return the previous generation
			if (newSurvivors.size() < numSurvivors) {
				return survivors;
			}
			survivors.clear();
			survivors.addAll(newSurvivors);

			newGeneration.clear();

			// recombine the best ones, to create babies.
			for (int i = 0; i < numSurvivors; i++) {
				for (int j = i + 1; j < numSurvivors; j++) {

					// 45 pairs, for each pair generate 2 babies.
					if (latestBid == null) { // global search
						newGeneration.addAll(crossOver(survivors.get(i),
								survivors.get(j), 2));
					} else { // local search
						newGeneration.addAll(crossOver(latestBid, maxDistance,
								survivors.get(i), survivors.get(j)));
					}
				}
			}// size = n*(n-1)

			// create a new random sample.
			BidDetails randomSample = getSample(latestBid, maxDistance);
			for (int j = 0; j < numSurvivors; j++) {

				// 10 pairs, for each pair generate 2 babies.
				if (latestBid == null) { // global search
					newGeneration.addAll(crossOver(survivors.get(j),
							randomSample, 2));
				} else { // local search
					newGeneration.addAll(crossOver(latestBid, maxDistance,
							survivors.get(j), randomSample));
				}
			}// size = n*(n-1) + 2n = n^2 + n

			// add the survivors from the previous generation.
			for (int i = 0; i < numSurvivors; i++) {
				newGeneration.add(survivors.get(i));
			}// size n^2 + 2n

			generation.clear();
			generation.addAll(newGeneration);

		}

		fillSurvivorList(generation);

		// if the survivors are not diverse enough the algorithm has converged
		// and we return the previous generation
		if (newSurvivors.size() < numSurvivors) {
			return survivors;
		}

		return newSurvivors;

	}

	/**
	 * Clears the list of survivors, sorts the given generation, and fills the
	 * list of survivors again with the best n samples from the given
	 * generation. If this is not possible it means that we have converged, so
	 * we should return the list.
	 * 
	 * @param generation
	 * @throws Exception
	 */
	void fillSurvivorList(PriorityQueue<BidDetails> generation)
			throws Exception {

		newSurvivors.clear();
		newSurvivors.add(generation.poll());
		int l = 1;
		while (newSurvivors.size() < numSurvivors && l < generation.size()) {

			// get the next best sample from the generation
			BidDetails samp = generation.poll();
			l++;

			// test if it isn't too close to any other survivor.
			boolean shouldBeAdded = true;
			for (BidDetails survivor : newSurvivors) {
				int dist = Utils.calculateManhattanDistance(samp.getBid(),
						survivor.getBid());
				if (dist < minDistance) {
					shouldBeAdded = false;
					break;
				}
			}

			if (shouldBeAdded) {
				newSurvivors.add(samp);
			}

		}

	}

	Bid getRandomBid() throws Exception {

		HashMap<Integer, Value> newValues = new HashMap<Integer, Value>(
				numIssues, 2);

		ArrayList<ArrayList<ValueInteger>> table;
		ArrayList<ArrayList<ValueInteger>> bin;
		if (useTable1) {
			table = genesTable1;
			bin = genesTable2;
		} else {
			table = genesTable2;
			bin = genesTable1;
		}

		for (int i = 1; i <= numIssues; i++) {

			int r = random.nextInt(table.get(i).size());
			ValueInteger val = table.get(i).remove(r);
			bin.get(i).add(val);

			newValues.put(new Integer(i), val);
		}

		if (table.get(1).size() == 0) { // table.get(0) is null.
			useTable1 = !useTable1;
		}

		Bid newBid = new Bid(utilitySpace.getDomain(), newValues);
		return newBid;
	}

	BidDetails getSample(Bid latestBid, int maxDistance) throws Exception {

		Bid bid;
		if (latestBid != null) {
			bid = getRandomBid(latestBid, maxDistance);
		} else {
			// bid = utilitySpace.getDomain().getRandomBid();
			bid = getRandomBid();
		}

		double val = utilitySpace.getUtility(bid);

		return new BidDetails(bid, val);
	}

	/**
	 * Returns a bid with distance smaller than or equal to maxDistance from the
	 * reference bid.
	 * 
	 * 
	 * @param utilitySpace
	 * @param referencebid
	 * @param maxDistance
	 * @return
	 * @throws Exception
	 */
	Bid getRandomBid(Bid referencebid, int maxDistance) throws Exception {

		// the direction in which we make the random step.
		int[] directions = new int[numIssues + 1];

		// make a copy of the reference bid.
		HashMap<Integer, Value> oldValues = referencebid.getValues();
		HashMap<Integer, Value> newValues = new HashMap<Integer, Value>(
				oldValues); // Do NOT move this variable to outside the method,
							// cause this will lead to problems!!

		int distance = random.nextInt(maxDistance) + 1;

		for (int i = 0; i < distance; i++) {

			// pick a random index to increase or decrease:
			int issue = random.nextInt(numIssues) + 1;

			// we will increase or decrease this issue by 1

			// first determine the direction:
			if (directions[issue] == 0) { // we have to remain consistent. if
											// one time we increase a value, we
											// cannot decrease it the next time.
											// Therefore we store the direction
											// and re-use it.
				directions[issue] = 2 * random.nextInt(2) - 1; // set direction
																// to 1 or -1.
			}

			int lowestVal = ((IssueInteger) utilitySpace.getDomain()
					.getIssues().get(issue - 1)).getLowerBound(); // WARNING:
																	// the
																	// issues
																	// are
																	// numbered
																	// 1 to 30,
																	// but
																	// when
																	// calling
																	// getIssue,
																	// they are
																	// indexed
																	// with the
																	// values 0
																	// to 29
			int highestVal = ((IssueInteger) utilitySpace.getDomain()
					.getIssues().get(issue - 1)).getUpperBound();

			int oldValue = ((ValueInteger) oldValues.get(new Integer(issue)))
					.getValue();
			if (oldValue == highestVal && directions[issue] == 1
					|| oldValue == lowestVal && directions[issue] == -1) {
				i--;
				continue;
			}
			newValues.put(new Integer(issue), new ValueInteger(oldValue
					+ directions[issue]));

		}

		Bid newBid = new Bid(utilitySpace.getDomain(), newValues);
		return newBid;
	}

	/**
	 * Creates two children from bid1 and bid2, and repeats this until
	 * numChildren children have been created.
	 * 
	 * @param bid1
	 * @param bid2
	 * @param numChildren
	 * @return
	 * @throws Exception
	 */
	ArrayList<BidDetails> crossOver(BidDetails bid1, BidDetails bid2,
			int numChildren) throws Exception {

		HashMap<Integer, Value> vals1 = bid1.getBid().getValues();
		HashMap<Integer, Value> vals2 = bid2.getBid().getValues();

		// Note: we can set the load factor as high as we want because we are
		// sure that the number of entries will never exceed the number of
		// buckets, so re-hashing should never occur.
		HashMap<Integer, Value> newVals1 = new HashMap<Integer, Value>(
				numIssues, 2);
		HashMap<Integer, Value> newVals2 = new HashMap<Integer, Value>(
				numIssues, 2);

		ArrayList<BidDetails> children = new ArrayList(numChildren);

		for (int c = 0; c < numChildren / 2; c++) {
			for (Integer i = 1; i <= numIssues; i++) {
				if (random.nextBoolean()) {
					newVals1.put(i, vals1.get(i));
					newVals2.put(i, vals2.get(i));
				} else {
					newVals1.put(i, vals2.get(i));
					newVals2.put(i, vals1.get(i));
				}
			}

			Bid _child1 = new Bid(utilitySpace.getDomain(), newVals1);
			double value1 = utilitySpace.getUtility(_child1);
			BidDetails child1 = new BidDetails(_child1, value1);
			children.add(child1);

			Bid _child2 = new Bid(utilitySpace.getDomain(), newVals2);
			double value2 = utilitySpace.getUtility(_child2);
			BidDetails child2 = new BidDetails(_child2, value2);
			children.add(child2);
		}

		return children;
	}

	/**
	 * Creates two babies that are close enough to the reference bid. Assumes
	 * that bid1 and bid2 are also close enough to the reference bid.
	 * 
	 * First creates two children in the standard way, then calculates the
	 * distances of both if not both are close enough randomly swaps genes until
	 * it is achieved.
	 * 
	 * @param refBid
	 * @param maxDistance
	 * @param bid1
	 * @param bid2
	 * @param numChildren
	 * @return
	 * @throws Exception
	 */
	ArrayList<BidDetails> crossOver(Bid refBid, int maxDistance,
			BidDetails bid1, BidDetails bid2) throws Exception {

		HashMap<Integer, Value> refVals = refBid.getValues();

		HashMap<Integer, Value> vals1 = bid1.getBid().getValues();
		HashMap<Integer, Value> vals2 = bid2.getBid().getValues();

		// Note: we can set the load factor as high as we want because we are
		// sure that the number of entries will never exceed the number of
		// buckets, so re-hashing should never occur.
		HashMap<Integer, Value> newVals1 = new HashMap<Integer, Value>(
				numIssues, 2);
		HashMap<Integer, Value> newVals2 = new HashMap<Integer, Value>(
				numIssues, 2);

		ArrayList<BidDetails> children = new ArrayList(2);

		int totalDistance1 = 0; // the distance between child1 and the reference
								// bid.
		int totalDistance2 = 0; // the distance between child2 and the reference
								// bid.
		int[] distances1 = new int[numIssues + 1]; // the distance between
													// child1 and the reference
													// bid, for each issue.
		int[] distances2 = new int[numIssues + 1]; // the distance between
													// child1 and the reference
													// bid, for each issue.

		for (Integer i = 1; i <= numIssues; i++) {

			if (random.nextBoolean()) {
				newVals1.put(i, vals1.get(i));
				newVals2.put(i, vals2.get(i));
			} else {
				newVals1.put(i, vals2.get(i));
				newVals2.put(i, vals1.get(i));
			}

			// iteratively calculate the distances between the new children and
			// the reference bid.
			distances1[i] = Math.abs(((ValueInteger) newVals1.get(i))
					.getValue() - ((ValueInteger) refVals.get(i)).getValue());
			distances2[i] = Math.abs(((ValueInteger) newVals2.get(i))
					.getValue() - ((ValueInteger) refVals.get(i)).getValue());
			totalDistance1 += distances1[i];
			totalDistance2 += distances2[i];
		}

		int counter = 0;
		while (totalDistance1 > maxDistance || totalDistance2 > maxDistance) {

			int randomIndex = 0;

			if (totalDistance1 > totalDistance2) {

				do {
					randomIndex = random.nextInt(numIssues) + 1;
				} while (distances1[randomIndex] < distances2[randomIndex]); // search
																				// for
																				// an
																				// index
																				// for
																				// which
																				// child1
																				// is
																				// closer
																				// to
																				// ref
																				// than
																				// child2

			} else {
				do {
					randomIndex = random.nextInt(numIssues) + 1;
				} while (distances1[randomIndex] > distances2[randomIndex]); // search
																				// for
																				// an
																				// index
																				// for
																				// which
																				// child2
																				// is
																				// closer
																				// to
																				// ref
																				// than
																				// child1
			}

			// swap the two values.
			Value temp = newVals1.get(randomIndex);
			newVals1.put(randomIndex, newVals2.get(randomIndex));
			newVals2.put(randomIndex, temp);

			// Recalculate the distances of the new children, in 3 steps

			// 1.subtract the issue distances for the swapped issue from the
			// total distances
			totalDistance1 -= distances1[randomIndex];
			totalDistance2 -= distances2[randomIndex];

			// 2.swap the issue distances.
			int tempp = distances1[randomIndex];
			distances1[randomIndex] = distances2[randomIndex];
			distances2[randomIndex] = tempp;

			// 3.add the issue swapped distances again to the total distances.
			totalDistance1 += distances1[randomIndex];
			totalDistance2 += distances2[randomIndex];

			// SECURITY MEASURE TO MAKE SURE THAT WE DON'T LOOP FOR EVER.
			counter++;
			if (counter > 500) {
				System.out
						.println("GenAlg.crossOver() WARNING!!! There seems to be an infinite loop in the genetic algorithm!!");
				break;
			}

		}

		Bid _child1 = new Bid(utilitySpace.getDomain(), newVals1);
		double value1 = utilitySpace.getUtility(_child1);
		BidDetails child1 = new BidDetails(_child1, value1);
		children.add(child1);

		Bid _child2 = new Bid(utilitySpace.getDomain(), newVals2);
		double value2 = utilitySpace.getUtility(_child2);
		BidDetails child2 = new BidDetails(_child2, value2);
		children.add(child2);

		return children;
	}
}
