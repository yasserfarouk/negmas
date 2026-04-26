package genius.core.uncertainty;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

import genius.core.Bid;
import genius.core.bidding.BidDetails;
import genius.core.bidding.BidDetailsSorterUtility;
import genius.core.boaframework.SortedOutcomeSpace;

public class ComparisonGenerator {
	private SortedOutcomeSpace outcomeSpace;

	public ComparisonGenerator(SortedOutcomeSpace outcomeSpace) {
		this.outcomeSpace = outcomeSpace;
	}

	/**
	 * TODO DOC
	 * 
	 * @param uncertaintyPercentage
	 * @param seed
	 *            if nonzero, this seed is used to pick the random bids
	 * @return
	 */
	public BidRanking generateComparisonsByUncertaintyPercentage(
			double uncertaintyPercentage, long seed) {
		int amountOfOutcomes = (int) (uncertaintyPercentage
				* (outcomeSpace.getAllOutcomes().size()));
		return generateRankingByAmount(amountOfOutcomes, seed);
	}

	/**
	 * 
	 * @param amountOfOutcomes
	 *            number of outcomes needed.
	 * @param seed
	 *            if nonzero, this seed is used to pick the random bids
	 * @return amountOfOutcomes randomly picked bids, sorted from low to high
	 *         utility, NOTICE this algorithm first generates a list with ALL
	 *         bids, and iterates multiple times over this list, which can be
	 *         prohibitively expensive.
	 */
	public BidRanking generateRankingByAmount(int amountOfOutcomes, long seed) {
		List<BidDetails> selectedBidsWithUtilities = new ArrayList<BidDetails>();
		List<BidDetails> allBids = outcomeSpace.getAllOutcomes();
		BidDetails minBid = outcomeSpace.getMinBidPossible();
		BidDetails maxBid = outcomeSpace.getMaxBidPossible();
		double minUtil = minBid.getMyUndiscountedUtil();
		double maxUtil = maxBid.getMyUndiscountedUtil();
		int numberOfBids = amountOfOutcomes - 2;
		
		//Here we make sure that the max and min bid are in the ranking to avoid an agent cheating by
		//discovering utilities of bids added at the ranking extremities. We deal with the case <= 2 first.
		if (numberOfBids <= 0) {
			List<Bid> baseRanking = new ArrayList<Bid>();
			baseRanking.add(minBid.getBid());
			baseRanking.add(maxBid.getBid());
			return new BidRanking(baseRanking,minUtil,maxUtil);
		}
		allBids.remove(maxBid);
		allBids.remove(minBid);
		if (seed == 0) {
			Collections.shuffle(allBids);
		} else {
			Collections.shuffle(allBids, new Random(seed));
		}
		selectedBidsWithUtilities = allBids.subList(0, numberOfBids);
		Collections.sort(selectedBidsWithUtilities,
				new BidDetailsSorterUtility().reversed());
		List<Bid> bids = new ArrayList<Bid>();
		selectedBidsWithUtilities.stream()
				.forEach(bid -> bids.add(bid.getBid()));
		//Add min and max bids to the ranking.
		bids.add(0,minBid.getBid());
		bids.add(maxBid.getBid());
		return new BidRanking(bids, minUtil, maxUtil);	
	}

	public List<OutcomeComparison> generateComparisonsFromRankedOutcomeSet(
			List<BidDetails> selectedBids) {
		List<OutcomeComparison> comparisons = new ArrayList<OutcomeComparison>();
		for (int i = 0; i < selectedBids.size() - 1; i++) {
			comparisons.add(new OutcomeComparison(selectedBids.get(i),
					selectedBids.get(i + 1)));
		}
		return comparisons;
	}

	/**
	 * TODO DOC
	 * 
	 * @param amount
	 * @param error
	 * @param seed
	 *            if nonzero, this seed is used to pick the random bids
	 * @return
	 */
	public List<OutcomeComparison> generateComparisonsWithError(int amount,
			double error, long seed) {
		List<OutcomeComparison> comparisons = generateRankingByAmount(amount,
				seed).getPairwiseComparisons();
		int wrongComparisonsAmount = (int) (error * comparisons.size());
		for (int i = 0; i < wrongComparisonsAmount; i++) {
			if (comparisons.get(i).getComparisonResult() == -1)
				comparisons.get(i).setComparisonResult(+1);
			else
				comparisons.get(i).setComparisonResult(-1);
		}
		return comparisons;
	}

	public SortedOutcomeSpace getOutcomeSpace() {
		return outcomeSpace;
	}

	public List<BidDetails> getAllBids() {
		return outcomeSpace.getAllOutcomes();
	}
}
