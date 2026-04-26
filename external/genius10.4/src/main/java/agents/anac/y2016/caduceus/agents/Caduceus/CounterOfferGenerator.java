package agents.anac.y2016.caduceus.agents.Caduceus;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;

import agents.anac.y2016.caduceus.agents.Caduceus.sanity.Pair;
import agents.anac.y2016.caduceus.agents.Caduceus.sanity.SaneBid;
import agents.anac.y2016.caduceus.agents.Caduceus.sanity.SaneIssue;
import agents.anac.y2016.caduceus.agents.Caduceus.sanity.SaneValue;
import genius.core.Bid;
import genius.core.BidIterator;

/**
 * Created by burakatalay on 20/03/16.
 */
public class CounterOfferGenerator {
	Bid nashBid;
	Caduceus party;
	double concessionStep = 0;
	private static final int NUMBER_OF_ROUNDS_FOR_CONCESSION = 10;
	int vectorSize = 0;
	private ArrayList<Bid> allPossibleBids = new ArrayList<>();
	private double[][] bidSpace;

	public CounterOfferGenerator(Bid nashBid, Caduceus party,
			double concessionStep) {
		this.nashBid = nashBid;
		this.party = party;
		this.concessionStep = concessionStep;
		this.vectorSize = this.nashBid.getIssues().size(); // FIXME: burak fix
															// this
		this.calculateAllPossibleBids();
		bidSpace = new double[allPossibleBids.size()][vectorSize];
		this.vectorizeAll();

	}

	public CounterOfferGenerator(Bid nashBid, Caduceus party) { // default
																// constructor
																// with linear
																// conceding
		this(nashBid, party, 0.2);

		try {
			this.concessionStep = (1.0 - party.getUtilitySpace().getUtility(
					this.nashBid))
					/ CounterOfferGenerator.NUMBER_OF_ROUNDS_FOR_CONCESSION;

			System.out.println("Concession Step: " + this.concessionStep);
		} catch (Exception e) {
			e.printStackTrace();
			System.out.println("Init error CFG");
		}

	}

	private double[] getUnitVector() throws Exception { // concession vector

		Bid maxBid = this.party.getUtilitySpace().getMaxUtilityBid();
		System.out.println("My max bid utility: "
				+ this.party.getMySaneUtilitySpace().getDiscountedUtility(
						maxBid, party.discountFactor,
						party.getTimeLine().getTime()));
		double[] maxBidPoint = this.vectorizeBid(maxBid);
		System.out.println("Max bid point: " + Arrays.toString(maxBidPoint));

		double[] nashPoint = this.vectorizeBid(nashBid);
		System.out.println("Nash point: " + Arrays.toString(nashPoint));

		double[] unitVector = UtilFunctions.calculateUnitVector(maxBidPoint,
				nashPoint);
		System.out.println("Unit vector: " + Arrays.toString(nashPoint));
		return unitVector;
	}

	private void vectorizeAll() {
		int index = 0;
		for (Bid bid : allPossibleBids) {
			double[] point = this.vectorizeBid(bid);
			this.bidSpace[index] = point;

			index++;
		}
	}

	private void calculateAllPossibleBids() {
		BidIterator bidIterator = new BidIterator(this.party.getUtilitySpace()
				.getDomain());
		while (bidIterator.hasNext()) {
			Bid currentBid = bidIterator.next();
			allPossibleBids.add(currentBid);
		}
	}

	private double[] vectorizeBid(Bid bid) {
		double[] point = new double[this.vectorSize];

		SaneBid saneBid = new SaneBid(bid, this.party.getMySaneUtilitySpace());

		Iterator<Pair<SaneIssue, SaneValue>> iterator = saneBid.getIterator();

		int issueIndex = 0;
		while (iterator.hasNext()) {
			Pair<SaneIssue, SaneValue> p = iterator.next();
			SaneIssue saneIssue = p.first;
			SaneValue saneValue = p.second;

			point[issueIndex] = saneIssue.weight * saneValue.utility;

			issueIndex++;
		}
		point = UtilFunctions.normalize(point);
		point = UtilFunctions.multiply(point, 10); // expand the space
		return point;
	}

	public Bid generateBid(double concessionRate) throws Exception { // start by
																		// 1
		Bid maxBid = this.party.getUtilitySpace().getMaxUtilityBid();
		double[] maxBidPoint = this.vectorizeBid(maxBid);

		double delta = concessionRate;
		System.out.println("Delta: " + delta);
		double[] unitVector = this.getUnitVector();

		double[] concessionDelta = UtilFunctions.multiply(unitVector, delta);
		System.out.println("Concession delta: "
				+ Arrays.toString(concessionDelta));

		double[] concessionPoint = UtilFunctions.add(maxBidPoint,
				concessionDelta);
		System.out.println("Concession point: "
				+ Arrays.toString(concessionPoint));

		Bid bid = this.getBidCloseToConcessionPoint(concessionPoint);
		double util = this.party.getMySaneUtilitySpace().getDiscountedUtility(
				bid, party.discountFactor, party.getTimeLine().getTime());

		return bid;
	}

	private Bid getBidCloseToConcessionPoint(double[] concessionPoint)
			throws Exception {
		Bid maxBid = this.party.getUtilitySpace().getMaxUtilityBid();
		double[] maxBidPoint = this.vectorizeBid(maxBid);

		double[] distances = new double[this.bidSpace.length];

		for (int i = 0; i < distances.length; i++) {
			double[] bidPoint = this.bidSpace[i];

			distances[i] = UtilFunctions.getEuclideanDistance(concessionPoint,
					bidPoint);
		}

		double minDistance = distances[0];
		int minDistanceIndex = 0;
		for (int i = 0; i < distances.length; i++) {
			double d = distances[i];
			if (!UtilFunctions.equals(this.bidSpace[i], maxBidPoint, 0.1)
					&& d < minDistance) {
				minDistanceIndex = i;
				minDistance = d;
			}
		}
		Bid bid = this.allPossibleBids.get(minDistanceIndex);
		System.out.println("Selected Concession point: "
				+ Arrays.toString(this.bidSpace[minDistanceIndex]));
		System.out.println("Utility for me: "
				+ this.party.getMySaneUtilitySpace().getDiscountedUtility(bid,
						party.discountFactor, party.getTimeLine().getTime()));
		return bid;
	}

}
