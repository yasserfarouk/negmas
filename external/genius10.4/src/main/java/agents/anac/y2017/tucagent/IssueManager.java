package agents.anac.y2017.tucagent;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.issue.Issue;
import genius.core.timeline.Timeline;
import genius.core.utility.AdditiveUtilitySpace;

public class IssueManager {

	AdditiveUtilitySpace US;
	Bid maxBid;
	Timeline T;
	TreeMap<Double, Bid> AllPossibleBids;
	ArrayList<Object> myCurrentBiggerBids;
	double overallExpectedutil = 0.0;
	boolean FirstOfferGiven = false;
	Bid myNextBidIs = null;

	BayesianOpponentModel OMone;
	BayesianOpponentModel OMtwo;

	// KAINOURIA APO DW KAI PERA
	ArrayList<Double> GreedyRatioListOne;
	ArrayList<Double> GreedyRatioListTwo;
	List<Issue> AllIssues;
	public int numOfIssues;
	Bid opponentsOneLastBid = null;
	Bid opponentsTwoLastBid = null;
	Bid myBid;
	Bid firstMaxBid;

	// newwwwwww
	AgentID agentA = null;
	AgentID agentB = null;
	int deltaA = 0;
	int deltaB = 0;
	int roundCount = 0;
	double NSA = 0.0;
	double NSB = 0.0;
	double NS = 0.0;
	int numOfMyCommitedOffers = 1;
	double selfFactor = 0.0;
	double Eagerness = 0.5;
	double enviromFactor = 0.11;
	double concessionRate = 0.0;
	double myTotalWeight = 0.0;
	double maxBidUtil = 0.0;

	// even newer
	int similarityA = 0;
	int similarityB = 0;
	double probOFSimilarityA = 0.0;
	double probOFSimilarityB = 0.0;

	double BestOpponBid = 0.0;

	// KAINOURIA SUNARTISI

	public void findMySimilarAgent() { // psaxnw na vrw ton agent pou exei
										// similar weights me emena
		// prospelavnw ola ta issue
		for (int i = 0; i < numOfIssues; i++) {

			if (GreedyRatioListOne.get(i) >= 0.8 && GreedyRatioListOne.get(i) <= 1.2) {
				similarityA++;
			}

			if (GreedyRatioListTwo.get(i) >= 0.8 && GreedyRatioListOne.get(i) <= 1.2) {
				similarityB++;
			}
		}

		probOFSimilarityA = similarityA / numOfIssues;
		probOFSimilarityB = similarityB / numOfIssues;
	}

	// an den exw ousiastika discount factor einai san na to pernw dedomeno =1
	public double GetDiscountFactor() {
		if (this.US.getDiscountFactor() <= 0.001 || this.US.getDiscountFactor() > 1.0) {
			return 1.0;
		}
		return this.US.getDiscountFactor();
	}

	public void ProcessBid(AgentID opponent, Bid IncomingBid) throws Exception { // elengw
																					// an
																					// auto
																					// einai
																					// to
																					// kalutero
																					// bid
																					// p
																					// m
																					// exei
																					// kanei
																					// o
																					// adipalos

		// elenxw an einai to kalutero bid pou m exei er8ei
		if (this.US.getUtility(IncomingBid) > this.BestOpponBid) {
			BestOpponBid = this.US.getUtility(IncomingBid);
		}

		roundCount++;

		if (!AllPossibleBids.isEmpty()) {
			if (!AllPossibleBids.containsValue(IncomingBid)) { // an ayto to bid
																// den to eixa
																// upopsin to
																// vazw k auto
																// sti lsita mou
				AllPossibleBids.put(US.getUtility(IncomingBid), IncomingBid);
			}
		}

		findGreedyRatio();
		findMySimilarAgent();

		findNegotiationStatus(opponent, IncomingBid);

		// SET MY SELF FACTOR
		double DF = this.GetDiscountFactor();
		double Time = this.T.getTime();

		if (DF != 1) {
			Eagerness = 1 - DF;
		}

		selfFactor = 0.25 * ((1 / numOfMyCommitedOffers) + NS + Time + Eagerness);

		// SET CONCESSION RATE

		if (Time <= 0.2) { // start

			concessionRate = 0.0;

		} else if (Time >= 0.9) { // end

			concessionRate = 0.50;

		} else { // otherwise

			concessionRate = myTotalWeight * selfFactor + enviromFactor;
		}
	}

	// kathorismos tou threshold mou me vasi to offer pou tha ekana edw!!!
	public double CreateThreshold() throws Exception {

		double offer;
		double reservationValue = 0.6; // set my own reservation value

		double Time = this.T.getTime();

		if (Time >= 0.95) {
			offer = BestOpponBid;
		} else {
			offer = (maxBidUtil + (reservationValue - maxBidUtil) * concessionRate);
		}

		return offer;

	}

	// kainouria
	public void findNegotiationStatus(AgentID opponent, Bid IncomingBid) {

		double incomingUtil = US.getUtility(IncomingBid);

		if (opponent == agentA && opponent != null) { // gia ton 1 agent

			if (incomingUtil < US.getUtility(opponentsOneLastBid)) {
				deltaA = deltaA + 1;
			} else {
				deltaA = deltaA + 3;
			}

			NSA = (deltaA + 2 * roundCount) / 3 * roundCount;

		} else if (opponent == agentB && opponent != null) { // gia ton 2 agent

			if (incomingUtil < US.getUtility(opponentsOneLastBid)) {
				deltaB = deltaB + 1;
			} else {
				deltaB = deltaB + 3;
			}

			NSB = (deltaB + 2 * roundCount) / 3 * roundCount;
		}

		if (similarityA == similarityB) {

			NS = (NSA + NSB) / 2;

		} else if (similarityA > similarityB) {

			double weight1 = probOFSimilarityA;
			double weight2 = 1 - weight1;

			NS = (weight1 * NSA + weight2 * NSB) / 2;

		} else { // o b pio polu similar apo ton a

			double weight1 = probOFSimilarityB;
			double weight2 = 1 - weight1;

			NS = (weight1 * NSA + weight2 * NSB) / 2;
		}

	}

	// epistrefei to bid me to amesws mikrotero utillity
	public Bid GenerateBidWithAtleastUtilityOf(double MinUtility) {

		Map.Entry<Double, Bid> e = this.AllPossibleBids.ceilingEntry(MinUtility);

		if (e == null) {
			return this.maxBid;
		}

		return e.getValue();
	}

	public Bid GetMaxBid() {
		return maxBid;
	}

	public void findAllmyBidsPossible() throws Exception {

		Random random = new Random();
		int numOfPossibleBids = (int) US.getDomain().getNumberOfPossibleBids();

		for (int i = 0; i < numOfPossibleBids; i++) { // prospathw na vrw ola ta
														// pithana bids--> kanw
														// mia arxiki lista
			Bid randomBid = US.getDomain().getRandomBid(random);
			if ((!AllPossibleBids.containsKey(US.getUtility(randomBid)))
					|| (!AllPossibleBids.containsValue(randomBid))) {
				AllPossibleBids.put(US.getUtility(randomBid), randomBid);
			}
		}

	}

	// KAINOURIA SUNARTISI
	public void findGreedyRatio() { // se kathe guro enhmerwnetai afou allazei
									// to expected weight

		for (int i = 0; i < numOfIssues; i++) { // ousiastika kanw update twn
												// duo greedy factor se kathe
												// guro

			double myWeight = US.getWeight(i);
			double oppOneWeight = OMone.getExpectedWeight(i);
			double oppTwoWeight = OMtwo.getExpectedWeight(i);

			double greedyFactorOne = myWeight / oppOneWeight;
			double greedyFactorTwo = myWeight / oppTwoWeight;

			GreedyRatioListOne.set(i, greedyFactorOne);// exw duo listes me ta r
														// k r' pou ananewnodai
														// sunexeia
			GreedyRatioListTwo.set(i, greedyFactorTwo);

		}
	}

	public IssueManager(AdditiveUtilitySpace US, Timeline T) throws Exception {
		this.T = T;
		this.US = US;

		try {
			double maxBidUtil = US.getUtility(this.maxBid); // pernei to utility
															// tou max bid
			if (maxBidUtil == 0.0) {
				this.maxBid = this.US.getMaxUtilityBid();
			}
		} catch (Exception e) {
			try {
				this.maxBid = this.US.getMaxUtilityBid();
			} catch (Exception var5_7) {
				// empty catch block
			}
		}

		myNextBidIs = maxBid;
		AllPossibleBids = new TreeMap<Double, Bid>();
		myCurrentBiggerBids = new ArrayList<Object>();
		myCurrentBiggerBids.add(maxBid);
		AllPossibleBids.put(US.getUtility(maxBid), maxBid);
		findAllmyBidsPossible();

		// kainouria apo dw k pera
		myBid = maxBid;
		GreedyRatioListOne = new ArrayList<Double>();
		GreedyRatioListTwo = new ArrayList<Double>();
		AllIssues = new ArrayList<Issue>();
		AllIssues = US.getDomain().getIssues();
		numOfIssues = AllIssues.size();
		opponentsOneLastBid = US.getMinUtilityBid();
		opponentsTwoLastBid = US.getMinUtilityBid();

		// newwwwwwwwwwwwwww
		maxBidUtil = US.getUtility(maxBid);
		findAllmyBidsPossible();

		for (int i = 0; i < numOfIssues; i++) {

			myTotalWeight = (myTotalWeight + US.getWeight(i)) / 2;

		}

	}
}