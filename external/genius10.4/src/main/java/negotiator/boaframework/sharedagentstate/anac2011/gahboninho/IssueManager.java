package negotiator.boaframework.sharedagentstate.anac2011.gahboninho;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map.Entry;

import genius.core.Bid;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.OMStrategy;
import genius.core.issue.ISSUETYPE;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.IssueInteger;
import genius.core.issue.IssueReal;
import genius.core.issue.Value;
import genius.core.issue.ValueDiscrete;
import genius.core.issue.ValueInteger;
import genius.core.issue.ValueReal;
import genius.core.timeline.TimeLineInfo;
import genius.core.utility.AdditiveUtilitySpace;

import java.util.Random;
import java.util.TreeMap;

public class IssueManager {
	// world state :
	private TimeLineInfo T;
	private AdditiveUtilitySpace US;
	private int TotalBiddingPossibilities; // amount of all possible bidsd
	// oponent properties :
	private Bid OpponentBestBid; // the first bid opponent gave us
	private double MyUtilityOnOpponentBestBid = 0;
	private double Noise = 0.4; // 0 to 1 - Tells us how varied the opponent's
								// bids are, lately
	// My State:
	private double NextOfferUtility = 1;
	private boolean FirstOfferGiven = false;
	private TreeMap<Double, Bid> Bids;
	private HashMap<Issue, ArrayList<Value>> relevantValuesPerIssue = new HashMap<Issue, ArrayList<Value>>();
	private Bid maxBid = null;
	private final int PlayerCount = 8; // if player count is 10, then we
	private GahboninhoOM OM;
	private boolean InFrenzy = false;
	private TreeMap<Integer, TreeMap<String, Integer>> IncomingValueAppeareancesByIssueNum = new TreeMap<Integer, TreeMap<String, Integer>>();
	private TreeMap<Integer, TreeMap<String, Integer>> OutgoingValueAppeareancesByIssueNum = new TreeMap<Integer, TreeMap<String, Integer>>(); // bids
	private TreeMap<Integer, Integer> DifferentValuesCountPerIssueNum = new TreeMap<Integer, Integer>();
	private int CountdownToStatisticsRefreshing = 20;
	private double PreviousAverageDistance = 0.2; // 0 to 1
	private double PreviousCountdownOpponentBestBidUtil = 1;
	private double BestEverOpponentBidUtil = 0;
	private double WorstOpponentBidEvaluatedOpponentUtil = 1;
	private double PrevWorstOpponentBidEvaluatedOpponentUtil = 1;
	private double PrevRoundWorstOpponentBidEvaluatedOpponentUtil = 1;
	private Bid BestEverOpponentBid = null;
	private int OutgoingBidsCount = 0;
	private Double BidsCreationTime = 0.0;
	private double EffectiveDomainBids = 1;
	private final boolean TEST_EQUIVALENCE = false;
	private Random random100;
	private Random random200;
	private Random random300;
	private Random random400;
	private NegotiationSession negotiationSession;
	private OMStrategy omStrategy;

	public TreeMap<Double, Bid> getBids() {
		return Bids;
	}

	public Bid getBestEverOpponentBid() {
		return BestEverOpponentBid;
	}

	public Bid GetMaxBidWithNoCost() throws Exception {
		Bid maxBid = this.US.getDomain().getRandomBid(random100);
		Bid justBidding = this.US.getDomain().getRandomBid(random100);

		for (Issue issue : this.US.getDomain().getIssues()) {
			double tmpUtil;
			double maxUtil = 0;
			int maxUtilValIndex = 0;

			switch (issue.getType()) {

			case INTEGER:

				IssueInteger integerIssue = (IssueInteger) issue;

				justBidding = justBidding.putValue(issue.getNumber(), new ValueInteger(integerIssue.getUpperBound()));
				maxUtil = US.getUtility(justBidding);

				justBidding = justBidding.putValue(issue.getNumber(), new ValueInteger(integerIssue.getLowerBound()));
				tmpUtil = US.getUtility(justBidding);

				if (maxUtil > tmpUtil)
					maxBid = maxBid.putValue(issue.getNumber(), new ValueInteger(integerIssue.getUpperBound()));
				else
					maxBid = maxBid.putValue(issue.getNumber(), new ValueInteger(integerIssue.getLowerBound()));

				break;

			case REAL:

				IssueReal realIssue = (IssueReal) issue;

				justBidding = justBidding.putValue(issue.getNumber(), new ValueReal(realIssue.getUpperBound()));
				maxUtil = US.getUtility(justBidding);

				justBidding = justBidding.putValue(issue.getNumber(), new ValueReal(realIssue.getLowerBound()));
				tmpUtil = US.getUtility(justBidding);

				if (maxUtil > tmpUtil)
					maxBid = maxBid.putValue(issue.getNumber(), new ValueReal(realIssue.getUpperBound()));
				else
					maxBid = maxBid.putValue(issue.getNumber(), new ValueReal(realIssue.getLowerBound()));

				break;
			case DISCRETE:
				IssueDiscrete discreteIssue = (IssueDiscrete) issue;
				int size = discreteIssue.getNumberOfValues();
				for (int i = 0; i < size; i++) {
					justBidding = justBidding.putValue(issue.getNumber(), discreteIssue.getValue(i));
					tmpUtil = US.getUtility(justBidding);
					if (tmpUtil > maxUtil) {
						maxUtilValIndex = i;
						maxUtil = tmpUtil;
					}
				}

				maxBid = maxBid.putValue(issue.getNumber(), discreteIssue.getValue(maxUtilValIndex));
				break;
			}
		}

		return maxBid;
	}

	// fill utility-to-bid map here:
	public IssueManager(NegotiationSession negoSession, TimeLineInfo T, GahboninhoOM om) {
		this.negotiationSession = negoSession;

		if (TEST_EQUIVALENCE) {
			random100 = new Random(100);
			random200 = new Random(200);
			random300 = new Random(300);
			random400 = new Random(400);
		} else {
			random100 = new Random();
			random200 = new Random();
			random300 = new Random();
			random400 = new Random();
		}

		this.T = T;
		this.US = (AdditiveUtilitySpace) negoSession.getUtilitySpace();
		this.OM = om;
		try {
			maxBid = GetMaxBidWithNoCost(); // try sparing the brute force
			double maxBidUtil = US.getUtility(maxBid);
			if (maxBidUtil == 0) // in case cost comes into play
				this.maxBid = this.US.getMaxUtilityBid(); // use only if the
															// simpler function
															// won't work

		} catch (Exception e) {
			try {
				this.maxBid = this.US.getMaxUtilityBid();
			} catch (Exception e2) {
			}
		}

		Bids = new TreeMap<Double, Bid>();

		for (int i = 0; i < US.getDomain().getIssues().size(); ++i) {
			Issue I = (Issue) US.getDomain().getIssues().get(i);
			if (I.getType() == ISSUETYPE.DISCRETE) {
				IssueDiscrete ID = (IssueDiscrete) I;

				DifferentValuesCountPerIssueNum.put(ID.getNumber(), ID.getNumberOfValues());
				OutgoingValueAppeareancesByIssueNum.put(ID.getNumber(), new TreeMap<String, Integer>());
			} else if (I.getType() == ISSUETYPE.REAL) {
				DifferentValuesCountPerIssueNum.put(I.getNumber(), (int) DiscretisationSteps);
				OutgoingValueAppeareancesByIssueNum.put(I.getNumber(), new TreeMap<String, Integer>());
			} else if (I.getType() == ISSUETYPE.INTEGER) {
				IssueInteger II = (IssueInteger) I;
				DifferentValuesCountPerIssueNum.put(I.getNumber(),
						Math.min((int) DiscretisationSteps, II.getUpperBound() - II.getLowerBound() + 1));

				OutgoingValueAppeareancesByIssueNum.put(I.getNumber(), new TreeMap<String, Integer>());
			}
		}
		ClearIncommingStatistics();
	}

	void ClearIncommingStatistics() {
		for (Issue I : US.getDomain().getIssues())
			IncomingValueAppeareancesByIssueNum.put(I.getNumber(), new TreeMap<String, Integer>());

	}

	public Bid getMaxBid() {
		return maxBid;
	}

	public double GetDiscountFactor() {
		if (US.getDiscountFactor() <= 0.001 || US.getDiscountFactor() > 1)
			return 1;
		return US.getDiscountFactor();
	}

	private void addPossibleValue(Issue issue, Value val) {
		if (!this.relevantValuesPerIssue.containsKey(issue)) {
			this.relevantValuesPerIssue.put(issue, new ArrayList<Value>());
		}

		int randIndex = 0;
		if (this.relevantValuesPerIssue.get(issue).size() > 0)
			randIndex = Math.abs(random200.nextInt()) % this.relevantValuesPerIssue.get(issue).size();
		this.relevantValuesPerIssue.get(issue).add(randIndex, val);
	}

	final double DiscretisationSteps = 20; // minimum 2

	private void buildIssueValues(Bid firstOppBid) throws Exception {

		Bid justBidding = this.US.getDomain().getRandomBid(random300);

		for (Issue issue : this.US.getDomain().getIssues()) {
			int AddedValues = 0;

			justBidding = justBidding.putValue(issue.getNumber(), firstOppBid.getValue(issue.getNumber()));
			double utilityWithOpp = this.US.getEvaluation(issue.getNumber(), justBidding);

			switch (issue.getType()) {
			case INTEGER:
				IssueInteger intIssue = (IssueInteger) issue;

				int iStep;
				int totalSteps = (int) Math.min(DiscretisationSteps - 1,
						intIssue.getUpperBound() - intIssue.getLowerBound());
				iStep = Math.max(1, (int) ((intIssue.getUpperBound() - intIssue.getLowerBound()) / totalSteps));

				for (int i = intIssue.getLowerBound(); i <= intIssue.getUpperBound(); i += iStep) {
					justBidding = justBidding.putValue(issue.getNumber(), new ValueInteger(i));
					double utilityWithCurrent = this.US.getEvaluation(issue.getNumber(), justBidding);

					// Only see it as a possible value if it is better for us
					// than the opponent offer
					if (utilityWithCurrent >= utilityWithOpp) {
						this.addPossibleValue(issue, new ValueInteger(i));
					}
				}

				AddedValues += Math.abs(intIssue.getUpperBound() - intIssue.getLowerBound());
				break;
			case REAL:

				IssueReal realIssue = (IssueReal) issue;
				double oneStep = (realIssue.getUpperBound() - realIssue.getLowerBound()) / (DiscretisationSteps - 1);
				for (double curr = realIssue.getLowerBound(); curr <= realIssue.getUpperBound(); curr += oneStep) {
					justBidding = justBidding.putValue(issue.getNumber(), new ValueReal(curr));
					double utilityWithCurrent = this.US.getEvaluation(issue.getNumber(), justBidding);
					// Only see it as a possible value if it is better for us
					// than the opponent offer
					if (utilityWithCurrent >= utilityWithOpp) {
						this.addPossibleValue(issue, new ValueReal(curr));
						AddedValues += 1000;
					}
				}

				break;
			case DISCRETE:
				IssueDiscrete discreteIssue = (IssueDiscrete) issue;
				int size = discreteIssue.getNumberOfValues();
				for (int i = 0; i < size; i++) {
					ValueDiscrete curr = discreteIssue.getValue(i);
					justBidding = justBidding.putValue(issue.getNumber(), curr);
					double utilityWithCurrent = this.US.getEvaluation(issue.getNumber(), justBidding);
					// Only see it as a possible value if it is better for us
					// than the opponent offer
					if (utilityWithCurrent >= utilityWithOpp) {
						this.addPossibleValue(issue, curr);
						AddedValues += 1;
					}

				}
				break;
			}

			EffectiveDomainBids *= AddedValues;
		}
	}

	public void learnBids(Bid firstOppBid) throws Exception {
		this.buildIssueValues(firstOppBid);

		double startTime = T.getTime();

		// very hard to deal with hash map, so copy it to arraylist:
		Iterator<Entry<Issue, ArrayList<Value>>> MyIssueIterator = relevantValuesPerIssue.entrySet().iterator();
		while (MyIssueIterator.hasNext())
			IssueEntries.add(MyIssueIterator.next());

		// if there is a discount factor, don't take your time searching for
		// bids
		BuildBid(new HashMap<Integer, Value>(), 0, 0.05 * Math.pow(GetDiscountFactor(), 0.6) + startTime);
		setBidsCreationTime(T.getTime() - startTime);

		// if there are about 5000 turns for the opponent, I expect him to be
		// able to
		// give a good bid every few turns
		NoiseDecreaseRate = 0.01 * EffectiveDomainBids / (400);
		NoiseDecreaseRate = Math.min(0.015, NoiseDecreaseRate); // beware of a
																// too large
																// rate
		NoiseDecreaseRate = Math.max(0.003, NoiseDecreaseRate);

	}

	private ArrayList<Entry<Issue, ArrayList<Value>>> IssueEntries = new ArrayList<Entry<Issue, ArrayList<Value>>>();

	int UtilityCollisionsLeft = 200000;

	private void BuildBid(HashMap<Integer, Value> B, int EntrySetIndex, double EndTime) {
		// TODO : build only bids with at least X util.
		// after some time, when we need lower utilities, re-build bid map, only
		// this time consider
		// opponent's preferences: just build a method that tells us if one bid
		// is preferrable (from opponent's point of view)
		// and if the less preferable one gives us less utility than the other
		// bid, simply remove the bid from map

		if (T.getTime() < EndTime) {
			if (EntrySetIndex < IssueEntries.size()) {
				Entry<Issue, ArrayList<Value>> currentEntry = IssueEntries.get(EntrySetIndex);
				for (Value v : currentEntry.getValue()) {
					B.put(currentEntry.getKey().getNumber(), v);
					BuildBid(B, EntrySetIndex + 1, EndTime);
				}
			} else {
				try {

					Bid newBid = new Bid(US.getDomain(), new HashMap<Integer, Value>(B));

					double BidUtil = US.getUtility(newBid);

					while (UtilityCollisionsLeft > 0 && Bids.containsKey(BidUtil)) {
						--UtilityCollisionsLeft;
						BidUtil -= 0.002 / (random400.nextInt() % 9999999);
					}

					Bids.put(BidUtil, newBid);
				} catch (Exception e) {
					e.printStackTrace();
				}
			}

		}
	}

	public double CompromosingFactor = 0.95;
	// returns the minimum utility we want in the next offer we send

	double TimeDiffStart = -1;
	double TimeDiffEnd = -1;
	double RoundCountBeforePanic = 1;

	double PrevRecommendation = 1;
	double MaximalUtilityDecreaseRate = 0.0009;
	private double minimumUtilForAcceptance;

	public double GetNextRecommendedOfferUtility() {
		if (FirstOfferGiven == false) {
			FirstOfferGiven = true;
			return 1;
		}

		double Time = T.getTime();
		double Min = Math.pow(GetDiscountFactor(), 2 * Time);
		double Max = Min * (1 + 6 * getNoise() * Math.pow(GetDiscountFactor(), 5));

		if (Time < 0.85 * GetDiscountFactor())
			Min *= Math.max(BestEverOpponentBidUtil, 0.9125);
		else if (Time <= 0.92 * GetDiscountFactor()) {
			CompromosingFactor = 0.94;
			Min *= Math.max(BestEverOpponentBidUtil, 0.84); // never ever accept
															// an offer with
															// less than that
															// utility

			Max /= Min; // slow down the decreasing

		} else if (Time <= 0.94 * GetDiscountFactor()) {
			CompromosingFactor = 0.93;
			Min *= Math.max(BestEverOpponentBidUtil, 0.775);
			Max /= Min;
		} else if (Time <= 0.985 * Min && (BestEverOpponentBidUtil <= 2 * (1.0 / PlayerCount) || // never
																									// accept
																									// an
																									// offer
																									// with
																									// less
																									// utility
																									// than
																									// that
				Time <= (1 - 3 * (TimeDiffEnd - TimeDiffStart) / RoundCountBeforePanic))) {
			CompromosingFactor = 0.91;
			MaximalUtilityDecreaseRate = 0.001;

			Min *= Math.max(BestEverOpponentBidUtil, 0.7);
			Max /= Min;

			TimeDiffEnd = TimeDiffStart = Time;
		} else if (Time <= 0.9996 && (BestEverOpponentBidUtil <= 2 * (1.0 / PlayerCount) || // never
																							// accept
																							// an
																							// offer
																							// with
																							// less
																							// utility
																							// than
																							// that
				Time <= (1 - 3 * (TimeDiffEnd - TimeDiffStart) / RoundCountBeforePanic))) // until
																							// last
																							// few
																							// rounds
		{
			TimeDiffEnd = Time;
			++RoundCountBeforePanic;

			MaximalUtilityDecreaseRate = 0.001 + 0.01 * (Time - 0.985) / (0.015);

			// MaximalUtilityDecreaseRate = 0.0018;
			// MaximalUtilityDecreaseRate = 0.0009;

			if (3 * (1.0 / PlayerCount) > BestEverOpponentBidUtil) {
				Min *= 3.5 * (1.0 / PlayerCount);
				CompromosingFactor = 0.8;
			} else {
				Min *= BestEverOpponentBidUtil;
				CompromosingFactor = 0.95;
			}

			Max /= Min;
		} else {
			CompromosingFactor = 0.92;

			// as low as I can go!
			if (BestEverOpponentBidUtil < 2 * (1.0 / PlayerCount)) {
				Min = 2 * (1.0 / PlayerCount); // 0.25 if 8 players
				Max = 1;
			} else {
				Max = Min = BestEverOpponentBidUtil;
				setInFrenzy(true);
			}
			MaximalUtilityDecreaseRate = 1;
		}

		// the more eager the opponent is to settle, the slower we give up our
		// utility.
		// the higher the discount factor loss, the faster we give it up

		Max = Math.max(Max, Min);

		NextOfferUtility = Math.min(1, Max - (Max - Min) * T.getTime());

		// slow down the change:
		if (NextOfferUtility + MaximalUtilityDecreaseRate < PrevRecommendation)
			NextOfferUtility = PrevRecommendation - MaximalUtilityDecreaseRate;
		else if (NextOfferUtility - 0.005 > PrevRecommendation)
			NextOfferUtility = PrevRecommendation + 0.005;

		PrevRecommendation = NextOfferUtility;
		return NextOfferUtility;
	}

	public double GetMinimumUtilityToAccept() // changes over time
	{
		double util1 = CompromosingFactor;
		double util2 = GetNextRecommendedOfferUtility();
		return util1 * util2;
	}

	public double getMinimumUtilForAcceptance() {
		return minimumUtilForAcceptance;
	}

	public void setMinimumUtilForAcceptance(double minimumUtilForAcceptance) {
		this.minimumUtilForAcceptance = minimumUtilForAcceptance;
	}

	int CountdownToNoiseReestimation;

	String EstimateValue(Issue I, Value v) throws Exception {
		switch (I.getType()) {
		case DISCRETE:
			ValueDiscrete DV = (ValueDiscrete) v;
			return DV.getValue();
		case INTEGER:
			int ValueIndex = 0;
			IssueInteger II = (IssueInteger) I;
			ValueInteger IV = (ValueInteger) v;
			double Step = II.getUpperBound() - II.getLowerBound();
			if (Step != 0) {
				int totalSteps = (int) Math.min(DiscretisationSteps, II.getUpperBound() - II.getLowerBound() + 1);
				Step /= totalSteps;
				ValueIndex = (int) (IV.getValue() / Step);
			}
			return String.valueOf(ValueIndex);
		case REAL:
			IssueReal RI = (IssueReal) I;
			ValueReal RV = (ValueReal) v;
			double StepR = RI.getUpperBound() - RI.getLowerBound();
			ValueIndex = 0;
			if (StepR != 0) {
				StepR /= DiscretisationSteps;
				ValueIndex = (int) (RV.getValue() / StepR);
			}
			return String.valueOf(ValueIndex);
		}

		throw new Exception("illegal issue");
	}

	public void AddMyBidToStatistics(Bid OutgoingBid) throws Exception {
		++OutgoingBidsCount;
		for (Issue I : US.getDomain().getIssues()) {
			String bidValueEstimation = EstimateValue(I, OutgoingBid.getValue(I.getNumber()));
			if (OutgoingValueAppeareancesByIssueNum.get(I.getNumber()).containsKey(bidValueEstimation)) {
				OutgoingValueAppeareancesByIssueNum.get(I.getNumber()).put(bidValueEstimation,
						OutgoingValueAppeareancesByIssueNum.get(I.getNumber()).get(bidValueEstimation) + 1);
			} else {
				OutgoingValueAppeareancesByIssueNum.get(I.getNumber()).put(bidValueEstimation, 1);
			}
		}
	}

	double NoiseDecreaseRate = 0.01;
	final int CountdownLength = 20;

	// receiveMessage noise here
	public void ProcessOpponentBid(Bid IncomingBid) throws Exception {
		if (CountdownToStatisticsRefreshing > 0) {
			if (US.getUtility(IncomingBid) > BestEverOpponentBidUtil) {
				BestEverOpponentBidUtil = US.getUtility(IncomingBid);
				BestEverOpponentBid = IncomingBid;
				Bids.put(BestEverOpponentBidUtil, BestEverOpponentBid);
			}

			double getopUtil = OM.EvaluateOpponentUtility(IncomingBid);

			if (PrevRoundWorstOpponentBidEvaluatedOpponentUtil < getopUtil)
				PrevRoundWorstOpponentBidEvaluatedOpponentUtil = getopUtil;
			if (WorstOpponentBidEvaluatedOpponentUtil > getopUtil) {
				WorstOpponentBidEvaluatedOpponentUtil = getopUtil;

			}

			--CountdownToStatisticsRefreshing;
			for (Issue I : US.getDomain().getIssues()) {
				String bidValueEstimation = EstimateValue(I, IncomingBid.getValue(I.getNumber()));
				if (IncomingValueAppeareancesByIssueNum.get(I.getNumber()).containsKey(bidValueEstimation)) {
					IncomingValueAppeareancesByIssueNum.get(I.getNumber()).put(bidValueEstimation,
							IncomingValueAppeareancesByIssueNum.get(I.getNumber()).get(bidValueEstimation) + 1);
				} else {
					IncomingValueAppeareancesByIssueNum.get(I.getNumber()).put(bidValueEstimation, 1);
				}
			}
		} else {
			double CurrentSimilarity = 0;

			for (Issue I : US.getDomain().getIssues()) {
				for (String val : OutgoingValueAppeareancesByIssueNum.get(I.getNumber()).keySet()) {
					if (IncomingValueAppeareancesByIssueNum.get(I.getNumber()).containsKey(val)) {
						float outgoingVal = ((float) (OutgoingValueAppeareancesByIssueNum.get(I.getNumber()).get(val)))
								/ OutgoingBidsCount;
						float incomingVal = (((float) (IncomingValueAppeareancesByIssueNum.get(I.getNumber()).get(val)))
								/ CountdownLength);
						float diff = outgoingVal - incomingVal;
						float diffSqr = diff * diff; // 0 to 1

						CurrentSimilarity += (1.0 / US.getDomain().getIssues().size())
								* (1.0 / DifferentValuesCountPerIssueNum.get(I.getNumber())) * (1 - diffSqr);
					}
				}
			}

			if (CurrentSimilarity > PreviousAverageDistance) {
				setNoise(getNoise() + 0.05); // opponent is trying harder to
												// search
			} else if (BestEverOpponentBidUtil < PreviousCountdownOpponentBestBidUtil
					|| WorstOpponentBidEvaluatedOpponentUtil < PrevWorstOpponentBidEvaluatedOpponentUtil) {
				setNoise(getNoise() + NoiseDecreaseRate); // Apparently, the
															// opponent just
															// gave up some of
															// his util
			} else {
				setNoise(getNoise() - NoiseDecreaseRate);

				if (PrevRoundWorstOpponentBidEvaluatedOpponentUtil > WorstOpponentBidEvaluatedOpponentUtil * 1.2)
					setNoise(getNoise() - NoiseDecreaseRate);
				if (CurrentSimilarity * 1.1 < PreviousAverageDistance)
					setNoise(getNoise() - NoiseDecreaseRate);
			}
			// if (CurrentSimilarity > PreviousAverageDistance ||
			// BestEverOpponentBidUtil > PreviousCountdownOpponentBestBidUtil)
			// {
			// Noise += 0.02;
			// }
			// else
			// {
			// Noise -= 0.02;
			// }

			// if (CurrentSimilarity > PreviousAverageDistance ||
			// BestEverOpponentBidUtil > PreviousCountdownOpponentBestBidUtil)
			// {
			// Noise += 0.02;
			// }
			// else if (CurrentSimilarity < PreviousAverageDistance &&
			// BestEverOpponentBidUtil < PreviousCountdownOpponentBestBidUtil)
			// {
			// Noise -= 0.06;
			// }
			// else
			// Noise -= 0.005;

			setNoise(Math.min(Math.max(getNoise(), 0), 1));
			PreviousAverageDistance = CurrentSimilarity;
			CountdownToStatisticsRefreshing = CountdownLength;
			PreviousCountdownOpponentBestBidUtil = BestEverOpponentBidUtil;
			PrevRoundWorstOpponentBidEvaluatedOpponentUtil = 1;
			PrevWorstOpponentBidEvaluatedOpponentUtil = WorstOpponentBidEvaluatedOpponentUtil;
			ClearIncommingStatistics();
		}
	}

	public Bid GenerateBidWithAtleastUtilityOf(double MinUtility) {
		Entry<Double, Bid> selectedBid = null;
		selectedBid = Bids.ceilingEntry(MinUtility);
		if (selectedBid == null) {
			return this.maxBid;
		}
		return selectedBid.getValue();
	}

	public double getNoise() {
		return Noise;
	}

	public void setNoise(double noise) {
		Noise = noise;
	}

	public boolean getInFrenzy() {
		return InFrenzy;
	}

	public void setInFrenzy(boolean inFrenzy) {
		InFrenzy = inFrenzy;
	}

	public double getMyUtilityOnOpponentBestBid() {
		return MyUtilityOnOpponentBestBid;
	}

	public void setMyUtilityOnOpponentBestBid(double myUtilityOnOpponentBestBid) {
		MyUtilityOnOpponentBestBid = myUtilityOnOpponentBestBid;
	}

	public Bid getOpponentBestBid() {
		return OpponentBestBid;
	}

	public void setOpponentBestBid(Bid opponentBestBid) {
		OpponentBestBid = opponentBestBid;
	}

	public int getTotalBiddingPossibilities() {
		return TotalBiddingPossibilities;
	}

	public void setTotalBiddingPossibilities(int totalBiddingPossibilities) {
		TotalBiddingPossibilities = totalBiddingPossibilities;
	}

	public void setBids(TreeMap<Double, Bid> bids) {
		this.Bids = bids;
	}

	public Double getBidsCreationTime() {
		return BidsCreationTime;
	}

	public void setBidsCreationTime(Double bidsCreationTime) {
		BidsCreationTime = bidsCreationTime;
	}

}
