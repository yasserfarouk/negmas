package agents.anac.y2017.rubick;
/*
 * OKAN TUNALI - 
 */

import java.util.List;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedList;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.Offer;
import genius.core.bidding.BidDetails;
import genius.core.boaframework.SortedOutcomeSpace;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.Value;
import genius.core.list.Tuple;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;
import genius.core.persistent.StandardInfo;
import genius.core.persistent.StandardInfoList;

public class Rubick extends AbstractNegotiationParty {

	private Bid lastReceivedBid = null;
	private StandardInfoList history;
	private LinkedList<String> parties = new LinkedList<>();
	private LinkedList<LinkedList<Double>> histOpp0 = new LinkedList<>();
	private LinkedList<LinkedList<Double>> histOpp1 = new LinkedList<>();
	private boolean isHistoryAnalyzed = false;
	private int numberOfReceivedOffer = 0;
	private LinkedList<Integer> profileOrder = new LinkedList<>();
	private String[] opponentNames = new String[2];
	private double[] acceptanceLimits = { 0.0, 0.0 };
	private double maxReceivedBidutil = 0.0;
	private String lastpartyname = "";
	private SortedOutcomeSpace sos = null;
	private LinkedList<Bid> bestAcceptedBids = new LinkedList<>();
	private double threshold = 0;

	protected LinkedList<LinkedHashMap<Value, Integer>> frequentValuesList0 = new LinkedList<>();
	protected LinkedList<LinkedHashMap<Value, Integer>> frequentValuesList1 = new LinkedList<>();

	ArrayList<Value> opp0bag = new ArrayList<Value>();
	ArrayList<Value> opp1bag = new ArrayList<Value>();

	@Override
	public void init(NegotiationInfo info) {

		super.init(info);

		String domainName = utilitySpace.getFileName();
		domainName = domainName.substring(domainName.lastIndexOf("/") + 1,
				domainName.lastIndexOf("."));

		sos = new SortedOutcomeSpace(utilitySpace);
		history = (StandardInfoList) getData().get();

		sortPartyProfiles(getPartyId().toString());
		threshold = getUtilitySpace().getReservationValue();
		maxReceivedBidutil = threshold; // to see the logic
		initializeOpponentModelling();
	}

	@SuppressWarnings("unchecked")
	public void initializeOpponentModelling() {

		int issueSize = utilitySpace.getDomain().getIssues().size();

		for (int i = 0; i < issueSize; i++) {
			LinkedHashMap<Value, Integer> valueAmountMap = new LinkedHashMap<>();

			frequentValuesList0.add(valueAmountMap); // first add an empty map.
			valueAmountMap = new LinkedHashMap<>();
			frequentValuesList1.add(valueAmountMap);

			int issuecnt = 0;
			for (Issue issue : utilitySpace.getDomain().getIssues()) {
				IssueDiscrete issued = (IssueDiscrete) issue;

				if (issuecnt == i) {
					for (Value value : issued.getValues()) {
						frequentValuesList0.get(i).put(value, 0);
						frequentValuesList1.get(i).put(value, 0);
					}
				}
				issuecnt++;
			}
		}
	}

	@Override
	public Action chooseAction(List<Class<? extends Action>> validActions) {

		double decisiveUtil = checkAcceptance();

		if (decisiveUtil == -1) {
			return new Accept(getPartyId(), lastReceivedBid);
		} else {
			Bid bid = generateBid(decisiveUtil);
			return new Offer(getPartyId(), bid);
		}
	}

	@Override
	public void receiveMessage(AgentID sender, Action action) {
		super.receiveMessage(sender, action);
		if (action instanceof Accept) {

			Bid acceptedBid = ((Accept) action).getBid();

			if (bestAcceptedBids.isEmpty()) {
				bestAcceptedBids.add(acceptedBid);
			} else if (!bestAcceptedBids.contains(acceptedBid)) {
				int size = bestAcceptedBids.size();
				for (int i = 0; i < size; i++) {
					if (getUtility(acceptedBid) > getUtility(
							bestAcceptedBids.get(i))) {
						bestAcceptedBids.add(i, acceptedBid); // collect best
																// accepted
																// Bids.
						break;
					} else if (i == bestAcceptedBids.size() - 1) { // if new
																	// accepted
																	// bid has
																	// the least
																	// util in
																	// the list.
						bestAcceptedBids.add(acceptedBid);
					}
				}
			}
		}

		if (action instanceof Offer) {
			lastReceivedBid = ((Offer) action).getBid();

			if (maxReceivedBidutil < getUtilityWithDiscount(lastReceivedBid))
				maxReceivedBidutil = getUtilityWithDiscount(lastReceivedBid)
						* 0.95;

			numberOfReceivedOffer++;

			String partyName = getPartyName(action.getAgent().toString());
			lastpartyname = partyName;

			BidResolver(lastReceivedBid, partyName);

			if (!parties.contains(partyName)) {
				sortPartyProfiles(action.getAgent().toString());
			}

			if (parties.size() == 3 && !history.isEmpty()
					&& isHistoryAnalyzed == false) {
				// System.out.println("about to analyze: ");
				analyzeHistory();
			}
		}
	}

	private int takeTheChance(double maxReceived) {

		int pow = 1;
		double chance = rand.nextDouble();
		// System.out.println("chance: " + chance);
		if (chance > 0.95 + 0.05 * maxReceived)
			pow = 2;
		else if (chance > 0.93 + 0.07 * maxReceived)
			pow = 3;
		else
			pow = 10;

		return pow;
	}

	private double checkAcceptance() {

		int pow = takeTheChance(maxReceivedBidutil);
		double targetutil = 1 - (Math.pow(timeline.getTime(), pow)
				* Math.abs(rand.nextGaussian() / 3));

		if (numberOfReceivedOffer < 2)
			return 1;
		else if (!history.isEmpty()) {
			double upperLimit = maxReceivedBidutil;

			// pick the highest as the upper limit
			for (double du : acceptanceLimits) {
				if (upperLimit < du)
					upperLimit = du;
			}
			upperLimit = 0.90 * upperLimit;
			pow = takeTheChance(upperLimit);
			targetutil = 1 - (Math.pow(timeline.getTime(), pow)
					* Math.abs(rand.nextGaussian() / 3));
			targetutil = upperLimit + (1 - upperLimit) * targetutil;

		} else {

			if (maxReceivedBidutil < 0.8)
				maxReceivedBidutil = 0.8;

			targetutil = maxReceivedBidutil
					+ (1 - maxReceivedBidutil) * targetutil;

		}

		if (getUtilityWithDiscount(lastReceivedBid) > targetutil
				|| timeline.getTime() > 0.999)
			return -1; // Accept
		else
			return targetutil;
	}

	private Bid generateBid(double targetutil) {
		Bid bid = null;

		if (timeline.getTime() > 0.995 && !bestAcceptedBids.isEmpty()) {
			// game has almost ended. Offer one from best accepted bids
			int s = bestAcceptedBids.size();

			if (s > 3)
				s = 3;

			// pick from top 3
			int ind = rand.nextInt(s);
			bid = bestAcceptedBids.get(ind);
		} else {

			// find candidate bids in range target utility and 1
			if (opp0bag.size() > 0 && opp1bag.size() > 0)
				bid = searchCandidateBids(targetutil);

			if (bid == null)
				bid = sos.getBidNearUtility(targetutil).getBid();
		}

		System.out.flush();
		return bid;

	}

	public Bid searchCandidateBids(double targetutil) {
		double bu = 0.0;
		Value valu = null;
		// search for maximum match
		LinkedList<Integer> intersection = new LinkedList<>();
		LinkedList<Bid> candidateBids = new LinkedList<>();

		for (BidDetails bd : sos.getAllOutcomes()) {
			bu = getUtility(bd.getBid());

			if (bu >= targetutil) {
				int score = 0;
				for (int isn = 0; isn < bd.getBid().getIssues().size(); isn++) {
					valu = bd.getBid().getValue(isn + 1);

					if (valu == opp0bag.get(isn))
						score++;

					if (valu == opp1bag.get(isn))
						score++;
				}

				intersection.add(score);
				candidateBids.add(bd.getBid());

			} else
				break;
		}

		int max = -1;
		for (int i = 0; i < intersection.size(); i++) {
			if (max < intersection.get(i))
				max = i; // if find find higher score, make it max.
		}

		if (candidateBids.size() > 1) {
			return candidateBids.get(max);
		}

		return null;
	}

	public void BidResolver(Bid bid, String partyname) {
		Value valu = null;
		if (partyname.equals(opponentNames[0])) {

			for (int isn = 0; isn < bid.getIssues().size(); isn++) {
				valu = bid.getValue(isn + 1);
				int prevAmount = frequentValuesList0.get(isn).get(valu);

				frequentValuesList0.get(isn).put(valu, prevAmount + 1);
			}

		} else if (partyname.equals(opponentNames[1])) {
			for (int isn = 0; isn < bid.getIssues().size(); isn++) {
				valu = bid.getValue(isn + 1);
				int prevAmount = frequentValuesList1.get(isn).get(valu);
				frequentValuesList1.get(isn).put(valu, prevAmount + 1);
			}
		}

		if (numberOfReceivedOffer > 2)
			extractOpponentPreferences();

	}

	public void printFreqs(int opid) {

		System.out.println("opid : " + opid);
		for (int i = 0; i < frequentValuesList0.size(); i++) {

			if (opid == 0) {
				for (Value val : frequentValuesList0.get(i).keySet()) {
					System.out.println("freq0: is: " + (i + 1) + " value :"
							+ val + " amount: "
							+ frequentValuesList0.get(i).get(val));
				}
			} else {
				for (Value val : frequentValuesList1.get(i).keySet()) {
					System.out.println("freq1: is: " + (i + 1) + " value :"
							+ val + " amount: "
							+ frequentValuesList1.get(i).get(val));
				}
			}
			System.out.println("\n");
		}
		System.out.println("\n");

	}

	public void extractOpponentPreferences() {
		// find the best intersection
		ArrayList<Value> opp0priors = new ArrayList<Value>();
		ArrayList<Value> opp1priors = new ArrayList<Value>();

		opp0bag = new ArrayList<Value>();
		opp1bag = new ArrayList<Value>();

		LinkedList<Double> meanEvalValues0 = new LinkedList<>();
		LinkedList<Double> meanEvalValues1 = new LinkedList<>();

		for (int i = 0; i < frequentValuesList0.size(); i++) {
			double sum = 0.0;
			for (Value val : frequentValuesList0.get(i).keySet()) {
				sum += frequentValuesList0.get(i).get(val); // find the average
															// eval value of
															// that issue
			}
			meanEvalValues0.add(sum / frequentValuesList0.size());

			sum = 0.0;
			for (Value val : frequentValuesList1.get(i).keySet()) {
				sum += frequentValuesList1.get(i).get(val); // find the average
															// eval value of
															// that issue
			}
			meanEvalValues1.add(sum / frequentValuesList1.size());
		}

		// select ones with over average
		for (int i = 0; i < frequentValuesList0.size(); i++) {
			for (Value val : frequentValuesList0.get(i).keySet()) {
				if (frequentValuesList0.get(i).get(val) >= meanEvalValues0
						.get(i)) {
					opp0priors.add(val);
				}
			}
			opp0bag.add(opp0priors.get(rand.nextInt(opp0priors.size())));
			opp0priors = new ArrayList<Value>();

			for (Value val : frequentValuesList1.get(i).keySet()) {
				if (frequentValuesList1.get(i).get(val) >= meanEvalValues1
						.get(i)) {
					opp1priors.add(val);
				}
			}
			opp1bag.add(opp1priors.get(rand.nextInt(opp1priors.size())));
			opp1priors = new ArrayList<Value>();

		}

	}

	private Integer extractPartyID(String partyID) {
		return Integer.parseInt(
				partyID.substring(partyID.indexOf("@") + 1, partyID.length()));
	}

	private void analyzeHistory() {
		isHistoryAnalyzed = true;

		for (int h = 0; h <= history.size() - 1; h++) { // from older to recent
														// history

			LinkedList<Double> utilsOp1 = new LinkedList<>();
			LinkedList<Double> utilsOp2 = new LinkedList<>();

			StandardInfo info = history.get(h);

			boolean historyMatch = true;

			int cnt = 0;
			for (Tuple<String, Double> offered : info.getUtilities()) {

				String partyname = getPartyName(offered.get1());
				Double util = offered.get2();

				if (cnt < 3 && !partyname.equals(parties.get(cnt))) {
					historyMatch = false;
					break;
				} else {
					// check if there's a confusion

					if (partyname.equals(opponentNames[0])) {
						utilsOp1.add(util);
						if (util > acceptanceLimits[0])
							acceptanceLimits[0] = util;

					} else if (partyname.equals(opponentNames[1])) {
						utilsOp2.add(util);
						if (util > acceptanceLimits[1])
							acceptanceLimits[1] = util;
					}

				}
				cnt++;
			}

		}

	}

	private void sortPartyProfiles(String partyID) {

		// System.out.println("\ninSorting ID: " + partyID);
		int pid = extractPartyID(partyID);
		String partyName = getPartyName(partyID);

		if (profileOrder.isEmpty()) {

			profileOrder.add(pid);
			parties.add(partyName);

		} else {
			int size = profileOrder.size();
			for (int id = 0; id < size; id++) {

				if (pid < profileOrder.get(id)) { // Find smaller put instead of
													// it ~ before it
					profileOrder.add(id, pid);
					parties.add(id, partyName);
					break;
				} else if (id == profileOrder.size() - 1) {
					profileOrder.add(pid);
					parties.add(partyName);
				}

			}
		}

		int p = 0;
		for (String party : parties) {

			// if it's not my name
			if (!party.equals(getPartyName(getPartyId().toString()))) {
				{
					opponentNames[p] = party;
					p++;
				}

			}
		}

	}

	private String getPartyName(String partyID) {
		return partyID.substring(0, partyID.indexOf("@"));
	}

	@Override
	public String getDescription() {
		return "ANAC2017";
	}

}
