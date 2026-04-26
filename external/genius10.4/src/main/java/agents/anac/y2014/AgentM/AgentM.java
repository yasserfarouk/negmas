package agents.anac.y2014.AgentM;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

import agents.SimpleAgentSavingBidHistory;
import genius.core.Agent;
import genius.core.Bid;
import genius.core.BidHistory;
import genius.core.NegotiationResult;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.ActionWithBid;
import genius.core.actions.EndNegotiation;
import genius.core.actions.Offer;
import genius.core.bidding.BidDetails;
import genius.core.issue.Issue;
import genius.core.issue.IssueInteger;
import genius.core.issue.Value;
import genius.core.issue.ValueInteger;

/**
 * @author S. Hourmann Some improvements over the standard Agent. Saving Bid
 *         History for the session.
 * 
 *         Random Walker, Zero Intelligence Agent
 */
public class AgentM extends Agent {
	// "state" represents "where am I in the code".
	// I want to print "state" when I print a message about saving data.
	private String state;
	private Action actionOfPartner = null;

	/**
	 * Note: {@link SimpleAgentSavingBidHistory} does not account for the
	 * discount factor in its computations
	 */
	private static double MINIMUM_BID_UTILITY = 0.0;
	private static int NUMBER_ITERATIONS = 1000;
	private BidHistory currSessOppBidHistory;
	private BidHistory prevSessOppBidHistory;
	private BidHistory mySessBidHistory;
	private Bid bidmax = null;
	private Bid endbid = null;
	private double concessionRate = 0.0;
	private ArrayList issueAllay = new ArrayList();
	private ArrayList myissue = new ArrayList();
	private ArrayList<Integer> movement = new ArrayList<Integer>();
	private ArrayList<Integer> maxissue = new ArrayList<Integer>();

	/**
	 * init is called when a next session starts with the same opponent.
	 */
	@Override
	public void init() {
		MINIMUM_BID_UTILITY = utilitySpace.getReservationValueUndiscounted();
		myBeginSession();
		currSessOppBidHistory = new BidHistory();
		prevSessOppBidHistory = new BidHistory();
		mySessBidHistory = new BidHistory();

		int[] numAllay = new int[10];
		for (int j = 0; j < 10; j++) {
			numAllay[j] = 0;
		}
		Integer num = 0;
		for (int i = 0; i < this.utilitySpace.getDomain().getIssues()
				.size(); i++) {
			issueAllay.add(numAllay.clone());
			movement.add(num);
			maxissue.add(num);
		}
	}

	public void myBeginSession() {
		System.out.println("Starting match num: " + sessionNr);

		// ---- Code for trying save and load functionality
		// First try to load saved data
		// ---- Loading from agent's function "loadSessionData"
		Serializable prev = this.loadSessionData();
		AgentMData a = (AgentMData) prev;
		if (a != null) {
			this.endbid = a.getBid();
			System.out.println("load complete");
			// System.out
			// .println("---------/////////// NEW NEW NEW
			// /////////////----------");
			// System.out.println("The size of the previous BidHistory is: "
			// + prevSessOppBidHistory.size());
		} else {
			// If didn't succeed, it means there is no data for this preference
			// profile
			// in this domain.
			System.out.println("There is no history yet.");
		}
	}

	@Override
	public String getVersion() {
		return "3.1";
	}

	@Override
	public String getName() {
		return "AgentM";
	}

	@Override
	public void ReceiveMessage(Action opponentAction) {
		actionOfPartner = opponentAction;
		if (opponentAction instanceof Offer) {
			Bid bid = ((Offer) opponentAction).getBid();
			// 2. store the opponent's trace
			try {
				BidDetails opponentBid = new BidDetails(bid,
						utilitySpace.getUtility(bid), timeline.getTime());
				if (currSessOppBidHistory.getLastBidDetails() != null) {
					this.prevSessOppBidHistory.add(
							this.currSessOppBidHistory.getLastBidDetails());
				}
				currSessOppBidHistory.add(opponentBid);
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		try {
			if (this.prevSessOppBidHistory.getLastBid() != null) {
				if (getUtility(this.prevSessOppBidHistory.getBestBidDetails()
						.getBid()) < getUtility(
								this.currSessOppBidHistory.getLastBid())) {
					this.concessionRate = Math.pow(
							getUtility(this.prevSessOppBidHistory
									.getWorstBidDetails().getBid())
									- getUtility(this.currSessOppBidHistory
											.getBestBidDetails().getBid()),
							2.0);
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	@Override
	public Action chooseAction() {
		Action action = null;
		try {
			if (actionOfPartner == null)
				action = chooseRandomBidAction(null); // original code but will
														// throw NPE
			if (actionOfPartner instanceof Offer) {
				Bid partnerBid = ((Offer) actionOfPartner).getBid();
				double offeredUtilFromOpponent = getUtility(partnerBid);
				// get current time
				double time = timeline.getTime();
				action = chooseRandomBidAction(partnerBid);

				Bid myBid = ((Offer) action).getBid();
				double myOfferedUtil = getUtility(myBid);

				// accept under certain circumstances
				if (isAcceptable(offeredUtilFromOpponent, myOfferedUtil,
						time)) {
					action = new Accept(getAgentID(), partnerBid);
					// ---- Code for trying save and load functionality
					// /////////////////////////////////
					state = "I accepted so I'm trying to save. ";
					tryToSaveAndPrintState();
					// /////////////////////////////////
				}
			}
			if (actionOfPartner instanceof EndNegotiation) {
				action = new Accept(getAgentID(),
						((ActionWithBid) actionOfPartner).getBid());
				// ---- Code for trying save and load functionality
				// /////////////////////////////////
				state = "Got EndNegotiation from opponent. ";
				tryToSaveAndPrintState();
				// /////////////////////////////////
			}
			// sleep(0.001); // just for fun
		} catch (Exception e) {
			System.out.println("Exception in ChooseAction:" + e.getMessage());

			// ---- Code for trying save and load functionality
			// /////////////////////////////////
			state = "Got Exception. ";
			tryToSaveAndPrintState();
			// /////////////////////////////////
			// best guess if things go wrong.
			action = new Accept(getAgentID(),
					((ActionWithBid) actionOfPartner).getBid());
		}
		return action;
	}

	// ---- Code for trying save and load functionality
	private void tryToSaveAndPrintState() {

		// ---- Saving from agent's function "saveSessionData"

		// Bid lastBid = this.currSessOppBidHistory.getLastBid();
		// System.out.println("testtettttttt");
		// AgentMData data = new AgentMData(lastBid);
		// System.out.println(data.getBid());
		// this.saveSessionData(data);
		// System.out.println(state +
		// "The size of the BidHistory I'm saving is: "
		// + currSessOppBidHistory.size());
	}

	private boolean isAcceptable(double offeredUtilFromOpponent,
			double myOfferedUtil, double time) throws Exception {
		// double P = Paccept(offeredUtilFromOpponent, time);
		// offeredUtilFromOpponent ‘ŠŽè‚Ì’ñˆÄ myOfferedUtil
		// Ž©•ª‚Ì’ñˆÄ
		// double offerOk =
		// (getUtility(mySessBidHistory.getWorstBidDetails().getBid()) +
		// getUtility(mySessBidHistory.getBestBidDetails().getBid()))/2;
		double offerOk = getUtility(
				mySessBidHistory.getWorstBidDetails().getBid());
		List<Issue> issues = utilitySpace.getDomain().getIssues();

		// ‘ŠŽè‚Ìlastbid‚©‚çissueAllay‚É‚»‚Ì’l‚ð’Ç‰Á
		int opfirstoplast = 0;
		for (int i = 1; i < issues.size() + 1; i++) {
			ValueInteger opLast = (ValueInteger) this.currSessOppBidHistory
					.getLastBid().getValue(i);
			int oplast = Integer.valueOf(opLast.toString());
			int[] tempAllay = (int[]) issueAllay.get(i - 1);
			tempAllay[oplast] += 1;
			this.issueAllay.set(i - 1, tempAllay.clone());
		}

		// System.out.println("check");

		// ’ñˆÄ‚³‚ê‚½bid‚©‚ç‚»‚Ìissue‚Ì’†‚Å�Å‚à‰ñ�”‚Ì‘½‚¢‚à‚Ì‚ðmaxissue‚É�Ý’è
		for (int i = 0; i < issues.size(); i++) {
			// issue‚ÌŽæ‚é’l‚ðŽæ‚è�o‚·
			IssueInteger lIssueInteger = (IssueInteger) issues.get(i);
			int issueNumber = lIssueInteger.getNumber();
			int issueIndexMin = lIssueInteger.getLowerBound();
			int issueIndexMax = lIssueInteger.getUpperBound();
			// issueAllay‚Å‚Íissue‚Ìbid‚³‚ê‚½‰ñ�”‚ð•Û‘¶
			int[] temp = (int[]) issueAllay.get(i);
			// num‚ÍissueIndexMin,issueIndexMax‚ÌŠÔ‚Ì’l
			int num = 0;
			// maxissuenum‚ÍŒÄ‚Ñ�o‚³‚ê‚½‰ñ�”‚Ì’l‚Ìmax
			int maxissuenum = 0;
			for (int j = issueIndexMin; j < issueIndexMax; j++) {
				if (maxissuenum < temp[j]) {
					// System.out.println(j);
					num = j;
					maxissuenum = temp[j];
				}
			}
			Integer a = num;
			this.maxissue.set(i, a);
		}

		// ‘ŠŽè—\‘ª’l:w
		if (endbid != null) {
			// System.out.println(getUtility(endbid));
			if (offeredUtilFromOpponent > offerOk
					|| offeredUtilFromOpponent >= getUtility(endbid)) {
				return true;
			}
		}
		if (offeredUtilFromOpponent > offerOk) {
			return true;
		}
		return false;
	}

	/**
	 * Wrapper for getRandomBid, for convenience.
	 * 
	 * @return new Action(Bid(..)), with bid utility > MINIMUM_BID_UTIL. If a
	 *         problem occurs, it returns an Accept() action.
	 */
	private Action chooseRandomBidAction(Bid opponentBid) {
		Bid nextBid = null;
		try {
			nextBid = getRandomBid();
		} catch (Exception e) {
			System.out.println("Problem with received bid:" + e.getMessage()
					+ ". cancelling bidding");
		}
		if (nextBid == null)
			return (new Accept(getAgentID(), opponentBid));
		return (new Offer(getAgentID(), nextBid));
	}

	/**
	 * @param value
	 * @return a random bid with high enough utility value.
	 * @throws Exception
	 *             if we can't compute the utility (eg no evaluators have been
	 *             set) or when other evaluators than a DiscreteEvaluator are
	 *             present in the util space.
	 */
	@Override
	public void endSession(NegotiationResult result) {
		// System.out.println(result);
		// System.out.println(result.isAgreement());
		if (result.isAgreement()) {
			Bid lastBid = result.getLastBid();
			AgentMData data = new AgentMData(lastBid);
			// System.out.println(data.getBid());
			this.saveSessionData(data);
		}
	}

	private Bid getRandomBid() throws Exception {
		HashMap<Integer, Value> values = new HashMap<Integer, Value>(); // pairs
		// <issuenumber,chosen
		// value
		// string>
		List<Issue> issues = utilitySpace.getDomain().getIssues();
		Random randomnr = new Random();

		// create a random bid with utility>MINIMUM_BID_UTIL.
		// note that this may never succeed if you set MINIMUM too high!!!
		// in that case we will search for a bid till the time is up (3 minutes)
		// but this is just a simple agent.
		Bid bid = null;
		if (this.bidmax == null) {
			int i = 0;

			double currentUtility, maxUtility = 0;
			do {
				bid = utilitySpace.getDomain().getRandomBid(null);
				currentUtility = utilitySpace.getUtility(bid);
				if (bidmax == null || getUtility(bidmax) < currentUtility) {
					bidmax = bid;
					maxUtility = getUtility(bidmax);
					if (maxUtility > 0.98) {
						break;
					}
				}
				i++;
			} while (i < NUMBER_ITERATIONS);
			i = 0;
			while ((i++ < NUMBER_ITERATIONS) && (maxUtility < 0.999)) {
				Bid nextBid = nextBid(bid); // ‹ß–TBid‚ð’T�õ
				double nextUtility = utilitySpace.getUtility(nextBid); // ‹ß–TBid‚ÌŒø—p’l
				double temperature = calculateTemperature(i); // ‰·“x‚ÌŽZ�o
				double probability = calculateProbability(currentUtility,
						nextUtility, temperature); // ‘JˆÚŠm—¦‚ÌŽZ�o
				if (probability > randomnr.nextDouble()) {
					bid = nextBid; // Bid‚Ì�X�V
					currentUtility = utilitySpace.getUtility(nextBid); // Œø—p’l‚Ì�X�V
					// �Å“K‰ð‚Ì�X�V
					if (nextUtility > maxUtility) {
						this.bidmax = nextBid;
						maxUtility = nextUtility;
					}
				}
			}
		} else {
			int i = 0;

			double currentUtility, maxUtility = 0;
			do {
				bid = utilitySpace.getDomain().getRandomBid(null);
				currentUtility = utilitySpace.getUtility(bid);
				if (bidmax == null || getUtility(bidmax) < currentUtility) {
					bidmax = bid;
					maxUtility = getUtility(bidmax);
					if (maxUtility > 0.98) {
						break;
					}
				}
				i++;
			} while (i < NUMBER_ITERATIONS);
			i = 0;
			int j = 0;

			values = this.mySessBidHistory.getBestBidDetails().getBid()
					.getValues();

			double endpoint;
			this.concessionRate = Math
					.pow(getUtility(this.currSessOppBidHistory
							.getWorstBidDetails().getBid())
							- getUtility(this.currSessOppBidHistory
									.getBestBidDetails().getBid()),
							2.0)
					+ timeline.getTime() / 10;
			if (this.endbid == null) {
				endpoint = 0.999 - this.concessionRate;
			} else {
				endpoint = getUtility(this.endbid);
			}
			while (((i++ < NUMBER_ITERATIONS)
					&& (maxUtility < 0.999 - this.concessionRate))
					|| ((i < NUMBER_ITERATIONS) && maxUtility < endpoint)) {
				Bid nextBid = nextBid(bid); // ‹ß–TBid‚ð’T�õ
				// for(j = 0; j < (timeline.getTime()*issues.size()/5); j++){
				for (j = 0; j < (issues.size() / 5); j++) {
					int issueIndex = randomnr.nextInt(issues.size());
					IssueInteger lIssueInteger = (IssueInteger) issues
							.get(issueIndex);
					int issueNumber = lIssueInteger.getNumber();
					int issueIndexMin = lIssueInteger.getLowerBound();
					int issueIndexMax = lIssueInteger.getUpperBound();
					int optionIndex = 0;
					ValueInteger lIssueValue = (ValueInteger) bid
							.getValue(issueNumber);
					int issueValue = Integer.valueOf(lIssueValue.toString())
							.intValue();
					optionIndex = nextOptionIndex(issueIndexMin, issueIndexMax,
							issueValue);
					lIssueInteger = (IssueInteger) issues.get(issueIndex);
					issueNumber = lIssueInteger.getNumber();
					int test_value = 0;
					if (j % 2 == 0) {
						test_value = this.maxissue.get(issueNumber - 1);
					} else {
						// ValueInteger aaa = (ValueInteger)
						// this.currSessOppBidHistory.getBestBid().getValue(issueNumber);
						ValueInteger aaa = (ValueInteger) this.currSessOppBidHistory
								.getBestBidDetails().getBid()
								.getValue(issueNumber);
						test_value = Integer.valueOf(aaa.toString());
					}
					nextBid = nextBid.putValue(issueNumber,
							new ValueInteger(test_value));
					issueValue = Integer.valueOf(lIssueValue.toString())
							.intValue();
					issueIndex = randomnr.nextInt(issues.size());
					lIssueInteger = (IssueInteger) issues.get(issueIndex);
					issueNumber = lIssueInteger.getNumber();
					ValueInteger bbb = (ValueInteger) this.currSessOppBidHistory
							.getBestBidDetails().getBid().getValue(issueNumber);
					test_value = Integer.valueOf(bbb.toString());
					nextBid = nextBid.putValue(issueNumber,
							new ValueInteger(test_value));
				}
				double nextUtility = utilitySpace.getUtility(nextBid); // ‹ß–TBid‚ÌŒø—p’l
				double temperature = calculateTemperature(i); // ‰·“x‚ÌŽZ�o
				double probability = calculateProbability(currentUtility,
						nextUtility, temperature); // ‘JˆÚŠm—¦‚ÌŽZ�o
				if (probability > randomnr.nextDouble()) {
					bid = nextBid; // Bid‚Ì�X�V
					currentUtility = utilitySpace.getUtility(nextBid); // Œø—p’l‚Ì�X�V
					// �Å“K‰ð‚Ì�X�V
					// if (nextUtility > maxUtility) {
					if (nextUtility > (getUtility(this.bidmax)
							- this.concessionRate)) {
						this.bidmax = nextBid;
						maxUtility = nextUtility;
						// System.out.println(getUtility(this.bidmax)-this.concessionRate);
					}
				}
			}

		}
		BidDetails opponentBid = new BidDetails(bidmax,
				utilitySpace.getUtility(bidmax), timeline.getTime());
		mySessBidHistory.add(opponentBid);
		return this.bidmax;
	}

	private int randomOptionIndex(int issueIndexMin, int issueIndexMax) {
		int optionIndex = 0;
		Random randomnr = new Random();
		if (issueIndexMin < issueIndexMax) {
			optionIndex = issueIndexMin
					+ randomnr.nextInt(issueIndexMax - issueIndexMin);
		} else {
			optionIndex = issueIndexMin; // issueIndexMin ==
											// issueIndexMax‚Ì�ê�‡
		}

		return optionIndex;
	}

	/**
	 * ‹ß–T’T�õ
	 * 
	 * @param issueIndexMin
	 *            ‰ºŒÀ’l
	 * @param issueIndexMax
	 *            �ãŒÀ’l
	 * @param issueValue
	 *            Œ»�Ý‚Ì’l
	 * @return ’l‚ð1‚¾‚¯‘�Œ¸‚³‚¹‚é
	 */
	private int nextOptionIndex(int issueIndexMin, int issueIndexMax,
			int issueValue) {
		int step = 1; // ’l‚ð‘�Œ¸‚³‚¹‚é•�
		Random randomnr = new Random();
		int direction = randomnr.nextBoolean() ? 1 : -1; // ƒ‰ƒ“ƒ_ƒ€‚É‘�Œ¸‚ðŒˆ’è

		if (issueIndexMin < issueIndexMax) {
			if ((issueValue + step) > issueIndexMax) { // +1‚·‚é‚Æ�ãŒÀ’l‚ð’´‚¦‚é�ê�‡
				direction = -1;
			} else if ((issueValue - step) < issueIndexMin) { // -1‚·‚é‚Æ‰ºŒÀ’l‚ð‰º‰ñ‚é�ê�‡
				direction = 1;
			}
		} else {
			return issueValue; // issueIndexMin == issueIndexMax‚Ì�ê�‡
		}

		return issueValue + step * direction;
	}

	/**
	 * This function determines the accept probability for an offer. At t=0 it
	 * will prefer high-utility offers. As t gets closer to 1, it will accept
	 * lower utility offers with increasing probability. it will never accept
	 * offers with utility 0.
	 * 
	 * @param u
	 *            is the utility
	 * @param t
	 *            is the time as fraction of the total available time (t=0 at
	 *            start, and t=1 at end time)
	 * @return the probability of an accept at time t
	 * @throws Exception
	 *             if you use wrong values for u or t.
	 * 
	 */
	double Paccept(double u, double t1) throws Exception {
		double t = t1 * t1 * t1; // steeper increase when deadline approaches.
		if (u < 0 || u > 1.05)
			throw new Exception("utility " + u + " outside [0,1]");
		// normalization may be slightly off, therefore we have a broad boundary
		// up to 1.05
		if (t < 0 || t > 1)
			throw new Exception("time " + t + " outside [0,1]");
		if (u > 1.)
			u = 1;
		if (t == 0.5)
			return u;
		return (u - 2. * u * t
				+ 2. * (-1. + t + Math.sqrt(sq(-1. + t) + u * (-1. + 2 * t))))
				/ (-1. + 2 * t);
	}

	double sq(double x) {
		return x * x;
	}

	private double calculateProbability(double currentUtil, double nextUtil,
			double temperature) {
		double diff = currentUtil - nextUtil;
		if (diff > 0.0) {
			return Math.exp(-diff / temperature); // Œ»�Ý‚ÌŒø—p’l‚Ì•û‚ª�‚‚¢�ê�‡
		} else {
			return 1.0; // ‹ß–T‚ÌŒø—p’l‚Ì•û‚ª�‚‚¢�ê�‡
		}
	}

	private double calculateTemperature(int iteration) {
		double t = 0.01;
		return t * Math.pow(1.0 - ((double) iteration / NUMBER_ITERATIONS), 2);
	}

	private Bid nextBid(Bid bid) throws Exception {
		List<Issue> issues = utilitySpace.getDomain().getIssues(); // ‘Sissue‚ÌŽæ“¾
		Bid nextBid = new Bid(bid); // Œ»�Ý‚ÌBid‚ðƒRƒs�[
		Boolean optionFlag = false; // ’T�õŽè–@
		int numberIndexes = utilitySpace.getDomain().getIssues().size() / 10;
		Random randomnr = new Random();
		for (int i = 0; i < numberIndexes; i++) {
			int issueIndex = randomnr.nextInt(issues.size()); // Issue‚ðƒ‰ƒ“ƒ_ƒ€‚ÉŽw’è
			IssueInteger lIssueInteger = (IssueInteger) issues.get(issueIndex); // Žw’è‚µ‚½index‚Ìissue‚ðŽæ“¾
			int issueNumber = lIssueInteger.getNumber(); // issue”Ô�†
			int issueIndexMin = lIssueInteger.getLowerBound(); // issue‚Ì‰ºŒÀ’l
			int issueIndexMax = lIssueInteger.getUpperBound(); // issue‚Ì�ãŒÀ’l
			int optionIndex = 0; // •Ï�X‚·‚éValue’l

			// ‹ß–T’T�õ
			if (optionFlag) {
				optionIndex = randomOptionIndex(issueIndexMin, issueIndexMax); // ƒ‰ƒ“ƒ_ƒ€‚É’l‚ð•Ï‚¦‚é
			} else {
				ValueInteger lIssueValue = (ValueInteger) bid
						.getValue(issueNumber); // Žw’è‚µ‚½issue‚ÌValue
				int issueValue = Integer.valueOf(lIssueValue.toString())
						.intValue();
				optionIndex = nextOptionIndex(issueIndexMin, issueIndexMax,
						issueValue); // ’l‚ð1‘�Œ¸‚³‚¹‚é
			}

			nextBid = nextBid.putValue(issueNumber,
					new ValueInteger(optionIndex)); // Œ»�Ý‚ÌBid‚©‚çIssue‚Ì’l‚ð“ü‚ê‘Ö‚¦‚é
		}
		return nextBid;
	}

	@Override
	public String getDescription() {
		return "ANAC2014 compatible with non-linear utility spaces";
	}
}

class AgentMData implements Serializable {
	Bid lastBid;
	Boolean isAgreement;

	public AgentMData(Bid last) {
		this.lastBid = last;
	}

	public Bid getBid() {
		return lastBid;
	}
}