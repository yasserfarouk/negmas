package agents.anac.y2014.E2Agent;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import agents.anac.y2014.E2Agent.myUtility.AgentKStorategy;
import agents.anac.y2014.E2Agent.myUtility.BidStorage;
import agents.anac.y2014.E2Agent.myUtility.BidStorageComparator;
import agents.anac.y2014.E2Agent.myUtility.BidStorageList;
import agents.anac.y2014.E2Agent.myUtility.IAgentKStorategyComponent;
import agents.anac.y2014.E2Agent.myUtility.Parameters;
import agents.anac.y2014.E2Agent.myUtility.SessionData;
import agents.anac.y2014.E2Agent.myUtility.SimulatedAnealing;
import agents.anac.y2014.E2Agent.myUtility.SummaryStatistics;
import genius.core.Agent;
import genius.core.Bid;
import genius.core.NegotiationResult;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.ActionWithBid;
import genius.core.actions.Offer;
import genius.core.issue.Issue;
import genius.core.issue.IssueInteger;
import genius.core.issue.ValueInteger;
import genius.core.utility.AbstractUtilitySpace;
import genius.core.utility.Bound;

public class AnacSampleAgent extends Agent {
	private Action actionOfPartner = null;
	private static int SAMPLE_NUMBER = 3; // å�–å¾—ã�™ã‚‹ã‚µãƒ³ãƒ—ãƒ«æ•°
	private static int SA_K_MAX = 10000; // ç„¼ã��ã�ªã�¾ã�—æ³•ã�®ãƒ«ãƒ¼ãƒ—æ•°
	private AbstractUtilitySpace nonlinear = null; // é�žç·šå½¢åŠ¹ç”¨ç©ºé–“
	// private ArrayList<BidStorage> partnerBidHistory = null; //
	// ç›¸æ‰‹ã�®Bidå±¥æ­´
	private BidStorageList partnerBidHistory = null;
	private ArrayList<BidStorage> candidateBids = null; // Bidå€™è£œ
	private List<Issue> issues = null; // åŠ¹ç”¨ç©ºé–“ã�®å…¨ã�¦ã�®è«–ç‚¹
	private Random randomnr = null;
	private SimulatedAnealing simulatedAnealing = null;
	private double discountFactor = 0;
	private double reservationValue = 0;
	private AgentKStorategy agentKStorategy = null;
	private SessionData sessionData = null;
	private double time = 0;
	private double worstUtility = 1.0;
	private Parameters param;
	private BidStorage bestBid;

	@Override
	public void init() {
		Serializable s = loadSessionData();
		if (s != null && sessionNr != 0) {

			sessionData = (SessionData) s;
		} else {
			// System.out.println("######################################################################################################################################################");
			sessionData = new SessionData(sessionsTotal);
		}

		param = sessionData.getParamters(reservationValue, discountFactor);
		// System.out.println(param);

		randomnr = new Random();
		nonlinear = (AbstractUtilitySpace) utilitySpace.copy();
		issues = utilitySpace.getDomain().getIssues();
		discountFactor = utilitySpace.getDiscountFactor();
		reservationValue = utilitySpace.getReservationValue();
		agentKStorategy = new AgentKStorategy(randomnr,
				new AgentKStorategyComponentImpl());

		partnerBidHistory = new BidStorageList();
		bestBid = new BidStorage(null, 0, 0);

		try {
			simulatedAnealing = new SimulatedAnealing(utilitySpace);

			candidateBids = searchOptimumBids(SAMPLE_NUMBER);
			// åŠ¹ç”¨å€¤ã�§ã‚½ãƒ¼ãƒˆ
			Collections.sort(candidateBids, new BidStorageComparator());
			Collections.reverse(candidateBids); // é™�é †ã�«å¤‰æ›´

			// System.out.println(candidateBids);
		} catch (Exception e) {
			e.printStackTrace();
			candidateBids = null;
		}
	}

	/**
	 * æœ€é�©ã�ªBidã‚’æŽ¢ç´¢
	 */
	private ArrayList<BidStorage> searchOptimumBids(int sampleNr)
			throws Exception {
		ArrayList<BidStorage> bids = new ArrayList<BidStorage>();
		for (int index = 0; index < sampleNr; ++index) {
			Bid startBid = utilitySpace.getDomain().getRandomBid(null);
			// ç„¼ã��ã�ªã�¾ã�—æ³•ã�«ã‚ˆã‚‹æœ€é�©è§£æŽ¢ç´¢
			BidStorage optimumBid = simulatedAnealing.run(startBid, 1.0,
					SA_K_MAX);
			// Bidå€™è£œè¿½åŠ 
			bids.add(optimumBid);
		}
		return bids;
	}

	/**
	 * è‡ªåˆ†ã�®è¡Œå‹•é�¸æŠž
	 */
	@Override
	public Action chooseAction() {
		Action action = null;
		try {
			// ä¸€ç•ªå§‹ã‚�ã�®Bid
			if (actionOfPartner == null) {
				BidStorage myBidStorage = candidateBids.get(0);
				action = generateOffer(myBidStorage.getBid());
			}
			if (actionOfPartner instanceof Offer) { // ç›¸æ‰‹ã�ŒOfferã�—ã�¦ã��ã�Ÿå ´å�ˆ
				time = timeline.getTime();
				double tau = 1;

				// ç›¸æ‰‹ã�®Bidã‚’è¨˜æ†¶
				Bid partnerBid = ((Offer) actionOfPartner).getBid();
				BidStorage partnerBidStorage = new BidStorage(partnerBid,
						utilitySpace.getUtility(partnerBid), time);
				partnerBidHistory.addBidStorage(partnerBidStorage);
				SummaryStatistics stat = partnerBidHistory
						.getSummaryStatistics();
				// BidStorage bestBid = partnerBidHistory.getBestBidStorage();
				// System.out.println(stat);
				BidStorage myBidStorage = null;
				double targetUtility = 1.0;

				// è‡ªåˆ†ã�«ã�¨ã�£ã�¦æœ€ã‚‚è‰¯ã�„ç›¸æ‰‹ã�®Bid
				if (bestBid.getUtility() < partnerBidStorage.getUtility()) {
					bestBid = partnerBidStorage;
				}

				if (discountFactor > 0.9) {
					targetUtility = agentKStorategy.fintarget(time,
							stat.getAve(), stat.getVar(), tau);
				} else {
					targetUtility = fintarget2(time, stat, param);
				}

				myBidStorage = search2(partnerBid, targetUtility);
				if (myBidStorage == null) {
					// æŒ‡å®šã�—ã�ŸåŠ¹ç”¨å€¤ã‚’æŒ�ã�£ã�ŸBidã‚’æ¤œç´¢
					myBidStorage = searchBidWithApproximateUtlity(partnerBid,
							targetUtility);
					if (myBidStorage == null) {
						myBidStorage = candidateBids.get(0);
					}
				}

				double myUtility = myBidStorage.getUtility();

				if (// worstUtility <= partnerBidStorage.getUtility() ||
				isAcceptable(time, partnerBidStorage.getUtility(),
						stat.getAve(), stat.getVar(), tau)) {
					// System.out.println("Accept");
					action = new Accept(getAgentID(), partnerBid);
				} else {
					if (targetUtility < bestBid.getUtility()) {
						// ç›®æ¨™åŠ¹ç”¨å€¤ã‚ˆã‚Šã‚‚é«˜ã�„å€¤ã‚’ç›¸æ‰‹ã�Œæ—¢ã�«Offerã�—ã�¦ã�„ã�Ÿã‚‰ã��ã‚Œã‚’Offer
						action = generateOffer(bestBid.getBid());
						// System.out.printf("bestUtility: %f.5 ",
						// bestBid.getUtility());
					} else {
						action = generateOffer(myBidStorage.getBid());
					}

					// è‡ªèº«ã�®æœ€ã‚‚æ‚ªã�„Bidã‚’è¨˜æ†¶
					if ((targetUtility - myUtility) < 0.05
							&& myUtility < worstUtility) {
						worstUtility = myUtility;
					}
					// ç›®æ¨™ã�¨ã�‚ã�¾ã‚Šã�«ã‚‚é›¢ã‚Œã�Ÿbidã�¯æœ€å¤§Bidã�«ç½®æ�›ã�™ã‚‹
					if (Math.abs(targetUtility - myUtility) > 0.1) {
						action = generateOffer(candidateBids.get(0).getBid());
					}
					/*
					 * if(sessionNr>0 && time > 0.999 && (reservationValue *
					 * discountFactor) * 1.5 < partnerBidStorage.getUtility()) {
					 * System.out.println("Limit"); action = new
					 * Accept(getAgentID()); }
					 */
				}
			}
		} catch (Exception e) { // ä¾‹å¤–ç™ºç”Ÿæ™‚ã€�ç›¸æ‰‹ã�®å�ˆæ„�æ¡ˆã‚’å�—ã�‘ã‚‹
			e.printStackTrace();
			// System.out.println("Exception in ChooseAction:"+e.getMessage());
			// best guess if things go wrong.
			action = new Accept(getAgentID(),
					((ActionWithBid) actionOfPartner).getBid());
		}
		return action;
	}

	private BidStorage search2(Bid bid, double targetUtil) throws Exception {
		double min = 1;
		BidStorage ret = null;

		for (int i = 0; i < issues.size(); ++i) {
			IssueInteger lIssueInteger = (IssueInteger) issues.get(i);
			int issueIndexMin = lIssueInteger.getLowerBound();
			int issueIndexMax = lIssueInteger.getUpperBound();

			for (int j = issueIndexMin; j <= issueIndexMax; ++j) {
				Bid neighbourBid = new Bid(bid);
				neighbourBid = neighbourBid.putValue(i + 1,
						new ValueInteger(j));
				double u = utilitySpace.getUtility(neighbourBid);
				if (/* Math.abs(u-targetUtil) < min && */u > targetUtil) {
					min = u;
					ret = new BidStorage(neighbourBid, u, -1);
					System.out.println(ret);
				}
			}
		}

		return ret;
	}

	/**
	 * ç›¸æ‰‹ã�®å�ˆæ„�æ¡ˆã‚’å�—ã�‘ã‚‹ã�‹
	 */
	private boolean isAcceptable(double time, double u, double myu, double var,
			double tau) throws Exception {
		// ç›¸æ‰‹ã�®åŠ¹ç”¨å€¤ã�¨æ™‚é–“ã�§0~1ã�®å€¤ã‚’å¾—ã‚‹
		double p = agentKStorategy.pAccept(time, u, myu, var, tau);
		if (p < 0.1) {
			return false;
		}

		// System.out.printf(" P: %f.5\n", p);
		return p > Math.random();
	}

	private double fintarget2(double time, SummaryStatistics stat,
			Parameters param) {
		double ret = 1;
		double target = agentKStorategy.fintarget(time, stat.getAve(),
				stat.getVar(), 1);
		if (Double.isNaN(target)) {
			target = 1.0;
		}
		double co = 0;
		if (target > param.utility) {
			co = 0.8;
			if (param.time - time > 0) {
				co *= (1 - Math.pow((param.time - time) / param.time,
						param.alpha));
			}
		}
		ret = co * param.utility + (1 - co) * target;
		// System.out.printf("%.4f ",ret);
		return ret;
	}

	private double fintarget(double time, SummaryStatistics stat,
			Parameters param) {
		double ret = 1;
		double param_t = param.time * 0.85;
		double e = agentKStorategy.emax(stat.getAve(), stat.getVar());

		if (time < param_t) {
			ret = 1 - ((1 - param.utility)
					* Math.pow(time / param_t, 1 / param.alpha));
		} else {
			/*
			 * double t = (time - param_t) / (1 - param_t); double ave =
			 * stat.getAve() / param.utility; double var = stat.getVar() /
			 * Math.pow(param.utility,2); double target =
			 * agentKStorategy.fintarget(t, ave, var, 1); ret = target /
			 * param.utility;
			 */

			if (e > param.utility || Double.isNaN(e)) {
				e = param.utility;
			}
			ret = param.utility - (param.utility - e)
					* Math.pow((time - param_t) / (1 - param_t), param.beta);
			if (Double.isNaN(ret)) {
				ret = 1.0;
			}
		}

		// System.out.printf("%.4f ",ret);
		return ret;
	}

	private BidStorage searchBidWithApproximateUtlity(Bid startBid, double u)
			throws Exception {
		// Bid startBid = utilitySpace.getDomain().getRandomBid();
		// ç„¼ã��ã�ªã�¾ã�—æ³•ã�«ã‚ˆã‚‹æœ€é�©è§£æŽ¢ç´¢
		BidStorage optimumBid = null;
		try {
			optimumBid = simulatedAnealing.run(startBid, u, SA_K_MAX);
		} catch (Exception e) {
			return null;
		}
		return optimumBid;
	}

	private ArrayList<Bound> searchConstraint(Bid bid) {
		ArrayList<Bound> constraint = new ArrayList<Bound>();
		double u = getUtility(bid);
		return constraint;
	}

	@Override
	public void endSession(NegotiationResult result) {
		// System.out.printf("\n");
		try {
			sessionData.save(sessionNr, partnerBidHistory, result,
					utilitySpace.getUtility(result.getLastBid()),
					timeline.getTime());
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		saveSessionData(sessionData);
	}

	@Override
	public void ReceiveMessage(Action opponentAction) {
		actionOfPartner = opponentAction;
	}

	private Offer generateOffer(Bid bid) {
		return new Offer(getAgentID(), bid);
	}

	@Override
	public String getVersion() {
		return "5.0";
	}

	@Override
	public String getName() {
		return "Anac Sample Agent";
	}

	class AgentKStorategyComponentImpl implements IAgentKStorategyComponent {
		@Override
		public double g(double t) {
			return param.g;
		}
	}

	@Override
	public String getDescription() {
		return "ANAC2014 compatible with non-linear utility spaces";
	}
}
