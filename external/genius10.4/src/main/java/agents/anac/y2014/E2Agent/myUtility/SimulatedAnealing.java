package agents.anac.y2014.E2Agent.myUtility;

import java.util.List;
import java.util.Random;

import genius.core.Bid;
import genius.core.issue.Issue;
import genius.core.issue.IssueInteger;
import genius.core.issue.ValueInteger;
import genius.core.utility.AbstractUtilitySpace;

public class SimulatedAnealing {
	private List<Issue> issues = null; // åŠ¹ç”¨ç©ºé–“ã�®å…¨ã�¦ã�®è«–ç‚¹
	private Random randomnr = null;
	private AbstractUtilitySpace utilitySpace = null;
	private MethodForSA methods = null;

	public SimulatedAnealing(AbstractUtilitySpace u) {
		utilitySpace = u;
		issues = utilitySpace.getDomain().getIssues();
		randomnr = new Random();
		;
		// methods = new RandomSearch();
		methods = new StepSearch();
	}

	/**
	 * ç„¼ã��ã�ªã�¾ã�—æ³•ã‚’åˆ©ç”¨ã�—ã�¦æœ€é�©è§£ã‚’æŽ¢ç´¢
	 * 
	 * @param startBid
	 *            æŽ¢ç´¢é–‹å§‹Bid
	 * @param thresholdUtility
	 *            åŠ¹ç”¨å€¤é–¾å€¤
	 * @param kmax
	 *            æœ€å¤§ãƒ«ãƒ¼ãƒ—å›žæ•°
	 * @return æœ€é�©Bid
	 * @throws Exception
	 */
	public BidStorage run(Bid startBid, double targetUtility, int kmax)
			throws Exception {
		int k = 0;
		Bid bid = startBid; // æœ€åˆ�ã�®Bidä½�ç½®
		double utility = utilitySpace.getUtility(bid); // åŠ¹ç”¨å€¤
		Bid bestBid = bid; // æœ€é�©å€™è£œBid
		double bestUtility = utility; // æœ€é�©å€™è£œBidåŠ¹ç”¨å€¤

		// æŒ‡å®šå›žæ•°ã�¾ã�§ãƒ«ãƒ¼ãƒ—
		while (k < kmax) {
			// è¿‘éš£ã�®Bidã‚’å�–å¾—
			Bid neighbourBid = methods.searchNeighbourBid(bid);
			double neighbourUtility = utilitySpace.getUtility(neighbourBid);
			// æ¸©åº¦
			double temperature = methods.calculateTemperature(k, kmax);
			// é�·ç§»ã�™ã‚‹ã�‹ã�©ã�†ã�‹
			boolean trans = methods.calculateWheterToUpdateBid(targetUtility,
					utility, neighbourUtility, temperature);

			if (trans) {
				utility = neighbourUtility;
				bid = neighbourBid;
				// System.out.printf("%f ",utility);
				// æœ€é�©åŠ¹ç”¨å€¤æ›´æ–°
				if (Math.abs(targetUtility - bestUtility) > Math
						.abs(targetUtility - utility)) {
					bestUtility = utility;
					bestBid = bid;
					if (Math.abs(targetUtility - utility) < 0.01) {
						break;
					}
				}
			}
			++k;
		}
		// System.out.printf("\n",utility);
		return new BidStorage(bestBid, bestUtility, -1);
	}

	/**
	 * æ¤œç´¢ã‚¯ãƒ©ã‚¹
	 */
	class RandomSearch implements MethodForSA {
		@Override
		public Bid searchNeighbourBid(Bid bid) {
			// æ•´æ•°åž‹è«–ç‚¹ã�®å®šç¾©ã€�é�¸æŠžå�¯èƒ½ã�ªè«–ç‚¹å€¤ã�®ç¯„å›²ã‚’å¾—ã‚‰ã‚Œã‚‹
			IssueInteger lIssueInteger = (IssueInteger) issues.get(randomnr
					.nextInt(issues.size()));
			int issueIndexMin = lIssueInteger.getLowerBound();
			int issueIndexMax = lIssueInteger.getUpperBound();
			int optionIndex = issueIndexMin;

			if (issueIndexMin < issueIndexMax) {
				// æœ€å¤§æœ€å°�å€¤ã�®é–“ã�®ã‚¤ãƒ³ãƒ‡ãƒƒã‚¯ã‚¹ã‚’å¾—ã‚‹
				optionIndex = issueIndexMin
						+ randomnr.nextInt(issueIndexMax - issueIndexMin);
			}

			Bid neighbourBid = new Bid(bid);
			neighbourBid = neighbourBid.putValue(lIssueInteger.getNumber(),
					new ValueInteger(optionIndex));
			return neighbourBid;
		}

		@Override
		public double calculateTemperature(int k, int kmax) {
			double T = 1000;
			// double val = T * Math.pow(1.0 - ((double)k / kmax), 2);
			double val = T * Math.pow(0.9, k);
			// System.out.printf("%f\n",val);
			return val;
		}

		@Override
		public boolean calculateWheterToUpdateBid(double targetUtility,
				double utility, double neighbourUtility, double temperature) {
			double diff = Math.abs(targetUtility - neighbourUtility)
					- Math.abs(targetUtility - utility);
			double p = 1.0;

			if (0.0 < diff) {
				p = Math.exp(-diff / temperature);
			}
			return randomnr.nextDouble() < p;
		}
	}

	/**
	 * æ¤œç´¢ã‚¯ãƒ©ã‚¹
	 */
	class StepSearch implements MethodForSA {
		private int indexes = 1;

		public StepSearch() {
			indexes = issues.size() / 10;
			if (indexes < 1) {
				indexes = 1;
			}
		}

		@Override
		public Bid searchNeighbourBid(Bid bid) {
			Bid neighbourBid = new Bid(bid);
			for (int i = 0; i < indexes; ++i) {
				// æ•´æ•°åž‹è«–ç‚¹ã�®å®šç¾©ã€�é�¸æŠžå�¯èƒ½ã�ªè«–ç‚¹å€¤ã�®ç¯„å›²ã‚’å¾—ã‚‰ã‚Œã‚‹
				int targetIssueIndex = randomnr.nextInt(issues.size());
				IssueInteger lIssueInteger = (IssueInteger) issues
						.get(targetIssueIndex);
				int issueIndexMin = lIssueInteger.getLowerBound();
				int issueIndexMax = lIssueInteger.getUpperBound();
				int currentIndex;
				try {
					currentIndex = ((ValueInteger) bid
							.getValue(targetIssueIndex + 1)).getValue();
				} catch (Exception e) {
					e.printStackTrace();
					currentIndex = issueIndexMin;
				}

				currentIndex += (randomnr.nextFloat() < 0.5) ? 1 : -1;

				if (currentIndex < issueIndexMin) {
					currentIndex = issueIndexMax;
				} else if (currentIndex > issueIndexMax) {
					currentIndex = issueIndexMin;
				}

				neighbourBid = neighbourBid.putValue(targetIssueIndex + 1,
						new ValueInteger(currentIndex));
			}
			return neighbourBid;
		}

		@Override
		public double calculateTemperature(int k, int kmax) {
			double T = 1000;
			// double val = T * Math.pow(1.0 - ((double)k / kmax), 2);
			double val = T * Math.pow(0.95, k);
			// System.out.printf("%f\n",val);
			return val;
		}

		@Override
		public boolean calculateWheterToUpdateBid(double targetUtility,
				double utility, double neighbourUtility, double temperature) {
			double diff = Math.abs(targetUtility - neighbourUtility)
					- Math.abs(targetUtility - utility);
			double p = 1.0;

			if (0.0 < diff) {
				p = Math.exp(-diff / temperature);
			}
			return randomnr.nextDouble() < p;
		}
	}

}

interface MethodForSA {
	/**
	 * éš£æŽ¥ã�™ã‚‹Bidã‚’æŽ¢ç´¢
	 * 
	 * @param bid
	 *            å¯¾è±¡ã�¨ã�™ã‚‹Bid
	 * @return éš£æŽ¥ã�™ã‚‹Bid
	 */
	Bid searchNeighbourBid(Bid bid);

	/**
	 * æ¸©åº¦ã�®ç®—å‡º
	 * 
	 * @param k
	 *            ãƒ«ãƒ¼ãƒ—å›žæ•°
	 * @param kmax
	 *            æœ€å¤§ãƒ«ãƒ¼ãƒ—å›žæ•°
	 * @return æ¸©åº¦
	 */
	double calculateTemperature(int k, int kmax);

	/**
	 * é�·ç§»ç¢ºçŽ‡ã�®ç®—å‡º
	 * 
	 * @param utility
	 *            åŠ¹ç”¨å€¤
	 * @param neighbourUtility
	 *            è¿‘éš£ã�®åŠ¹ç”¨å€¤
	 * @param temperature
	 *            æ¸©åº¦
	 * @return é�·ç§»ç¢ºçŽ‡
	 */
	// double calculateTransitionProbability(double targetUtility,
	// double utility, double neighbourUtility, double temperature);
	boolean calculateWheterToUpdateBid(double targetUtility, double utility,
			double neighbourUtility, double temperature);
}
