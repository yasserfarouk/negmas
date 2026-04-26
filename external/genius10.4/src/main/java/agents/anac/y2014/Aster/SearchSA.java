package agents.anac.y2014.Aster;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import genius.core.Bid;
import genius.core.issue.Issue;
import genius.core.issue.IssueInteger;
import genius.core.issue.ValueInteger;
import genius.core.utility.AbstractUtilitySpace;

public class SearchSA {
	private AbstractUtilitySpace utilitySpace;
	private Random randomnr = new Random();
	private ArrayList<Bid> bidList = new ArrayList<Bid>();
	private double temperature;
	private static final double MINIMUM_UTILITY = 0.70; // åˆ�æœŸè§£ã�®æœ€ä½ŽåŠ¹ç”¨å€¤ï¼ˆé«˜ã�™ã�Žã‚‹ã�¨åˆ�æœŸè§£ã�®ç”Ÿæˆ�ã�Œã�§ã��ã�ªã�„ï¼‰
	private static final int NUMBER_ITERATIONS = 10000; // ç¹°ã‚Šè¿”ã�—å›žæ•°
	private static final int NUMBER_SOLUTIONS = 50;

	public SearchSA(AbstractUtilitySpace utilitySpace, int sessionNr) {
		this.utilitySpace = utilitySpace;
		this.temperature = 1000000;
		// this.fileWriter = createFileWriter(sessionNr);
	}

	public Bid getBidbySA(double bidTarget) throws Exception {
		Bid currentBid, nextBid, maxBid;
		double currentUtility, nextUtility, maxUtility, probability;
		int iteration = 0;

		// åˆ�æœŸè¨­å®š
		temperature = 1000000;
		do {
			currentBid = utilitySpace.getDomain().getRandomBid(null);
			currentUtility = utilitySpace.getUtility(currentBid);
		} while (currentUtility < MINIMUM_UTILITY);

		maxBid = currentBid;
		maxUtility = currentUtility;

		while ((iteration++ < NUMBER_ITERATIONS) && (maxUtility < bidTarget)) {
			nextBid = nextBid(currentBid);
			nextUtility = utilitySpace.getUtility(nextBid);
			temperature = calculateTemperature();
			probability = calculateProbability(currentUtility, nextUtility,
					temperature);

			if (probability > randomnr.nextDouble()) {
				currentBid = nextBid;
				currentUtility = utilitySpace.getUtility(nextBid);
				if (nextUtility > maxUtility) {
					maxBid = nextBid;
					maxUtility = nextUtility;
				}
			}
		}

		if (maxUtility < bidTarget) {
			return null;
		} else {
			return maxBid;
		}
	}

	public Bid getFirstBidbySA(double bidTarget) throws Exception {
		Bid currentBid, maxBid; // ç�¾åœ¨ã�®Bidã�¨æœ€é�©Bid
		double currentUtility, maxUtility; // ç�¾åœ¨ã�®åŠ¹ç”¨å€¤ã�¨æœ€å¤§åŠ¹ç”¨å€¤
		int iteration = 0; // ç¹°ã‚Šè¿”ã�—å›žæ•°

		// åˆ�æœŸè§£ã�®ç”Ÿæˆ�
		for (int i = 0; i < NUMBER_SOLUTIONS; i++) {
			do {
				currentBid = utilitySpace.getDomain().getRandomBid(null);
				currentUtility = utilitySpace.getUtility(currentBid);
			} while (currentUtility < MINIMUM_UTILITY);
			bidList.add(currentBid);
		}

		// æœ€é�©è§£ã�®æ›´æ–°
		maxBid = getMaxBid(bidList);
		maxUtility = utilitySpace.getUtility(maxBid);

		// fileWrite(fileWriter, maxUtility);

		// æŒ‡å®šå›žæ•°ã�‹æœ€é�©è§£ã�Œè¦‹ã�¤ã�‹ã‚‹ã�¾ã�§ç¹°ã‚Šè¿”ã�—
		while ((iteration++ < NUMBER_ITERATIONS) && (maxUtility < bidTarget)) {
			for (int i = 0; i < bidList.size(); i++) {
				Bid curBid = bidList.get(i);
				currentUtility = utilitySpace.getUtility(curBid);
				Bid nextBid = nextBid(curBid); // è¿‘å‚�Bidã‚’æŽ¢ç´¢
				double nextUtility = utilitySpace.getUtility(nextBid); // è¿‘å‚�Bidã�®åŠ¹ç”¨å€¤
				double probability = calculateProbability(currentUtility,
						nextUtility, temperature); // é�·ç§»ç¢ºçŽ‡ã�®ç®—å‡º

				if (probability > randomnr.nextDouble()) {
					bidList.set(i, nextBid); // Bidã�®æ›´æ–°
				}
			}

			// æ¸©åº¦ã�®ç®—å‡º
			temperature = calculateTemperature();

			// æœ€é�©è§£ã�®æ›´æ–°
			maxBid = getMaxBid(bidList);
			maxUtility = utilitySpace.getUtility(maxBid);

			// fileWrite(fileWriter, maxUtility);

		}

		// fileWriter.close();
		return maxBid;
	}

	/**
	 * é�·ç§»ç¢ºçŽ‡ã�®ç®—å‡º
	 *
	 * @param currentUtil
	 *            ç�¾åœ¨ã�®åŠ¹ç”¨å€¤
	 * @param nextUtil
	 *            è¿‘å‚�ã�®åŠ¹ç”¨å€¤
	 * @param temperature
	 *            æ¸©åº¦
	 * @return é�·ç§»ç¢ºçŽ‡
	 */
	private double calculateProbability(double currentUtil, double nextUtil,
			double temperature) {
		double diff = currentUtil - nextUtil;
		if (diff > 0.0) {
			return Math.exp(-diff / temperature); // ç�¾åœ¨ã�®åŠ¹ç”¨å€¤ã�®æ–¹ã�Œé«˜ã�„å ´å�ˆ
		} else {
			return 1.0; // è¿‘å‚�ã�®åŠ¹ç”¨å€¤ã�®æ–¹ã�Œé«˜ã�„å ´å�ˆ
		}
	}

	/**
	 * æ¸©åº¦ã�®ç®—å‡º
	 *
	 * @param iteration
	 *            ç�¾åœ¨ã�®è©¦è¡Œå›žæ•°
	 * @return æ¸©åº¦
	 */
	private double calculateTemperature() {
		return temperature *= 0.97;
	}

	/**
	 * æŒ‡å®šã�—ã�ŸIssueã�«ã�Šã�‘ã‚‹è¿‘å‚�ã�®Bidã‚’æŽ¢ç´¢
	 *
	 * @param bid
	 *            ç�¾åœ¨ã�®Bid
	 * @return è¿‘å‚�Bid
	 * @throws Exception
	 */
	private Bid nextBid(Bid bid) throws Exception {
		List<Issue> issues = utilitySpace.getDomain().getIssues(); // å…¨issueã�®å�–å¾—
		Bid nextBid = new Bid(bid); // ç�¾åœ¨ã�®Bidã‚’ã‚³ãƒ”ãƒ¼

		int issueIndex = randomnr.nextInt(issues.size()); // Issueã‚’ãƒ©ãƒ³ãƒ€ãƒ ã�«æŒ‡å®š
		IssueInteger lIssueInteger = (IssueInteger) issues.get(issueIndex); // æŒ‡å®šã�—ã�Ÿindexã�®issueã‚’å�–å¾—
		int issueNumber = lIssueInteger.getNumber(); // issueç•ªå�·
		int issueIndexMin = lIssueInteger.getLowerBound(); // issueã�®ä¸‹é™�å€¤
		int issueIndexMax = lIssueInteger.getUpperBound(); // issueã�®ä¸Šé™�å€¤
		int optionIndex = 0; // å¤‰æ›´ã�™ã‚‹Valueå€¤

		ValueInteger lIssueValue = (ValueInteger) bid.getValue(issueNumber); // æŒ‡å®šã�—ã�Ÿissueã�®Value
		int issueValue = Integer.valueOf(lIssueValue.toString()).intValue();
		optionIndex = nextOptionIndex(issueIndexMin, issueIndexMax, issueValue); // å€¤ã‚’1å¢—æ¸›ã�•ã�›ã‚‹

		nextBid = nextBid.putValue(issueNumber, new ValueInteger(optionIndex)); // ç�¾åœ¨ã�®Bidã�‹ã‚‰Issueã�®å€¤ã‚’å…¥ã‚Œæ›¿ã�ˆã‚‹

		return nextBid;
	}

	/**
	 * è¿‘å‚�æŽ¢ç´¢
	 *
	 * @param issueIndexMin
	 *            ä¸‹é™�å€¤
	 * @param issueIndexMax
	 *            ä¸Šé™�å€¤
	 * @param issueValue
	 *            ç�¾åœ¨ã�®å€¤
	 * @return å€¤ã‚’1ã� ã�‘å¢—æ¸›ã�•ã�›ã‚‹
	 */
	private int nextOptionIndex(int issueIndexMin, int issueIndexMax,
			int issueValue) {
		int step = 1; // å€¤ã‚’å¢—æ¸›ã�•ã�›ã‚‹å¹…
		int direction = randomnr.nextBoolean() ? 1 : -1; // ãƒ©ãƒ³ãƒ€ãƒ ã�«å¢—æ¸›ã‚’æ±ºå®š

		if (issueIndexMin < issueIndexMax) {
			if ((issueValue + step) > issueIndexMax) { // +1ã�™ã‚‹ã�¨ä¸Šé™�å€¤ã‚’è¶…ã�ˆã‚‹å ´å�ˆ
				direction = -1;
			} else if ((issueValue - step) < issueIndexMin) { // -1ã�™ã‚‹ã�¨ä¸‹é™�å€¤ã‚’ä¸‹å›žã‚‹å ´å�ˆ
				direction = 1;
			}
		} else {
			return issueValue; // issueIndexMin == issueIndexMaxã�®å ´å�ˆ
		}

		return issueValue + step * direction;
	}

	private Bid getMaxBid(ArrayList<Bid> bidList) throws Exception {
		Bid maxBid = null;
		double maxUtil = 0.0;
		for (Bid curBid : bidList) {
			double curUtil = utilitySpace.getUtility(curBid);
			if (curUtil > maxUtil) {
				maxBid = curBid;
				maxUtil = curUtil;
			}
		}
		return maxBid;
	}
}
