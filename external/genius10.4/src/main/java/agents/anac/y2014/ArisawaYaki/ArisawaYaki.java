package agents.anac.y2014.ArisawaYaki;

import java.io.Serializable;
import java.util.List;
import java.util.Random;

import genius.core.Agent;
import genius.core.Bid;
import genius.core.BidHistory;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.DefaultAction;
import genius.core.actions.Offer;
import genius.core.bidding.BidDetails;
import genius.core.issue.Issue;
import genius.core.issue.IssueInteger;
import genius.core.issue.ValueInteger;

public class ArisawaYaki extends Agent {

	/** The minimum utility a bid should have to be accepted or offered. */
	private double MINIMUM_BID_UTILITY;
	/** The opponent's last action. */
	private Bid opponentLastBid;
	/** Bid with the highest possible utility. */
	// private Bid maxBid;
	/** ‘ŠŽè‚ª�o‚µ‚½ƒrƒbƒh‚Ì“à�A�Å‚à�‚‚¢Œø—p’l‚Ìƒrƒbƒh‚ð•Û‘¶ */
	private Bid opponentBestBid;
	/** ‘ŠŽè‚ÌBid‚ð•Û‘¶‚·‚éArrayList */
	// private ArrayList<Bid> opponentBids;
	/** sigmoid‚ÌŒX‚« */
	private double a = 40.0;
	/** �¡‰ñ‚ÌBidHistory */
	private BidHistory currBidHistory;
	/** Accept‚·‚éŒø—p’l */
	private double acceptUtility;
	/** ‚ ‚é‹æŠÔ‚ÌŒø—p’l‚Ì•½‹Ï */
	private double ave;
	/** ‘O‰ñ‚Ì‚ ‚é‹æŠÔ‚ÌŒø—p’l‚Ì•½‹Ï */
	private double pre_ave;
	/** ŽžŠÔ‚ð‹L˜^ */
	private double laptime;
	/** �s“®‰ñ�”‚ð‹L˜^ */
	private int count;
	/** �I—¹ŠÔ�Û‚Ì�s“®‰ñ�” */
	private int countFinal;

	public ArisawaYaki() {
	}

	/**
	 * Initialize the target utility to MAX(rv, max). Where rv is the
	 * reservation value of the preference profile and max is the highest
	 * utility received on the current preference profile.
	 */
	@Override
	public void init() {
		MINIMUM_BID_UTILITY = 1.0;
		opponentBestBid = null;
		opponentLastBid = null;
		currBidHistory = new BidHistory();
		acceptUtility = 0.85;
		pre_ave = 0.0;
		ave = 0.0;
		laptime = 0.0;
		count = 1;
		countFinal = -1;

		int num = sessionNr;
		Serializable prev = this.loadSessionData();
		// if (prev != null) {
		// double previousOutcome = (Double) prev;
		// //�@‘O‰ñŠl“¾Œø—p’l
		// //BidHistory history = (BidHistory) prev;
		// MINIMUM_BID_UTILITY =
		// Math.max(Math.max(utilitySpace.getReservationValueUndiscounted(),
		// previousOutcome), 0.5);
		// }
		// else{
		// MINIMUM_BID_UTILITY =
		// Math.max(utilitySpace.getReservationValueUndiscounted(), 0.5);
		// }
	}

	@Override
	public String getVersion() {
		return "1.0";
	}

	@Override
	public String getName() {
		return "ArisawaYaki";
	}

	/**
	 * Set the target utility for the next match on the same preference profile.
	 * If the received utility is higher than the current target, save the
	 * received utility as the new target utility.
	 */
	// public void endSession(NegotiationResult result) {
	// if (result.getMyDiscountedUtility() > MINIMUM_BID_UTILITY) {
	// saveSessionData(new Double(result.getMyDiscountedUtility()));
	// }
	// //System.out.println(result);
	// }

	/**
	 * Retrieve the bid from the opponent's last action.
	 */
	@Override
	public void ReceiveMessage(Action opponentAction) {
		count++;
		if (opponentAction != null) {
			opponentLastBid = DefaultAction.getBidFromAction(opponentAction); // ‘ŠŽè‚ÌlastBid
			currBidHistory.add(new BidDetails(opponentLastBid,
					getUtility(opponentLastBid), timeline.getTime()));

			// opponentBestBid‚Ì�X�V
			if (opponentBestBid != null) {
				if (getUtility(opponentLastBid) > getUtility(opponentBestBid)) {
					opponentBestBid = new Bid(opponentLastBid);
				}
			} else {
				opponentBestBid = new Bid(opponentLastBid);
			}
		}
		// System.out.println("debug: opponentBestBid"+opponentBestBid);
	}

	/**
	 * Accept if the utility of the opponent's is higher than the target
	 * utility; else return a random bid with a utility at least equal to the
	 * target utility.
	 */
	@Override
	public Action chooseAction() {
		double time = timeline.getTime();
		if (time >= 0.99) {
			return finalPhase();
		}
		calculateMinimumBidUtil(time);

		if (opponentLastBid != null
				&& getUtility(opponentLastBid) >= acceptUtility) {
			return new Accept(getAgentID(), opponentLastBid);
		}

		Bid bid = null;
		try {
			if (opponentLastBid != null) {
				bid = getBidBySA2(MINIMUM_BID_UTILITY);
			} else {
				bid = getBidBySA(MINIMUM_BID_UTILITY);
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		return new Offer(getAgentID(), bid);
	}

	/**
	 * MINIMUM_BID_UTILITY‚ÆacceptUtility‚Ì’l‚ðŒˆ‚ß‚é
	 * 
	 * @param time
	 */
	private void calculateMinimumBidUtil(double time) {
		double reservationValue = utilitySpace
				.getReservationValueWithDiscount(time);
		double nowUtility = sigmoid(time);

		if (reservationValue > nowUtility) {
			acceptUtility = reservationValue;
		} else {
			acceptUtility = nowUtility;
		}

		if (acceptUtility > 0.87) {
			acceptUtility = 0.87;
		}

		if (count % 50 == 0) {
			setUtilityByAve();
		}

		if (opponentBestBid != null) {
			double minUtility = getUtility(opponentBestBid);
			if (acceptUtility < minUtility) {
				acceptUtility = minUtility;
			}
			if (MINIMUM_BID_UTILITY < minUtility) {
				MINIMUM_BID_UTILITY = minUtility;
			}
		}

		if (MINIMUM_BID_UTILITY >= 0.95) {
			MINIMUM_BID_UTILITY = 0.95;
		}
		// System.out.println("debug MINIMUM_BID_UTILITY "+MINIMUM_BID_UTILITY);
		// System.out.println("debug acceptUtility "+acceptUtility);
	}

	/**
	 * ‘ŠŽè‚ÌBid‚Ì•½‹Ï‚ð—p‚¢‚ÄOffer‚·‚éBid‚ÌŒø—p’l‚ðŒˆ‚ß‚é
	 * 
	 * @param time
	 */
	private void setUtilityByAve() {
		double time = timeline.getTime();
		// System.out.println("debug size "+currBidHistory.size());
		BidHistory bids = currBidHistory.filterBetweenTime(laptime, time);
		// System.out.println("debug size "+bids.size());

		List<BidDetails> list = bids.getHistory();
		int size = list.size();
		double sum = 0.0;

		// System.out.println("debug list "+list.toString());
		// System.out.println("debug laptime "+laptime);
		// System.out.println("debug time "+time);

		for (int i = 0; i < size; i++) {
			sum += getUtility(list.get(i).getBid());
		}

		ave = sum / size;

		if (ave - pre_ave > 0) {
			MINIMUM_BID_UTILITY -= 0.03;
		} else if (ave - pre_ave < 0) {
			MINIMUM_BID_UTILITY += 0.03;
		}

		laptime = time;
		pre_ave = ave;
		System.out.println("debug ave " + ave);
		System.out.println("debug pre_ave " + pre_ave);
	}

	private Action finalPhase() {
		countFinal++;
		List<BidDetails> list = currBidHistory.getNBestBids(7);
		return new Offer(getAgentID(), list.get(countFinal).getBid());
	}

	private double sigmoid(double x) {
		return (Math.exp(a * (1 - x)) - 1) / (Math.exp(a * (1 - x)) + 1)
				* Math.pow(utilitySpace.getDiscountFactor(), x);
	}

	/**
	 * �Ä‚«‚È‚Ü‚µ–@‚ð—p‚¢‚ÄtargetˆÈ�ã‚ÌBid‚ð‹�‚ß‚é
	 * 
	 * @param target
	 * @return
	 * @throws Exception
	 */
	private Bid getBidBySA(double target) throws Exception {
		double t = 1000000;
		double cool = 0.999;
		List<Issue> issues = utilitySpace.getDomain().getRandomBid(null)
				.getIssues();
		Random rnd = new Random();
		Bid nowBid;

		nowBid = utilitySpace.getDomain().getRandomBid(null);
		Bid nextBid = null;

		// ƒ‰ƒ“ƒ_ƒ€‚Å‚Æ‚Á‚½’l‚ªtarget‚ð’´‚¦‚Ä‚¢‚½‚ç‚±‚ê‚ð•Ô‚·
		if (getUtility(nowBid) >= target) {
			return nowBid;
		}

		while (t > 0.001) {
			int index = rnd.nextInt(issues.size());
			IssueInteger isInt = (IssueInteger) issues.get(index);
			ValueInteger vlInt = (ValueInteger) nowBid.getValue(index + 1);

			// nowBid‚ðƒ‰ƒ“ƒ_ƒ€‚É‚Ð‚Æ‚Â‚¸‚ç‚µ‚½nextBid‚ð�¶�¬
			if (isInt.getUpperBound() == vlInt.getValue()) { // issue‚Ì�Å‘å’l‚Ì‚Æ‚«
				vlInt = new ValueInteger(vlInt.getValue() - 1);
				nextBid = new Bid(nowBid);
				nextBid = nextBid.putValue(index + 1, vlInt);
			} else if (isInt.getLowerBound() == vlInt.getValue()) { // issue‚Ì�Å�¬’l‚Ì‚Æ‚«
				vlInt = new ValueInteger(vlInt.getValue() + 1);
				nextBid = new Bid(nowBid);
				nextBid = nextBid.putValue(index + 1, vlInt);
			} else {
				int updown = rnd.nextInt(2);
				if (updown == 0) {
					nextBid = new Bid(nowBid);
					vlInt = new ValueInteger(vlInt.getValue() - 1);
					nextBid = nextBid.putValue(index + 1, vlInt);
				} else {
					nextBid = new Bid(nowBid);
					vlInt = new ValueInteger(vlInt.getValue() + 1);
					nextBid = nextBid.putValue(index + 1, vlInt);
				}
			}

			// ‘JˆÚ�æ‚Ìƒrƒbƒh‚ªtarget‚ð’´‚¦‚½‚ç‚±‚ê‚ð•Ô‚·
			if (getUtility(nextBid) > target) {
				return nextBid;
			}
			double utility = getUtility(nextBid) - getUtility(nowBid);

			// utility‚ª‘�‰Á‚µ‚Ä‚¢‚é‚©�A‚ ‚éŠm—¦‚Å‚»‚Ì•ûŒü‚ÖˆÚ“®
			double p = Math.pow(Math.E, -Math.abs(utility) / t);
			if (utility >= 0 || Math.random() < p) {
				nowBid = new Bid(nextBid);
			}
			t *= cool;
		}
		return nextBid;
	}

	/**
	 * �Ä‚«‚È‚Ü‚µ–@‚ð—p‚¢‚ÄtargetˆÈ�ã‚ÌBid‚ð‹�‚ß‚é
	 * 
	 * @param target
	 * @return
	 * @throws Exception
	 */
	private Bid getBidBySA2(double target) throws Exception {
		double t = 1000;
		double cool = 0.99;
		List<Issue> issues = utilitySpace.getDomain().getRandomBid(null)
				.getIssues();
		Bid nowBid = new Bid(opponentLastBid);
		Bid nextBid = new Bid(opponentLastBid);

		while (t > 0.001) {
			// Bid‚Ìissue‚Ì‚Ð‚Æ‚Â‚ð•Ï‚¦‚Ä‚Ý‚é
			Bid tempBid = null;
			double tempUtility = -1.0;
			// nextBid‚ð�X�V
			for (int i = 0; i < issues.size(); i++) {
				IssueInteger isInt = (IssueInteger) issues.get(i);
				int min = isInt.getLowerBound();
				int max = isInt.getUpperBound();
				for (int j = min; j < max; j++) {
					tempBid = new Bid(nowBid);
					tempBid = tempBid.putValue(i + 1, new ValueInteger(j));
					if (getUtility(tempBid) > tempUtility) {
						tempUtility = getUtility(tempBid);
						// ‘JˆÚ�æ‚Ìƒrƒbƒh‚ªtarget‚ð’´‚¦‚½‚ç‚±‚ê‚ð•Ô‚·
						if (tempUtility > target) {
							return tempBid;
						}
						nextBid = new Bid(tempBid);
					}
				}
			}

			// ‘JˆÚ�æ‚Ìƒrƒbƒh‚ªtarget‚ð’´‚¦‚½‚ç‚±‚ê‚ð•Ô‚·
			if (getUtility(nextBid) > target) {
				return nextBid;
			}
			double utility = getUtility(nextBid) - getUtility(nowBid);

			// utility‚ª‘�‰Á‚µ‚Ä‚¢‚é‚©�A‚ ‚éŠm—¦‚Å‚»‚Ì•ûŒü‚ÖˆÚ“®
			double p = Math.pow(Math.E, -Math.abs(utility) / t);
			if (utility >= 0 || Math.random() < p) {
				nowBid = new Bid(nextBid);
			}
			t *= cool;
		}
		return nextBid;
	}

	@Override
	public String getDescription() {
		return "ANAC2014 compatible with non-linear utility spaces";
	}
}

// class MySessionData implements Serializable{
// BidHistory history;
// Bid lastBid;
//
// public MySessionData(){
//
// }
// }