package agents.anac.y2014.Aster;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Iterator;

import genius.core.Agent;
import genius.core.Bid;
import genius.core.BidHistory;
import genius.core.NegotiationResult;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.Offer;
import genius.core.bidding.BidDetails;

/**
 * @author S.Morii Some improvements over the AgentMRK2.
 *
 **/

public class Aster extends Agent {
	private double time;
	private double bidTarget;
	private double acceptP;
	private double opponentLastUtility;
	private double opponentMaxUtility;
	private double opponentMinUtility;
	private double opponentPrevMaxUtility;
	private double opponentPrevMinUtility;
	private boolean immediateDecision;
	private Bid myLastBid;
	private Bid opponentMaxBid;
	private Bid opponentLastBid;
	private ArrayList<Bid> agreementBidList;
	private ArrayList<Bid> selectAgreementBidList;
	private ArrayList<Bid> selectMyBidList;
	private ArrayList<Bid> selectOpponentBidList;
	private BidHistory myBidHistory;
	private BidHistory opponentBidHistory;
	private ChooseAction chooseAction;
	private GetAcceptProb getAcceptProb;
	private SearchBid searchBid;
	private SearchSA searchSA;
	private static final double AGREE_EPSILON = 0.005;
	private static final double DEAD_LINE = 0.990;
	private static final double ACCEPT_CONCESSION = 0.95;
	private static final double ACCEPT_PROB = 0.40;

	@Override
	public void init() {
		this.bidTarget = 0.990;
		this.opponentMinUtility = 1.0;
		this.selectAgreementBidList = new ArrayList<Bid>();
		this.selectMyBidList = new ArrayList<Bid>();
		this.selectOpponentBidList = new ArrayList<Bid>();
		this.myBidHistory = new BidHistory();
		this.opponentBidHistory = new BidHistory();
		this.chooseAction = new ChooseAction(utilitySpace);
		this.getAcceptProb = new GetAcceptProb(utilitySpace);
		this.searchBid = new SearchBid(utilitySpace);
		this.searchSA = new SearchSA(utilitySpace, this.sessionNr);
		this.beginSession();
	}

	@Override
	public String getVersion() {
		return "1.0.2";
	}

	@Override
	public String getName() {
		return "Aster";
	}

	@Override
	public void ReceiveMessage(Action opponentAction) {
		try {
			if (opponentAction instanceof Offer) {
				time = timeline.getTime();

				// ç›¸æ‰‹ã�®æœ€çµ‚Bid
				opponentLastBid = ((Offer) opponentAction).getBid();
				opponentLastUtility = utilitySpace.getUtility(opponentLastBid);

				// ç›¸æ‰‹ã�®æœ€å°�Bidã‚’æ›´æ–°
				if (opponentLastUtility < opponentMinUtility) {
					opponentMinUtility = opponentLastUtility;
					getAcceptProb.updateMinConcessionUtil(opponentMinUtility);
				}

				// ç›¸æ‰‹ã�®æœ€å¤§Bidã‚’æ›´æ–°
				if (opponentLastUtility > opponentMaxUtility) {
					opponentMaxBid = opponentLastBid;
					opponentMaxUtility = opponentLastUtility;
					if (opponentMaxUtility > opponentPrevMaxUtility) {
						immediateDecision = false;
					}
				}

				// ç›¸æ‰‹ã�®Bidã‚’ä¿�å­˜
				opponentBidHistory
						.add(convertBidDetails(opponentLastBid, time));
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	@Override
	public Action chooseAction() {
		Action action = null;
		Bid actionBid = null;

		try {
			time = timeline.getTime();

			// åˆ�æœŸBid
			if ((myLastBid == null) || (opponentLastBid == null)) {
				myLastBid = searchSA.getFirstBidbySA(bidTarget);
				selectMyBidList.add(myLastBid);
				return offerAndSaveBid(myLastBid);
			}

			// å�ˆæ„�å½¢æˆ�çŽ‡ã�®ç®—å‡º
			double maxUtil = opponentMaxUtility > opponentPrevMaxUtility
					? opponentMaxUtility : opponentPrevMaxUtility;
			acceptP = getAcceptProb.getAcceptProbability(opponentLastBid,
					maxUtil, time);

			// é–¾å€¤ã�®ç®—å‡º
			bidTarget = getAcceptProb.getCurrentBidTarget();

			// å�ˆæ„�åˆ¤å®š
			if (isAcceptable(time)) {
				action = new Accept(getAgentID(), opponentLastBid);
				return action;
			}

			// è­²æ­©åˆ¤å®š
			if ((actionBid = isConsiderable()) != null) {
				return offerAndSaveBid(actionBid);
			}

			// BidæŽ¢ç´¢
			// ç›¸æ‰‹ã�®Bidã�®è¿‘å‚�æŽ¢ç´¢
			selectOpponentBidList = searchBid.searchOfferingBid(
					selectOpponentBidList, opponentLastBid, bidTarget);
			if (!selectMyBidList.isEmpty()) {
				addSelectBidList(bidTarget);
			}

			// SAã�§æŽ¢ç´¢
			actionBid = searchSA.getBidbySA(bidTarget);
			if (actionBid != null) {
				if (!selectMyBidList.contains(actionBid)) {
					selectMyBidList.add(actionBid);
				}
			}

			// Bidã�®æ±ºå®š
			if (!selectOpponentBidList.isEmpty()) {
				actionBid = chooseAction.nextOfferingBid(bidTarget,
						selectMyBidList, selectOpponentBidList);
			} else {
				actionBid = chooseAction.nextOfferingBid(bidTarget,
						selectMyBidList, true);
			}

			// FOR_DEBUG
			// printCurrentDetails();

			// ã‚¢ãƒƒãƒ—ãƒ‡ãƒ¼ãƒˆ
			myLastBid = actionBid;

		} catch (Exception e) {
			e.printStackTrace();
		}

		// ã‚ªãƒ•ã‚¡ãƒ¼
		return offerAndSaveBid(actionBid);
	}

	@Override
	public void endSession(NegotiationResult result) {
		MyPrevSessionData mySessionData = null;

		try {
			boolean immediateDecision = false;
			double resultUtil;
			double agreediff = 0.0;

			if (result.isAgreement()) {
				Bid agreementBid = result.getLastBid();
				resultUtil = utilitySpace.getUtility(agreementBid);

				// é�ŽåŽ»ã�®å�ˆæ„�æ¡ˆã�®å¹³å�‡åŠ¹ç”¨å€¤
				if (agreementBidList.size() >= this.sessionsTotal / 3) {
					double agreementAverage = 0.0;
					for (Bid bid : agreementBidList) {
						agreementAverage += utilitySpace.getUtility(bid);
					}
					agreementAverage /= agreementBidList.size();
					agreediff = Math.abs(agreementAverage - resultUtil);
					if ((agreediff < AGREE_EPSILON)
							&& (utilitySpace.getDiscountFactor() < 0.95)) {
						immediateDecision = true; // é�ŽåŽ»ã�®å�ˆæ„�å¹³å�‡å€¤ã�¨ã�®å·®ã�Œå��åˆ†ã�«å°�ã�•ã�„ã�ªã‚‰æ¬¡å›žä»¥é™�å�³æ±º
					}
				}

				agreementBidList.add(agreementBid);

			} else {
				resultUtil = opponentMaxUtility;
			}

			// é�ŽåŽ»ã�®å�ˆæ„�æœ€å¤§åŠ¹ç”¨å€¤
			if (resultUtil > opponentPrevMaxUtility) {
				opponentPrevMaxUtility = resultUtil;
			}

			opponentPrevMinUtility = opponentBidHistory.getWorstBidDetails()
					.getMyUndiscountedUtil();

			mySessionData = new MyPrevSessionData(agreementBidList,
					opponentBidHistory, result, opponentPrevMaxUtility,
					opponentPrevMinUtility, time, immediateDecision);
		} catch (Exception e) {
			e.printStackTrace();
		}

		this.saveSessionData(mySessionData);
	}

	private void beginSession() {
		Serializable prev = this.loadSessionData();
		MyPrevSessionData mySessionData = (MyPrevSessionData) prev;

		try {
			if (mySessionData != null) {
				agreementBidList = mySessionData.agreementBidList;
				opponentPrevMaxUtility = mySessionData.opponentPrevMaxUtility;
				opponentPrevMinUtility = mySessionData.opponentPrevMinUtility;
				immediateDecision = mySessionData.immediateDecision;
			} else {
				agreementBidList = new ArrayList<Bid>();
				opponentPrevMaxUtility = 0.0;
				opponentPrevMinUtility = 0.0;
				opponentMinUtility = 1.0;
				immediateDecision = false;
			}

			double maxConcessionUtility = (1.0
					- ((opponentMaxUtility - opponentPrevMinUtility)
							/ (1.0 - opponentPrevMinUtility)))
					* 100;

			// å�ˆæ„�ç‚¹ã�®è¿‘å‚�æŽ¢ç´¢
			if (!agreementBidList.isEmpty()) {
				double target = opponentMaxUtility > opponentPrevMaxUtility
						? opponentMaxUtility : opponentPrevMaxUtility;
				for (Bid bid : agreementBidList) {
					selectAgreementBidList = searchBid.searchOfferingBid(
							selectAgreementBidList, bid, target);
				}
			}

			// è­²æ­©è¨­å®š
			getAcceptProb.setTargetParam(this.sessionNr, maxConcessionUtility,
					opponentMinUtility);

		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private Bid isConsiderable() {
		Bid actionBid = null;
		double concessionDeg = (opponentMaxUtility - opponentMinUtility)
				/ (1.0 - opponentMinUtility);

		// åˆ�å›žã�¯è­²æ­©ã�—ã�ªã�„
		if (this.sessionNr < 1) {
			return actionBid;
		}

		// ************ è­²æ­©åˆ¤å®š ************//
		// 1.
		// ç›¸æ‰‹ã�ŒACCEPT_CONCESSIONä»¥ä¸Šè­²æ­©ã�—ã�¦ã��ã�Ÿå ´å�ˆ
		// 2.
		// å�³æ±ºåˆ¤æ–­ã�Œtrueã�‹ã�¤(ç›¸æ‰‹ã�®æœ€å¤§åŠ¹ç”¨å€¤)=(ç›¸æ‰‹ã�®é�ŽåŽ»ã�®æœ€å¤§åŠ¹ç”¨å€¤)ã�®å ´å�ˆ
		// 3.
		// æ™‚é–“ã�ŒDEAD_LINEã�‹ã�¤(ç›¸æ‰‹ã�®æœ€å¤§åŠ¹ç”¨å€¤)=(ç›¸æ‰‹ã�®é�ŽåŽ»ã�®æœ€å¤§åŠ¹ç”¨å€¤)ã�®å ´å�ˆ
		// 4.
		// DFã�Œæœ‰åŠ¹ã�‹ã�¤ç›¸æ‰‹ã�®æœ€è‰¯Bidã�ŒbidTargetä»¥ä¸Šã�«ã�ªã�£ã�Ÿå ´å�ˆ
		if (concessionDeg >= ACCEPT_CONCESSION) {
			actionBid = opponentMaxBid;
		} else if ((immediateDecision)
				&& (opponentMaxUtility >= opponentPrevMaxUtility)) {
			actionBid = opponentMaxBid;
		} else if (time > DEAD_LINE) {
			double standardUtil = opponentPrevMaxUtility < bidTarget
					? opponentPrevMaxUtility : bidTarget;
			if (opponentMaxUtility >= standardUtil) {
				actionBid = opponentMaxBid;
			}
		} else if (utilitySpace.getDiscountFactor() < 1.0) {
			if (opponentMaxUtility > bidTarget) {
				actionBid = opponentMaxBid;
			}
		}

		return actionBid;
	}

	private boolean isAcceptable(double t) throws Exception {
		boolean accept = false;
		double minUtil = myBidHistory.getWorstBidDetails()
				.getMyUndiscountedUtil();

		// ************ å�ˆæ„�åˆ¤å®š ************//
		// 1.
		// ç›¸æ‰‹ã�®åŠ¹ç”¨å€¤ã�Œè‡ªåˆ†ã�®é–¾å€¤ã‚’è¶…ã�ˆã�Ÿå ´å�ˆ
		// 2.
		// AcceptPã�Œä¸€å®šå€¤ã‚’è¶…ã�ˆã�Ÿå ´å�ˆ
		if (opponentLastUtility >= minUtil) {
			accept = true;
		} else if (acceptP > ACCEPT_PROB) {
			accept = true;
		}

		return accept;
	}

	private Action offerAndSaveBid(Bid actionBid) {
		Action action = new Offer(getAgentID(), actionBid);
		myBidHistory.add(convertBidDetails(actionBid, time));

		return action;
	}

	private BidDetails convertBidDetails(Bid bid, double time) {
		BidDetails bd = null;
		try {
			bd = new BidDetails(bid, utilitySpace.getUtility(bid), time);
		} catch (Exception e) {
			e.printStackTrace();
		}
		return bd;
	}

	private void addSelectBidList(double bidTarget) {
		Bid bid;
		double util;
		try {
			for (Iterator<Bid> it = selectAgreementBidList.iterator(); it
					.hasNext();) {
				bid = it.next();
				util = utilitySpace.getUtility(bid);
				if (util >= bidTarget) {
					selectOpponentBidList.add(bid);
					it.remove();
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	// FOR_DEBUG
	/*
	 * private void printCurrentDetails() {
	 * System.out.println("****CURRENT DETAILS****");
	 * System.out.println("bidTarget:" + bidTarget);
	 * System.out.println("estimateMax:" + getAcceptProb.getEstimateMax());
	 * System.out.println("opponentMax:" + opponentMaxUtility);
	 * System.out.println("opponentMin:" + opponentMinUtility); System.out
	 * .println("concessionDeg:" + ((opponentMaxUtility - opponentMinUtility) /
	 * (1.0 - opponentMinUtility)) 100 + "%"); System.out.println("prevMax:" +
	 * opponentPrevMaxUtility); System.out.println("prevMin:" +
	 * opponentPrevMinUtility); System.out.println("mySize:" +
	 * selectMyBidList.size()); System.out.println("opponentSize:" +
	 * selectOpponentBidList.size()); System.out.println("agreeSize:" +
	 * selectAgreementBidList.size()); System.out.println("immediateDecision:" +
	 * immediateDecision); }
	 */

	@Override
	public String getDescription() {
		return "ANAC2014 compatible with non-linear utility spaces";
	}
}
