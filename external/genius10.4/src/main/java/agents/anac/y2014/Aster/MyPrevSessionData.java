package agents.anac.y2014.Aster;

import java.io.Serializable;
import java.util.ArrayList;

import genius.core.Bid;
import genius.core.BidHistory;
import genius.core.NegotiationResult;


public class MyPrevSessionData implements Serializable {
	private static final long serialVersionUID = 1L;

	ArrayList<Bid> agreementBidList;
	BidHistory opponentBidHistory;
	Boolean isAgreement;
	Bid lastBid;
	double opponentPrevMaxUtility;
	double opponentPrevMinUtility;
	double endNegotiationTime;
	boolean immediateDecision;

	public MyPrevSessionData(ArrayList<Bid> agreementBidList, BidHistory opponentBidHistory, NegotiationResult result, double opponentPrevMaxUtility, double opponentPrevMinUtility, double endNegotiationTime, boolean immediateDecision) {
		this.agreementBidList = agreementBidList;
		this.opponentBidHistory = opponentBidHistory;
		this.isAgreement = result.isAgreement();
		this.lastBid = result.getLastBid();
		this.opponentPrevMaxUtility = opponentPrevMaxUtility;
		this.opponentPrevMinUtility = opponentPrevMinUtility;
		this.endNegotiationTime = endNegotiationTime;
		this.immediateDecision = immediateDecision;
	}
}
