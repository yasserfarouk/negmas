package negotiator.boaframework.sharedagentstate.anac2010;

import java.util.List;

import genius.core.Bid;
import genius.core.bidding.BidDetails;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.SharedAgentState;
import genius.core.issue.Issue;
import genius.core.utility.AdditiveUtilitySpace;
import genius.core.utility.Evaluator;

/**
 * This is the shared code of the acceptance condition and bidding strategy of
 * ANAC 2010 Nozomi. The code was taken from the ANAC2010 Nozomi and adapted to
 * work within the BOA framework.
 * 
 * @author Alex Dirkzwager
 */
public class NozomiSAS extends SharedAgentState {

	private NegotiationSession negotiationSession;

	public double maxUtilityOfPartnerBid = 0;
	private boolean compromise = false;
	private boolean updateMaxPartnerUtility = false;
	private double prevAverageUtility = 0.0;
	private double averageUtility = 0.0;
	private double averagePartnerUtility = 0.0;
	private double prevAveragePartnerUtility = 0.0;
	private double minGap = 0.0;
	private double maxCompromiseUtility = 1.0;
	private double prevMaxCompromiseUtility = 1.0;
	public BidDetails maxUtilityPartnerBidDetails = null;
	private Bid restoreBid = null;
	private Bid prevRestoreBid = null;
	private int bidNumber;

	public NozomiSAS(NegotiationSession negoSession, BidDetails bid) {
		negotiationSession = negoSession;
		NAME = "Nozomi";

		BidDetails maxBid = negoSession.getMaxBidinDomain();
		maxCompromiseUtility = maxBid.getMyUndiscountedUtil() * 0.95;
		prevAverageUtility = maxBid.getMyUndiscountedUtil();
		minGap = (double) negotiationSession.getIssues().size();
		restoreBid = maxBid.getBid();
		prevRestoreBid = restoreBid;
	}

	public void checkCompromise(double time) {
		prevMaxCompromiseUtility = maxCompromiseUtility;
		averageUtility /= bidNumber;
		averagePartnerUtility /= bidNumber;

		double diff = prevAverageUtility - averageUtility;
		double partnerDiff = averagePartnerUtility - prevAveragePartnerUtility;
		if (compromise) {
			double gap = Math.abs(averageUtility - averagePartnerUtility);
			if (partnerDiff < diff * 0.90
					|| gap > Math.pow(1.0 - time / 100, 2.0) * 0.90) {
				double p1 = maxCompromiseUtility, p2 = maxCompromiseUtility;
				for (int attenuation = 95; attenuation <= 100; attenuation++) {
					if (maxCompromiseUtility * (double) attenuation / 100 > maxUtilityOfPartnerBid) {
						p1 = maxCompromiseUtility * (double) attenuation / 100;
						break;
					}
				}

				for (int attenuation = 10; attenuation < 1000; attenuation++) {
					if (maxCompromiseUtility - gap / attenuation > maxUtilityOfPartnerBid) {
						p2 = maxCompromiseUtility - gap / attenuation;
						break;
					}
				}
				maxCompromiseUtility = (p1 + p2) / 2;
				prevAverageUtility = averageUtility;
				compromise = false;
			}
		} else {
			if (partnerDiff > diff * 0.90
					|| (time > 50 && updateMaxPartnerUtility)) {
				prevAveragePartnerUtility = averagePartnerUtility;
				compromise = true;
			}
		}
		updateMaxPartnerUtility = false;
	}

	public int getBidNumber() {
		return bidNumber;
	}

	public void setBidNumber(int number) {
		bidNumber = number;
	}

	public void updateRestoreBid(Bid nextBid) {
		prevRestoreBid = restoreBid;
		List<Issue> issues = negotiationSession.getIssues();

		try {
		} catch (Exception e) {

		}
		double evalGap = 0.0;
		for (Issue issue : issues) {
			int issueID = issue.getNumber();
			Evaluator eval = ((AdditiveUtilitySpace) negotiationSession
					.getUtilitySpace()).getEvaluator(issueID);
			try {
				double evalGapissueID = Math.abs(eval.getEvaluation(
						(AdditiveUtilitySpace) negotiationSession
								.getUtilitySpace(), nextBid, issueID)
						- eval.getEvaluation(
								(AdditiveUtilitySpace) negotiationSession
										.getUtilitySpace(),
								getMaxUtilityPartnerBidDetails().getBid(),
								issueID));

				evalGap += evalGapissueID;
			} catch (Exception e) {
				evalGap += 1.0;
				e.printStackTrace();
			}
		}

		if (evalGap < minGap) {
			restoreBid = nextBid;
			minGap = evalGap;
		} else if (evalGap == minGap) {
			try {
				if (negotiationSession.getUtilitySpace().getUtility(nextBid) > negotiationSession
						.getUtilitySpace().getUtility(restoreBid)) {
					restoreBid = nextBid;
				}
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
	}

	public void setAverageUtility(double util) {
		averageUtility = util;
	}

	public double getAverageUtility() {
		return averageUtility;
	}

	public double getMaxUtilityOfPartnerBid() {
		return maxUtilityOfPartnerBid;
	}

	public void setMaxUtilityPartnerBidDetails(BidDetails b) {
		maxUtilityPartnerBidDetails = b;
		maxUtilityOfPartnerBid = b.getMyUndiscountedUtil();
	}

	public BidDetails getMaxUtilityPartnerBidDetails() {
		if (maxUtilityPartnerBidDetails == null) {
			return negotiationSession.getOpponentBidHistory()
					.getBestBidDetails();
		} else
			return maxUtilityPartnerBidDetails;
	}

	public void setUpdateMaxPartnerUtility(boolean updated) {
		updateMaxPartnerUtility = updated;
	}

	public Bid getRestoreBid() {
		return restoreBid;
	}

	public Bid getPrevRestoreBid() {
		return prevRestoreBid;
	}

	public void setRestoreBid(Bid bid) {
		prevRestoreBid = restoreBid;
		restoreBid = bid;
	}

	public void setPrevAverageUtility(double util) {
		prevAverageUtility = util;
	}

	public void setMaxCompromiseUtility(double util) {
		prevMaxCompromiseUtility = maxCompromiseUtility;
		maxCompromiseUtility = util;
	}

	public double getMaxCompromiseUtility() {
		return maxCompromiseUtility;
	}

	public double getPrevMaxCompromiseUtility() {
		return prevMaxCompromiseUtility;
	}

	public void setAveragePartnerUtility(double util) {
		averagePartnerUtility = util;
	}

	public double getAveragePartnerUtility() {
		return averagePartnerUtility;
	}
}