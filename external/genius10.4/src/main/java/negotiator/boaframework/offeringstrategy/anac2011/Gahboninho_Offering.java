package negotiator.boaframework.offeringstrategy.anac2011;

import java.util.Map;

import genius.core.Bid;
import genius.core.bidding.BidDetails;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.NoModel;
import genius.core.boaframework.OMStrategy;
import genius.core.boaframework.OfferingStrategy;
import genius.core.boaframework.OpponentModel;
import genius.core.boaframework.SortedOutcomeSpace;
import negotiator.boaframework.opponentmodel.DefaultModel;
import negotiator.boaframework.sharedagentstate.anac2011.GahboninhoSAS;

/**
 * This is the decoupled Offering Strategy for Gahboninho (ANAC2011) The code
 * was taken from the ANAC2011 Gahboninho and adapted to work within the BOA
 * framework
 * 
 * DEFAULT OM: None
 * 
 * Decoupling Negotiating Agents to Explore the Space of Negotiation Strategies
 * T. Baarslag, K. Hindriks, M. Hendrikx, A. Dirkzwager, C.M. Jonker
 * 
 * @author Mark Hendrikx
 */
public class Gahboninho_Offering extends OfferingStrategy {
	final int PlayerCount = 8;
	private boolean WereBidsFiltered = false;
	private int RoundCount = 0;
	private SortedOutcomeSpace outcomespace;

	private int TotalFirstActions = 40;

	@Override
	public void init(NegotiationSession domainKnow, OpponentModel model, OMStrategy omStrategy,
			Map<String, Double> parameters) throws Exception {
		if (model instanceof DefaultModel) {
			model = new NoModel();
		}
		super.init(domainKnow, model, omStrategy, parameters);
		helper = new GahboninhoSAS(negotiationSession);
		if (!(opponentModel instanceof NoModel)) {
			outcomespace = new SortedOutcomeSpace(negotiationSession.getUtilitySpace());
		}
	}

	@Override
	public BidDetails determineOpeningBid() {
		return determineNextBid();
	}

	@Override
	public BidDetails determineNextBid() {

		BidDetails previousOpponentBid = null;
		BidDetails opponentBid = negotiationSession.getOpponentBidHistory().getLastBidDetails();
		int histSize = negotiationSession.getOpponentBidHistory().getHistory().size();
		if (histSize >= 2) {
			previousOpponentBid = negotiationSession.getOpponentBidHistory().getHistory().get(histSize - 1);
		}
		double threshold = -1;
		if (opponentBid != null) {

			if (previousOpponentBid != null) {
				try {
					((GahboninhoSAS) helper).getIssueManager().ProcessOpponentBid(opponentBid.getBid());
					((GahboninhoSAS) helper).getOpponentModel().UpdateImportance(opponentBid.getBid());
				} catch (Exception e) {
					e.printStackTrace();
				}
			} else {
				try {
					((GahboninhoSAS) helper).getIssueManager().learnBids(opponentBid.getBid());
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
			threshold = ((GahboninhoSAS) helper).getIssueManager().GetMinimumUtilityToAccept();
			((GahboninhoSAS) helper).getIssueManager().setMinimumUtilForAcceptance(threshold);
		}

		try {
			// on the first few rounds don't get tempted so fast

			++RoundCount;
			if (WereBidsFiltered == false && (negotiationSession
					.getTime() > ((GahboninhoSAS) helper).getIssueManager().GetDiscountFactor() * 0.9
					|| negotiationSession.getTime()
							+ 3 * ((GahboninhoSAS) helper).getIssueManager().getBidsCreationTime() > 1)) {
				WereBidsFiltered = true;

				int DesiredBidcount = (int) (RoundCount * (1 - negotiationSession.getTime()));

				if (((GahboninhoSAS) helper).getIssueManager().getBids().size() > 200) {
					((GahboninhoSAS) helper).getIssueManager().setBids(((GahboninhoSAS) helper).getOpponentModel()
							.FilterBids(((GahboninhoSAS) helper).getIssueManager().getBids(), DesiredBidcount));
				}
			}

		} catch (Exception e) {
			e.printStackTrace();
		}

		// on the first time we act offer max bid
		if (previousOpponentBid == null) {
			try {
				((GahboninhoSAS) helper).getIssueManager()
						.AddMyBidToStatistics(((GahboninhoSAS) helper).getIssueManager().getMaxBid());
			} catch (Exception e) {
				e.printStackTrace();
			}
			Bid maxBid = ((GahboninhoSAS) helper).getIssueManager().getMaxBid();

			try {
				return new BidDetails(maxBid, negotiationSession.getUtilitySpace().getUtility(maxBid),
						negotiationSession.getTime());
			} catch (Exception e) {
				e.printStackTrace();
			}
		}

		Bid myBid;
		if (((GahboninhoSAS) helper).getFirstActions() >= 0 && negotiationSession.getTime() < 0.15) {
			// on first few bids let the opponent learn some more about our
			// preferences

			double utilDecrease = (1 - 0.925) / TotalFirstActions;

			myBid = ((GahboninhoSAS) helper).getIssueManager()
					.GenerateBidWithAtleastUtilityOf(0.925 + utilDecrease * ((GahboninhoSAS) helper).getFirstActions());
			((GahboninhoSAS) helper).decrementFirstActions();
		} else {
			// always execute this one, even when an OM has been set as this
			// method has side-effects.
			myBid = ((GahboninhoSAS) helper).getIssueManager().GenerateBidWithAtleastUtilityOf(
					((GahboninhoSAS) helper).getIssueManager().GetNextRecommendedOfferUtility());
			if (((GahboninhoSAS) helper).getIssueManager().getInFrenzy() == true)
				myBid = ((GahboninhoSAS) helper).getIssueManager().getBestEverOpponentBid();

		}

		try {
			double utility = negotiationSession.getUtilitySpace().getUtility(myBid);
			if (!(opponentModel instanceof NoModel)) {
				BidDetails selectedBid = omStrategy.getBid(outcomespace, utility);
				((GahboninhoSAS) helper).getIssueManager().AddMyBidToStatistics(selectedBid.getBid());
				return selectedBid;
			}
			((GahboninhoSAS) helper).getIssueManager().AddMyBidToStatistics(myBid);
			return new BidDetails(myBid, utility, negotiationSession.getTime());
		} catch (Exception e) {
			e.printStackTrace();
		}
		return null;
	}

	@Override
	public String getName() {
		return "2011 - Gahboninho";
	}
}