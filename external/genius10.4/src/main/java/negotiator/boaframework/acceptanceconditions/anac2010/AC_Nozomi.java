package negotiator.boaframework.acceptanceconditions.anac2010;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import genius.core.BidHistory;
import genius.core.bidding.BidDetails;
import genius.core.boaframework.AcceptanceStrategy;
import genius.core.boaframework.Actions;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.OfferingStrategy;
import genius.core.boaframework.OpponentModel;
import genius.core.issue.Issue;
import genius.core.issue.IssueReal;
import genius.core.issue.ValueReal;
import genius.core.utility.AdditiveUtilitySpace;
import genius.core.utility.Evaluator;
import negotiator.boaframework.sharedagentstate.anac2010.NozomiSAS;

/**
 * This is the decoupled Acceptance Conditions for Nozomi (ANAC2010). The code
 * was taken from the ANAC2010 Nozomi and adapted to work within the BOA
 * framework.
 * 
 * Decoupling Negotiating Agents to Explore the Space of Negotiation Strategies
 * T. Baarslag, K. Hindriks, M. Hendrikx, A. Dirkzwager, C.M. Jonker
 * 
 * @author Alex Dirkzwager, Mark Hendrikx
 */
public class AC_Nozomi extends AcceptanceStrategy {

	private double maxUtilityOfPartnerBid = 0.0;
	private int measureT = 1;
	private int continuousKeep = 0;
	private BidDetails maxBid;

	/**
	 * Empty constructor for the BOA framework.
	 */
	public AC_Nozomi() {
	}

	public AC_Nozomi(NegotiationSession negoSession, OfferingStrategy strat) throws Exception {
		init(negoSession, strat, null, null);
	}

	@Override
	public void init(NegotiationSession negoSession, OfferingStrategy strat, OpponentModel opponentModel,
			Map<String, Double> parameters) throws Exception {
		this.negotiationSession = negoSession;
		offeringStrategy = strat;

		// checking if offeringStrategy helper is a BRAMAgentHelper
		if (offeringStrategy.getHelper() == null || (!offeringStrategy.getHelper().getName().equals("Nozomi"))) {
			helper = new NozomiSAS(negotiationSession, offeringStrategy.getNextBid());
		} else {
			helper = (NozomiSAS) offeringStrategy.getHelper();
		}

		maxBid = negotiationSession.getMaxBidinDomain();
		try {
			((NozomiSAS) helper).setRestoreBid(negotiationSession.getUtilitySpace().getMaxUtilityBid());
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	@Override
	public Actions determineAcceptability() {
		try {
			BidDetails prevBid = negotiationSession.getOwnBidHistory().getLastBidDetails();
			BidDetails partnerBid = negotiationSession.getOpponentBidHistory().getLastBidDetails();
			double prevUtility;
			if (!negotiationSession.getOwnBidHistory().getHistory().isEmpty()) {
				prevUtility = negotiationSession.getOwnBidHistory().getLastBidDetails().getMyUndiscountedUtil();
			} else {
				prevUtility = 1;
			}
			double offeredUtil = negotiationSession.getOpponentBidHistory().getLastBidDetails().getMyUndiscountedUtil();
			double time = negotiationSession.getTime();

			ArrayList<BidDetails> list = (ArrayList<BidDetails>) negotiationSession.getOpponentBidHistory()
					.getHistory();
			BidHistory historyList = new BidHistory((ArrayList<BidDetails>) list.clone());
			BidDetails lastBid = negotiationSession.getOpponentBidHistory().getLastBidDetails();
			historyList.getHistory().remove(lastBid);

			if (!historyList.getHistory().isEmpty())
				maxUtilityOfPartnerBid = historyList.getBestBidDetails().getMyUndiscountedUtil();

			// this comes from other method in Nozomi (not in isAccept)
			if (maxUtilityOfPartnerBid > ((NozomiSAS) helper).getMaxCompromiseUtility()) {
				((NozomiSAS) helper).setMaxCompromiseUtility(maxUtilityOfPartnerBid);
			}

			if (((NozomiSAS) helper).getMaxCompromiseUtility() > maxBid.getMyUndiscountedUtil() * 0.95) {
				((NozomiSAS) helper).setMaxCompromiseUtility(maxBid.getMyUndiscountedUtil() * 0.95);
			}

			if (continuousKeep <= 0) {
				if (helper != offeringStrategy.getHelper()) {
					((NozomiSAS) helper).updateRestoreBid(offeringStrategy.getNextBid().getBid());
				}
			}

			if (time > (double) measureT * 3.0) {
				if (helper != offeringStrategy.getHelper()) {
					((NozomiSAS) helper).checkCompromise(time);
				}
				measureT++;
			}

			if ((negotiationSession.getOwnBidHistory().getLastBidDetails() != null)
					&& offeringStrategy.getNextBid().getMyUndiscountedUtil() == negotiationSession.getOwnBidHistory()
							.getLastBidDetails().getMyUndiscountedUtil()) {
				continuousKeep++;
			} else {
				continuousKeep = 0;
			}

			if (offeredUtil > maxUtilityOfPartnerBid) {
				((NozomiSAS) helper).setMaxUtilityPartnerBidDetails(partnerBid);
				maxUtilityOfPartnerBid = offeredUtil;
				((NozomiSAS) helper).setUpdateMaxPartnerUtility(true);
			}

			if (offeredUtil > maxBid.getMyUndiscountedUtil() * 0.95 || offeredUtil >= prevUtility) {
				return Actions.Accept;

			} else {
				List<Issue> issues = negotiationSession.getIssues();

				double evalGap = 0.0;
				for (Issue issue : issues) {
					int issueID = issue.getNumber();

					switch (issue.getType()) {
					case REAL:
						IssueReal issueReal = (IssueReal) issue;
						ValueReal valueReal = (ValueReal) prevBid.getBid().getValue(issueID);
						ValueReal partnerValueReal = (ValueReal) partnerBid.getBid().getValue(issueID);
						double gap = Math.abs(valueReal.getValue() - partnerValueReal.getValue());
						if (gap > (issueReal.getUpperBound() - issueReal.getLowerBound()) / 100) {
							evalGap += 0.5;
						}
						break;
					default:
						if ((prevBid != null)
								&& !prevBid.getBid().getValue(issueID).equals(partnerBid.getBid().getValue(issueID))) {
							evalGap += 0.5;
						}
						break;
					}

					Evaluator eval = ((AdditiveUtilitySpace) negotiationSession.getUtilitySpace())
							.getEvaluator(issueID);

					try {
						evalGap += Math.abs(eval.getEvaluation(
								(AdditiveUtilitySpace) negotiationSession.getUtilitySpace(), prevBid.getBid(), issueID)
								- eval.getEvaluation((AdditiveUtilitySpace) negotiationSession.getUtilitySpace(),
										partnerBid.getBid(), issueID));
					} catch (Exception e) {
						evalGap += 1.0;
					}

					evalGap += 0.5;
				}

				evalGap /= 2.0 * (double) issues.size();
				if (time < 0.50) {
					if (negotiationSession.getOpponentBidHistory().getLastBidDetails() != null
							&& offeredUtil > ((NozomiSAS) helper).getMaxCompromiseUtility()
							&& offeredUtil >= maxUtilityOfPartnerBid) {
						double acceptCoefficient = -0.1 * time + 1.0;
						// double sameIssueRatio = (double)sameIssueNumber /
						// (double)issues.size();
						if (evalGap < 0.30 && offeredUtil > prevUtility * acceptCoefficient) {
							return Actions.Accept;

						}
					}
				} else if (time < 0.80) {
					if (negotiationSession.getOpponentBidHistory().getLastBidDetails() != null
							&& offeredUtil > ((NozomiSAS) helper).getMaxCompromiseUtility() * 0.95
							&& offeredUtil > maxUtilityOfPartnerBid * 0.95) {
						double diffMyBidAndOffered = prevUtility - offeredUtil;
						// double ratioOfferedToMaxUtility = offeredUtil /
						// maxUtility;
						// double ratioOfferedToMaxOffered = offeredUtil /
						// maxUtilityOfPartnerBid;

						if (diffMyBidAndOffered < 0.0)
							diffMyBidAndOffered = 0.0;

						double acceptCoefficient = -0.16 * (time - 0.50) + 0.95;
						// double sameIssueRatio = (double)sameIssueNumber /
						// (double)issues.size();
						if (evalGap < 0.35 && offeredUtil > prevUtility * acceptCoefficient) {
							return Actions.Accept;
						}
					}
				} else {

					if (negotiationSession.getOpponentBidHistory().getLastBidDetails() != null
							&& offeredUtil > ((NozomiSAS) helper).getPrevMaxCompromiseUtility() * 0.90
							&& offeredUtil > maxUtilityOfPartnerBid * 0.95) {

						double restoreEvalGap = 0.0;
						for (Issue issue : issues) {
							int issueID = issue.getNumber();

							switch (issue.getType()) {
							case REAL:
								IssueReal issueReal = (IssueReal) issue;
								ValueReal valueReal = (ValueReal) ((NozomiSAS) helper).getPrevRestoreBid()
										.getValue(issueID);
								ValueReal partnerValueReal = (ValueReal) partnerBid.getBid().getValue(issueID);
								double gap = Math.abs(valueReal.getValue() - partnerValueReal.getValue());
								if (gap > (issueReal.getUpperBound() - issueReal.getLowerBound()) / 100) {
									restoreEvalGap += 0.5;
								}
								break;
							default:
								if (!((NozomiSAS) helper).getPrevRestoreBid().getValue(issueID)
										.equals(partnerBid.getBid().getValue(issueID))) {
									restoreEvalGap += 0.5;
								}
								break;
							}

							Evaluator eval = ((AdditiveUtilitySpace) negotiationSession.getUtilitySpace())
									.getEvaluator(issueID);

							try {
								restoreEvalGap += Math.abs(
										eval.getEvaluation((AdditiveUtilitySpace) negotiationSession.getUtilitySpace(),
												prevBid.getBid(), issueID)
												- eval.getEvaluation(
														(AdditiveUtilitySpace) negotiationSession.getUtilitySpace(),
														partnerBid.getBid(), issueID));
							} catch (Exception e) {
								restoreEvalGap += 1.0;
							}

							restoreEvalGap += 0.5;
						}
						restoreEvalGap /= 2.0 * (double) issues.size();

						double threshold = 0.40;

						if (time > 0.90) {
							threshold = 0.50;
						}

						if (restoreEvalGap <= threshold || evalGap <= threshold) {

							return Actions.Accept;
						}
					}
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
			return Actions.Reject;
		}
		return Actions.Reject;
	}

	@Override
	public String getName() {
		return "2010 - Nozomi";
	}
}