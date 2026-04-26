package negotiator.boaframework.omstrategy;

import java.util.ArrayDeque;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import genius.core.bidding.BidDetails;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.OMStrategy;
import genius.core.boaframework.OpponentModel;

/**
 * Opponent Model Strategy
 */
public final class TheFawkes_OMS extends OMStrategy {
	private ArrayDeque<Double> lastTen;
	private int secondBestCounter = 1;

	@Override
	public void init(NegotiationSession nSession, OpponentModel oppModel, Map<String, Double> parameters) {
		initializeAgent(nSession, oppModel);
	}

	private void initializeAgent(NegotiationSession negotiationSession, OpponentModel model) {
		super.init(negotiationSession, model, new HashMap<String, Double>());
		this.lastTen = new ArrayDeque<Double>(11);
	}

	@Override
	public BidDetails getBid(List<BidDetails> list) { // gets as input a List
														// with bid that have an
														// utily u', given
														// formula 14
		Collections.sort(list, new Comparing(this.model));
		BidDetails opponentBestBid = list.get(0);
		boolean allEqual = true;

		for (double bid : this.lastTen) {
			if (bid != opponentBestBid.getMyUndiscountedUtil()) { // Use our own
																	// undiscounted
																	// util to
																	// check if
																	// we're
																	// effectively
																	// offering
																	// the same
																	// thing
				allEqual = false;
			}
		}
		if (allEqual) { // Offer the second best bid when we're offering the
						// same every time... does this work, and if so does it
						// need expansion?
			this.secondBestCounter++;
			if (list.size() > 1) {
				opponentBestBid = list.get(1);
			}
		}

		this.lastTen.addLast(opponentBestBid.getMyUndiscountedUtil());
		if (this.lastTen.size() > 10) {
			this.lastTen.removeFirst();
		}

		return opponentBestBid;
	}

	public int getSecondBestCount() {
		return this.secondBestCounter;
	}

	@Override
	public boolean canUpdateOM() {
		return true; // other variants mess up the whole code apparantley?!
	}

	private final static class Comparing implements Comparator<BidDetails> { // Sort
																				// according
																				// to
																				// what
																				// we
																				// think
																				// are
																				// the
																				// best
																				// bids
																				// for
																				// the
																				// opponent
																				// (best
																				// at
																				// the
																				// head
																				// of
																				// the
																				// list)
		private final OpponentModel model;

		protected Comparing(OpponentModel model) {
			this.model = model;
		}

		@Override
		public int compare(final BidDetails a, BidDetails b) {
			double evalA = this.model.getBidEvaluation(a.getBid());
			double evalB = this.model.getBidEvaluation(b.getBid());
			if (evalA < evalB) {
				return 1;
			} else if (evalA > evalB) {
				return -1;
			} else {
				return 0;
			}
		}
	}

	@Override
	public String getName() {
		return "TheFawkes";
	}
}
