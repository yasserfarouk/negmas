package agents.anac.y2013.MetaAgent.portfolio.AgentLG;

import genius.core.Agent;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.Offer;
import genius.core.utility.AdditiveUtilitySpace;

/**
 * The Agent is most of the time stubborn but in the end it starts to
 * compromise. In the beginning it votes in the beginning its best votes. when
 * the time is 0-0.6 : it votes incrementally from its best utility to a minimum
 * of 0.75* utility and learns opponent utility. when the time is 0.6-0.9: it
 * starts to compromise and chooses better bid for the opponents that still good
 * for it only if the opponent is also comprising (the rate of compromise
 * depends on the opponent and on the discount factor). when the time over 0.9:
 * the agents starts to "panic" and compromise more frequently. when the time is
 * over(more then 0.9995 ) it chooses opponents max utility bid for it. The
 * Agent accept an offer only if it offered before by the agent or the time is
 * almost over and the bid is close enough to its worst offer.
 * 
 * @author Luba Golosman
 */
public class AgentLG extends Agent {

	private Bid myLastBid = null;
	private Bid oponnetLastBid = null;

	private BidChooser bidChooser = null;
	private OpponentBids oppenentsBid;
	private boolean bidLast = false;

	@Override
	public Action chooseAction() {

		Action currentAction = null;
		try {
			double time = timeline.getTime();

			// first bid -> vote the optimal bid
			if (myLastBid == null || oponnetLastBid == null) {
				myLastBid = this.utilitySpace.getMaxUtilityBid();
				currentAction = new Offer(getAgentID(), myLastBid);
			} else {
				double opponentUtility = this.utilitySpace
						.getUtilityWithDiscount(oponnetLastBid, time);
				double myUtility = this.utilitySpace.getUtilityWithDiscount(
						myLastBid, time);

				// accept if opponent offer is good enough or there is no time
				// and the offer is 'good'
				if (opponentUtility >= myUtility * 0.99
						|| (time > 0.999 && opponentUtility >= myUtility * 0.9)
						|| bidChooser.getMyBidsMinUtility(time) <= opponentUtility) {
					currentAction = new Accept(getAgentID(), oponnetLastBid);
				} else if (bidLast) {
					currentAction = new Offer(getAgentID(), myLastBid);
				}
				// there is lot of time ->learn the opponent and bid the 1/4
				// most optimal bids
				else if (time < 0.6) {
					currentAction = bidChooser.getNextOptimicalBid(time);
				} else {
					// the time is over -> bid the opponents max utility bid for
					// me
					if (time >= 0.9995) {
						myLastBid = oppenentsBid.getMaxUtilityBidForMe();
						if (utilitySpace
								.getUtilityWithDiscount(myLastBid, time) < utilitySpace
								.getReservationValueWithDiscount(time))
							myLastBid = bidChooser.getMyminBidfromBids();
						currentAction = new Offer(getAgentID(), myLastBid);
					} else {
						// Comprise and chose better bid for the opponents that
						// still good for me
						currentAction = bidChooser.getNextBid(time);
					}
				}
			}
			if (currentAction instanceof Offer) {
				myLastBid = ((Offer) currentAction).getBid();
				if (oppenentsBid.getOpponentsBids().contains(myLastBid))
					bidLast = true;
			}

		} catch (Exception e) {
			// System.out.println("Error: " + e);
			currentAction = new Accept(getAgentID(), oponnetLastBid);
		}

		return currentAction;
	}

	@Override
	public String getName() {
		return "AgentLG";
	}

	public void init() {
		oppenentsBid = new OpponentBids(utilitySpace);
		bidChooser = new BidChooser((AdditiveUtilitySpace) this.utilitySpace,
				getAgentID(), oppenentsBid);
	}

	public void ReceiveMessage(Action opponentAction) {
		if (opponentAction instanceof Offer) {
			oponnetLastBid = ((Offer) opponentAction).getBid();

			try {
				oppenentsBid.addBid(((Offer) opponentAction).getBid());
				oppenentsBid.getOpponentBidUtility(
						this.utilitySpace.getDomain(), oponnetLastBid);
			} catch (Exception e) {
				e.printStackTrace();
			}
		}

	}

	@Override
	public String getVersion() {
		return "1.1";
	}

}
