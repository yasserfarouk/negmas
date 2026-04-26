/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package agents.anac.y2016.parscat;

import java.util.List;

import java.util.HashMap;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.BidHistory;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.EndNegotiationWithAnOffer;
import genius.core.actions.Offer;
import genius.core.bidding.BidDetails;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.IssueInteger;
import genius.core.issue.Value;
import genius.core.issue.ValueInteger;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;
import genius.core.timeline.TimeLineInfo;
import genius.core.utility.AbstractUtilitySpace;

/**
 *
 * @author Delaram
 */
public class ParsCat extends AbstractNegotiationParty {

	/**
	 * @param args
	 *            the command line arguments
	 */
	public static void main(String[] args) {
		// TODO code application logic here
	}

	private TimeLineInfo TimeLineInfo = null;
	private Bid maxBid = null;
	private AbstractUtilitySpace utilSpace = null;
	private BidHistory OtherAgentsBidHistory;
	private double tresh;
	private double t1 = 0;
	private double u2 = 1;

	public ParsCat() {
		this.OtherAgentsBidHistory = new BidHistory();
	}

	@Override
	public void init(NegotiationInfo info) {
		super.init(info);

		this.utilSpace = getUtilitySpace();// read utility space
		TimeLineInfo = timeline; // read time line info

		try {
			maxBid = utilSpace.getMaxUtilityBid();
		} catch (Exception ex) {
			Logger.getLogger(ParsCat.class.getName()).log(Level.SEVERE, null,
					ex);
		}

	}

	/**
	 * Each round this method gets called and ask you to accept or offer. The
	 * first party in the first round is a bit different, it can only propose an
	 * offer.
	 *
	 * @param validActions
	 *            Either a list containing both accept and offer or only offer.
	 * @return The chosen action.
	 */

	@Override
	public Action chooseAction(List<Class<? extends Action>> validActions) {
		Action action;
		try {
			if (OtherAgentsBidHistory.isEmpty()) {
				return new Offer(getPartyId(), maxBid);
			}
			action = new Offer(getPartyId(), getRandomBid());
			Bid myBid = ((Offer) action).getBid();
			double myOfferedUtil = getUtility(myBid);
			double time = TimeLineInfo.getTime();// get current time

			if (OtherAgentsBidHistory.getLastBid() == myBid) {
				return new Accept(getPartyId(),
						OtherAgentsBidHistory.getLastBid());
			} else {
				Bid OtherAgentBid = OtherAgentsBidHistory.getLastBid();
				double offeredUtilFromOpponent = getUtility(OtherAgentBid);
				if (isAcceptable(offeredUtilFromOpponent, myOfferedUtil,
						time)) {
					return new Accept(getPartyId(), OtherAgentBid);
				} else {
					return action;
				}

			}
		} catch (Exception e) {
			return new Offer(getPartyId(), maxBid);
		}
	}

	private boolean isAcceptable(double offeredUtilFromOtherAgent,
			double myOfferedUtil, double time) throws Exception {
		if (offeredUtilFromOtherAgent == myOfferedUtil)
			return true;
		double t = time;
		double Util = 1;
		if (time <= .25) {
			Util = 1 - t * 0.4;
		}
		if ((time > .25) && (time <= .375)) {
			Util = .9 + (t - .25) * 0.4;
		}
		if ((time > .375) && (time <= .5)) {
			Util = .95 - (t - .375) * 0.4;
		}
		if ((time > .5) && (time <= .6)) {
			Util = .9 - (t - .5);
		}
		if ((time > .6) && (time <= .7)) {
			Util = .8 + (t - .6) * 2;
		}
		if ((time > .7) && (time <= .8)) {
			Util = 1 - (t - .7) * 3;
		}
		if ((time > .8) && (time <= .9)) {
			Util = .7 + (t - 0.8) * 1;
		}
		if ((time > .9) && (time <= .95)) {
			Util = .8 - (t - .9) * 6;
		}
		if ((time > .95)) {
			Util = .5 + (t - .95) * 4;
		}
		if (Util > 1) {
			Util = .8;
		}
		// it will accept other agents offer if their offer is bigger than the
		// utility calculated
		return offeredUtilFromOtherAgent >= Util;
	}

	private Bid getRandomBid() throws Exception {
		HashMap<Integer, Value> values = new HashMap<>();
		List<Issue> issues = utilSpace.getDomain().getIssues();
		Random randomnr = new Random();
		Bid bid = null;
		double xxx = .001;
		long counter = 1000;
		double check = 0;
		while (counter == 1000) {
			counter = 0;
			do {
				for (Issue lIssue : issues) {
					switch (lIssue.getType()) {
					case DISCRETE:
						IssueDiscrete lIssueDiscrete = (IssueDiscrete) lIssue;
						int optionIndex = randomnr
								.nextInt(lIssueDiscrete.getNumberOfValues());
						values.put(lIssue.getNumber(),
								lIssueDiscrete.getValue(optionIndex));
						break;

					case INTEGER:
						IssueInteger lIssueInteger = (IssueInteger) lIssue;
						int optionIndex2 = lIssueInteger.getLowerBound()
								+ randomnr.nextInt(lIssueInteger.getUpperBound()
										- lIssueInteger.getLowerBound());
						values.put(lIssueInteger.getNumber(),
								new ValueInteger(optionIndex2));
						break;
					default:
						throw new Exception("issue type " + lIssue.getType()
								+ " not supported by SamantaAgent2");
					}
				}
				bid = new Bid(utilitySpace.getDomain(), values);

				if (t1 < .5) {
					tresh = 1 - t1 / 4;
					xxx = 0.01;
				}
				if ((t1 >= .5) && (t1 < .8)) {
					tresh = .9 - t1 / 5;
					xxx = .02;
				}
				if ((t1 >= .8) && (t1 < .9)) {
					tresh = .7 + t1 / 5;
					xxx = .02;
				}
				if ((t1 >= .9) && (t1 < .95)) {
					tresh = .8 + t1 / 5;
					xxx = .02;
				}
				if (t1 >= .95) {
					tresh = 1 - t1 / 4 - .01;
					xxx = .02;
				}
				if (t1 == 1) {
					tresh = .5;
					xxx = .05;
				}
				tresh = tresh - check;
				if (tresh > 1) {
					tresh = 1;
					xxx = .01;
				}
				if (tresh <= 0.5) {
					tresh = 0.49;
					xxx = .01;

				}
				counter++;
			} // check if the utility of the bid is in the correct interval if
				// not it will search again.
			while (((getUtility(bid) < (tresh - xxx))
					|| (getUtility(bid) > (tresh + xxx))) && (counter < 1000));
			check = check + .01;
		}
		// if the utility of my bid is smaller than the other Agent bid we will
		// send the best bid we get till that time
		// otherwise we will send our random bid
		if ((getUtility(bid) < getUtility(
				OtherAgentsBidHistory.getBestBidDetails().getBid()))
				&& getNumberOfParties() == 2)
			return OtherAgentsBidHistory.getBestBidDetails().getBid();

		return bid;
	}

	/**
	 * All offers proposed by the other parties will be received as a message.
	 * You can use this information to your advantage, for example to predict
	 * their utility.
	 *
	 * @param sender
	 *            The party that did the action. Can be null.
	 * @param action
	 *            The action that party did.
	 */

	@Override
	public void receiveMessage(AgentID sender, Action action) {
		super.receiveMessage(sender, action);
		// Here you hear other parties' messages
		if (action instanceof Offer) {
			Bid bid = ((Offer) action).getBid();
			try {
				BidDetails opponentBid = new BidDetails(bid,
						utilSpace.getUtility(bid), TimeLineInfo.getTime());
				u2 = utilSpace.getUtility(bid);
				t1 = TimeLineInfo.getTime();
				OtherAgentsBidHistory.add(opponentBid);

			} catch (Exception e) {
				EndNegotiationWithAnOffer end = new EndNegotiationWithAnOffer(
						this.getPartyId(), maxBid);
			}
		}
	}

	@Override
	public String getDescription() {
		return "ANAC2016";
	}

}
