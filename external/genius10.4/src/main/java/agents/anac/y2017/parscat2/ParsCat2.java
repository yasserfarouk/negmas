/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package agents.anac.y2017.parscat2;

import java.util.List;

import java.util.ArrayList;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.BidHistory;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.EndNegotiationWithAnOffer;
import genius.core.actions.Offer;
import genius.core.bidding.BidDetails;
import genius.core.list.Tuple;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;
import genius.core.persistent.PersistentDataType;
import genius.core.persistent.StandardInfo;
import genius.core.persistent.StandardInfoList;
import genius.core.timeline.TimeLineInfo;
import genius.core.utility.AbstractUtilitySpace;

/**
 *
 * @author Delaram - Agent Name: ParsCat2 - Team members: Delaram Javdani,
 *         Maedeh Najar, Faria Nassiri-Mofakham - Affiliation: University of
 *         Isfahan - Contact person: Faria Nassiri-Mofakham - Contact E-mail(s):
 *         djavdani1391@yahoo.com, nmaedeh@rocketmail.com,
 *         fnasirimofakham@yahoo.com
 */
public class ParsCat2 extends AbstractNegotiationParty {

	/**
	 * @param args
	 *            the command line arguments
	 */
	public static void main(String[] args) {
		// TODO code application logic here
	}

	private StandardInfoList history;
	private TimeLineInfo TimeLineInfo = null;
	private AbstractUtilitySpace utilSpace = null;
	private Bid maxBid = null;
	private Bid lastReceivedBid = null;
	private BidHistory currSessOppBidHistory;
	private BidHistory prevSessionsAgreementBidHistory;
	private int num = 0;
	private int num1 = 0;
	private List prevSessAvgUtilLis;

	public ParsCat2() {
		currSessOppBidHistory = new BidHistory();
		this.prevSessAvgUtilLis = new ArrayList();
	}

	@Override
	public void init(NegotiationInfo info) {
		super.init(info);
		this.utilSpace = getUtilitySpace();
		TimeLineInfo = timeline;
		try {
			maxBid = getUtilitySpace().getMaxUtilityBid();
		} catch (Exception ex) {
		}
		if (getData().getPersistentDataType() != PersistentDataType.STANDARD) {
			throw new IllegalStateException("need standard persistent data");
		}
		history = (StandardInfoList) getData().get();

		if (!history.isEmpty()) {
			agreementHistory();
		}
	}

	@Override
	public Action chooseAction(List<Class<? extends Action>> validActions) {
		if (lastReceivedBid != null) {
			try {
				Bid myBidByTime = generateRandomBidByTime();
				double myBidUtilByTime = getUtility(myBidByTime);
				double time = TimeLineInfo.getTime();
				if (isAcceptable(getUtility(lastReceivedBid), time,
						myBidUtilByTime))
					return new Accept(getPartyId(), lastReceivedBid);
				else
					return new Offer(getPartyId(), myBidByTime);
			} catch (Exception ex) {
			}
		}
		return new Offer(getPartyId(), maxBid);
	}

	private boolean isAcceptable(double offeredUtilFromOtherAgent, double time,
			double myBidUtilByTime) throws Exception {
		if (offeredUtilFromOtherAgent >= myBidUtilByTime)
			return true;
		double Util = 1;
		if (time <= .25)
			Util = 1 - time * 0.4;
		else if ((time > .25) && (time <= .375))
			Util = .9 + (time - .25) * 0.4;
		else if ((time > .375) && (time <= .5))
			Util = .95 - (time - .375) * 0.4;
		else if ((time > .5) && (time <= .6))
			Util = .9 - (time - .5);
		else if ((time > .6) && (time <= .7))
			Util = .8 + (time - .6) * 2;
		else if ((time > .7) && (time <= .8))
			Util = 1 - (time - .7) * 3;
		else if ((time > .8) && (time <= .9))
			Util = .7 + (time - 0.8) * 1;
		else if ((time > .9) && (time <= .95))
			Util = .8 - (time - .9) * 6;
		else if ((time > .95))
			Util = .5 + (time - .95) * 4;
		if (Util > 1)
			Util = .8;
		if (Util < 0)
			Util = 0;
		return offeredUtilFromOtherAgent >= Util;
	}

	private Bid generateRandomBidByTime() throws Exception {
		double width = .01;
		double thinner = 0;
		double time = TimeLineInfo.getTime();
		double tresh = 0;
		if (time < .5) {
			tresh = 1 - time / 4;
			width = 0.01;
		} else if ((time >= .5) && (time < .8)) {
			tresh = .9 - time / 5;
			width = .02;
		} else if ((time >= .8) && (time < .9)) {
			tresh = .7 + time / 5;
			width = .02;
		} else if ((time >= .9) && (time < .95)) {
			tresh = .8 + time / 5;
			width = .02;
		} else if (time >= .95) {
			tresh = 1 - time / 4 - .01;
			width = .02;
		} else if (time == 1) {
			tresh = .5;
			width = .05;
		}
		if (tresh > 1) {
			tresh = 1;
			width = .01;
		}
		if (tresh <= 0.5) {
			tresh = .49;
			width = .01;
		}
		if (!currSessOppBidHistory.isEmpty()
				&& tresh < getUtility(
						currSessOppBidHistory.getBestBidDetails().getBid())
				&& getNumberOfParties() == 2)
			tresh = getUtility(
					currSessOppBidHistory.getBestBidDetails().getBid());
		if (time > .25 && time < .3
				&& !prevSessionsAgreementBidHistory.isEmpty()
				&& tresh < getUtility(prevSessionsAgreementBidHistory
						.getBestBidDetails().getBid()))
			tresh = getUtility(prevSessionsAgreementBidHistory
					.getBestBidDetails().getBid());
		else if (time > .55 && time < .6
				&& !prevSessionsAgreementBidHistory.isEmpty()
				&& getUtility(prevSessionsAgreementBidHistory.getHistory()
						.get(num).getBid()) > .5) {
			tresh = getUtility(prevSessionsAgreementBidHistory.getHistory()
					.get(num).getBid());
			num++;
			if (num >= this.prevSessionsAgreementBidHistory.size())
				num = 0;
		} else if (time > .65 && time < .7
				&& !prevSessionsAgreementBidHistory.isEmpty()
				&& getUtility(prevSessionsAgreementBidHistory
						.getFirstBidDetails().getBid()) > .5)
			tresh = getUtility(prevSessionsAgreementBidHistory
					.getFirstBidDetails().getBid());
		else if (time > .75 && time < .8
				&& !prevSessionsAgreementBidHistory.isEmpty()
				&& prevSessionsAgreementBidHistory.getAverageUtility() > .5)
			tresh = prevSessionsAgreementBidHistory.getAverageUtility();
		else if (time > .85 && time < .9 && !prevSessAvgUtilLis.isEmpty()) {
			double util = (double) prevSessAvgUtilLis.get(num1);
			if (util > .4) {
				tresh = util;
				num1++;
				if (num1 >= prevSessAvgUtilLis.size())
					num1 = 0;
			}
		}
		if (tresh < 0)
			tresh = .4;
		for (int count = 0; count < 100; count++) {
			for (int counter = 0; counter < 1000; counter++) {
				Bid bid = generateRandomBid();
				if ((getUtility(bid) > (tresh - width))
						&& (getUtility(bid) < (tresh + width)))
					return bid;
			}
			thinner = thinner + .01;
			tresh = tresh - thinner;
			if (tresh < 0)
				return generateRandomBid();
		}
		return generateRandomBid();
	}

	@Override
	public void receiveMessage(AgentID sender, Action action) {
		super.receiveMessage(sender, action);
		if (action instanceof Offer) {
			lastReceivedBid = ((Offer) action).getBid();
			try {
				BidDetails opponentBid = new BidDetails(lastReceivedBid,
						getUtility(lastReceivedBid), TimeLineInfo.getTime());
				currSessOppBidHistory.add(opponentBid);
			} catch (Exception e) {
				EndNegotiationWithAnOffer endNegotiationWithAnOffer = new EndNegotiationWithAnOffer(
						this.getPartyId(), maxBid);
			}
		}
	}

	public void agreementHistory() {
		prevSessionsAgreementBidHistory = new BidHistory();
		int k = 0;
		if (history.size() > 10)
			k = history.size() - 11;
		for (int h = history.size() - 1; h >= k; h--) {
			StandardInfo lastinfo = history.get(h);
			Tuple<Bid, Double> agree = lastinfo.getAgreement();
			BidDetails opponentBid = new BidDetails(agree.get1(), agree.get2(),
					TimeLineInfo.getTime());
			prevSessionsAgreementBidHistory.add(opponentBid);
			double count = 0;
			double sum = 0;
			for (Tuple<String, Double> offered : lastinfo.getUtilities()) {
				Double util = offered.get2();
				count++;
				sum = sum + util;
			}
			if (count != 0) {
				double avg = sum / count;
				prevSessAvgUtilLis.add(avg);
			}
		}
	}

	@Override
	public String getDescription() {
		return "ANAC2017";
	}

}
