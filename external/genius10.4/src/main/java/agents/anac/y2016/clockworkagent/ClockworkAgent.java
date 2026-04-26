package agents.anac.y2016.clockworkagent;

import java.util.List;

import java.util.ArrayList;
import java.util.Random;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.ActionWithBid;
import genius.core.actions.DefaultAction;
import genius.core.actions.Offer;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;

public class ClockworkAgent extends AbstractNegotiationParty {

	private double discountFactor = 0;
	private double reservationValue = 0;
	private double rad = 0;
	Random rand = new Random();
	private Bid[] candidateArray = new Bid[10];
	private ArrayList<Bid> history;
	private int Opcnt = 3, Ofcnt = 0;

	@Override
	public void init(NegotiationInfo info) {

		super.init(info);
		discountFactor = getUtilitySpace().getDiscountFactor(); // read discount factor
		System.out.println("Discount Factor is " + discountFactor);
		reservationValue = getUtilitySpace()
				.getReservationValueUndiscounted();
		System.out.println("Reservation Value is " + reservationValue);
		history = new ArrayList<Bid>();

	}

	@Override
	public Action chooseAction(List<Class<? extends Action>> validActions) {

		if (history.size() == 0) {
			return new Offer(getPartyId(), bidFunction());
		} else if (history.size() > 3 && checkAcceptance() == true) {

			return new Accept(getPartyId(),
					((ActionWithBid) getLastReceivedAction()).getBid());
		} else if (timeline.getTime() <= 0.95) {
			return new Offer(getPartyId(), bidFunction());
		} else if (timeline.getTime() <= 0.97 && timeline.getTime() > 0.95) {

			Bid tempBid = null;
			int c = 0;
			while (c < 30) {
				c++;
				tempBid = candidateArray[rand.nextInt(9)];
				if (tempBid != null)
					break;
			}

			if (tempBid != null) {
				return new Offer(getPartyId(), tempBid);
			} else {
				return new Offer(getPartyId(), bidFunction());
			}
		} else if (timeline.getTime() > 0.97) {
			return new Offer(getPartyId(), getMaxU());
		} else
			return new Offer(getPartyId(), bidFunction());
	}

	@Override
	public void receiveMessage(AgentID sender, Action action) {
		super.receiveMessage(sender, action);

		if (action instanceof Offer) {
			Bid opponentBid = DefaultAction.getBidFromAction(action);
			history.add(opponentBid);
		} else if (action instanceof Accept && getUtilityWithDiscount(
				history.get(history.size() - 1)) > uFunctionC()) {
			candidateArray[Opcnt] = history.get(history.size() - 1);
			Opcnt++;

			if (Opcnt == 9)
				Opcnt = 3;
		}

	}

	public Boolean checkAcceptance() {

		double tn, dt, ad = 0, t, botlim, uhist, avgT;
		t = timeline.getTime();
		tn = t + 0.01;

		if (history.size() > 3) {
			uhist = getUtilityWithDiscount(history.get(history.size() - 1));

			if (t <= 0.95) {
				if (checkConcession()) {
					ad = 0.05;
				}

				dt = Math.pow(discountFactor, tn) * (0.95 - ad);

				if (uhist > dt)
					return true;
				else
					return false;
			} else if (t > 0.95 && t <= 0.97) {
				if (checkConcession()) {
					ad = 0.1;
				}

				dt = Math.pow(discountFactor, tn);
				botlim = uBotlim(tn);

				avgT = ((dt + botlim) / 2) * (0.95 - ad);

				if (uhist > avgT)
					return true;
				else
					return false;
			} else if (t > 0.97) {
				if (checkConcession()) {
					ad = 0.12;
				}

				dt = Math.pow(discountFactor, tn);
				botlim = uBotlim(tn);

				avgT = ((dt + botlim) / 2) * (0.95 - ad);

				if (uhist > dt)
					return true;
				else
					return false;
			} else
				return false;
		} else
			return false;
	}

	public double uBotlim(double t) {
		double botlim, dt;

		dt = Math.pow(discountFactor, t);
		botlim = (1 - Math.pow(t, 2)) * (dt * (1 - reservationValue))
				+ dt * reservationValue;

		return botlim;
	}

	public double uFunction() {
		double botlim, avg, f, dt;

		botlim = uBotlim(timeline.getTime());

		dt = Math.pow(discountFactor, timeline.getTime());

		avg = (dt + botlim) / 2;

		if (checkConcession() == true && Ofcnt % 100 == 0) {
			f = Math.abs(Math.cos(rad)) * (avg - botlim) + botlim;
			Ofcnt = 0;
		} else
			f = Math.abs(Math.cos(rad)) * (dt - avg) + avg;

		rad += Math.PI / 4;
		Ofcnt++;
		if (rad % Math.PI == 0)
			rad = 0;

		return f;
	}

	public double uFunctionC() {
		double botlim, avg, f, dt;

		botlim = uBotlim(timeline.getTime());

		dt = Math.pow(discountFactor, timeline.getTime());

		f = 0.85 * (dt - botlim) + botlim;

		return f;
	}

	public Bid bidFunction() {
		Bid newBid = null;
		Bid bBid = null;
		int count = 0;
		double t = timeline.getTime();

		if (t < 0.3) {
			newBid = getMaxU();
		} else {
			newBid = generateRandomBid();
			double f = uFunction();

			while (count != 10000) {
				bBid = generateRandomBid();

				if (getUtilityWithDiscount(bBid) >= f
						&& getUtility(bBid) > getUtility(newBid))
					newBid = bBid;

				count++;
			}

		}
		return newBid;

	}

	public boolean checkConcession() {
		double h1, h2, h3;

		if (history.size() > 3) {
			h1 = getUtility(history.get(history.size() - 1));
			h2 = getUtility(history.get(history.size() - 2));
			h3 = getUtility(history.get(history.size() - 3));

			if (h1 > h2 || h1 > h3)
				return true;
			else
				return false;
		} else
			return false;
	}

	public Bid getMaxU() {
		Bid newBid = generateRandomBid();
		Bid tempBid = null;
		int count = 0;

		while (count != 13000) {
			tempBid = generateRandomBid();

			if (getUtility(tempBid) > getUtility(newBid)) {
				newBid = tempBid;
			}

			count++;
		}

		return newBid;
	}

	@Override
	public String getDescription() {
		return "ANAC2016 Considers time.";
	}
}
