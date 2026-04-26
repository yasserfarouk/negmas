package agents.anac.y2016.caduceus.agents.Caduceus;

import java.util.List;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Random;

import java.math.RoundingMode;
import java.text.DecimalFormat;

import agents.anac.y2016.caduceus.agents.Caduceus.sanity.Pair;
import agents.anac.y2016.caduceus.agents.Caduceus.sanity.SaneBid;
import agents.anac.y2016.caduceus.agents.Caduceus.sanity.SaneIssue;
import agents.anac.y2016.caduceus.agents.Caduceus.sanity.SaneUtilitySpace;
import agents.anac.y2016.caduceus.agents.Caduceus.sanity.SaneValue;
import genius.core.AgentID;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.ActionWithBid;
import genius.core.actions.Offer;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.IssueInteger;
import genius.core.issue.IssueReal;
import genius.core.issue.Value;
import genius.core.issue.ValueInteger;
import genius.core.issue.ValueReal;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;
import genius.core.utility.AdditiveUtilitySpace;

/**
 * This is your negotiation party.
 */
public class Caduceus extends AbstractNegotiationParty {

	/**
	 * Each round this method gets called and ask you to accept or offer. The
	 * first party in the first round is a bit different, it can only propose an
	 * offer.
	 *
	 * @param validActions
	 *            Either a list containing both accept and offer or only offer.
	 * @return The chosen action.
	 */

	public double discountFactor = 0; // if you want to keep the discount factor
	private int numberOfOpponents;
	private double selfReservationValue = 0.75;
	private double percentageOfOfferingBestBid = 0.83;

	private SaneUtilitySpace mySaneUtilitySpace = null;
	private OpponentProfiles opponentProfiles = new OpponentProfiles();
	private Bid previousBid;
	private boolean takeConcessionStep = true;

	private HashMap<AgentID, Opponent> opponentMap = new HashMap<AgentID, Opponent>();

	@Override
	public void init(NegotiationInfo info) {

		super.init(info);

		discountFactor = getUtilitySpace().getDiscountFactor(); // read
																		// discount
																		// factor
		double reservationValue = getUtilitySpace().getReservationValueUndiscounted();

		System.out.println("Discount Factor is " + discountFactor);
		System.out.println("Reservation Value is " + reservationValue);

		numberOfOpponents = this.getNumberOfParties() - 1;
		selfReservationValue = Math.max(selfReservationValue, reservationValue);
		percentageOfOfferingBestBid = percentageOfOfferingBestBid * discountFactor;
	}

	@Override
	public Action chooseAction(List<Class<? extends Action>> validActions) {
		if (isBestOfferTime()) {
			Bid bestBid = this.getBestBid();
			if (bestBid != null)
				return new Offer(getPartyId(), bestBid);
			else
				System.err.println("Best Bid is null?");
		} else {
			Bid bid = getMyBestOfferForEveryone(this.getTimeLine().getTime());
			if (bid != null) {

				System.out.println("Offered bid util:" + this.getUtilityWithDiscount(previousBid));
				System.out.println("My good offer for everyone (considering nash product):" + this.getUtility(bid));

				if (this.getUtilityWithDiscount(bid) < selfReservationValue) {
					System.out.println("I will not play nice!");
					bid = this.getRandomBid();
					if (bid == null) {
						System.err.println("Failed to generate bid with getRandomBid()");
					}

				}

				if (this.getUtilityWithDiscount(previousBid) > (this.getUtilityWithDiscount(bid)) + 0.2) {
					System.out.println("Accept!");
					return new Accept(getPartyId(), ((ActionWithBid) getLastReceivedAction()).getBid());
				}

			} else {
				System.err.println("Failed to generate bid with getMyBestOfferForEveryone!");
			}

			return new Offer(getPartyId(), bid); // conceding to Nash
		}

		return new Offer(getPartyId(), getBestBid());
	}

	private Bid getMyBestOfferForEveryone(double time) {
		ArrayList<SaneUtilitySpace> utilitySpaces = new ArrayList<>();
		utilitySpaces.add(this.getMySaneUtilitySpace());

		for (Map.Entry<AgentID, Opponent> entry : this.opponentProfiles.getOpponentProfiles().entrySet()) {
			utilitySpaces.add(entry.getValue().saneUtilitySpace);
		}
		NashProductCalculator npc = null;
		try {
			npc = new NashProductCalculator(utilitySpaces);
			npc.calculate(this.utilitySpace);
			if (npc.nashBid == null) {
				Bid bestBid = this.getBestBid();
				CounterOfferGenerator offerGenerator = new CounterOfferGenerator(bestBid, this);
				return offerGenerator.generateBid(time);
			}
		} catch (Exception e) {
			System.out.println("Nash failed to calculate utility spaces");
			e.printStackTrace();
			return null;
		}
		CounterOfferGenerator cog = new CounterOfferGenerator(npc.nashBid, this);
		try {
			return cog.generateBid(time);
		} catch (Exception e) {
			e.printStackTrace();
			return null;
		}

	}

	private boolean isBestOfferTime() {
		return this.getTimeLine().getCurrentTime() < (this.getTimeLine().getTotalTime() * percentageOfOfferingBestBid);
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

		if (sender != null) {
			if (action instanceof Offer) {

				Opponent opponentProfile;
				Bid uglyBid = ((Offer) action).getBid();

				SaneBid saneBid = new SaneBid(uglyBid, this.getMySaneUtilitySpace());
				this.getMySaneUtilitySpace();

				if (this.opponentProfiles.getOpponentProfiles().get(sender) == null)
					opponentProfile = new Opponent(sender, this.utilitySpace);
				else
					opponentProfile = this.opponentProfiles.getOpponentProfiles().get(sender);

				previousBid = uglyBid;
				Bid previousBid = null;

				if (!opponentProfile.getBidHistory().isEmpty())
					previousBid = opponentProfile.getBidHistory().get(opponentProfile.getBidHistory().size() - 1);

				Iterator<Pair<SaneIssue, SaneValue>> iterator = saneBid.getIterator();

				while (iterator.hasNext()) {
					Pair<SaneIssue, SaneValue> p = iterator.next();
					SaneIssue issue = p.first;
					SaneValue value = p.second;

					SaneUtilitySpace.IssueSpace issueSpace = opponentProfile.saneUtilitySpace.saneSpaceMap
							.get(issue.name);

					issueSpace.findValue(value.name).utility += getRoundValue();

					if (previousBid != null) {
						for (Map.Entry<Integer, Value> previousBidEntry : previousBid.getValues().entrySet()) {
							Value previousBidValue = previousBidEntry.getValue();
							if (previousBidValue.toString().equalsIgnoreCase(value.name)) {
								SaneIssue correspondingIssue = opponentProfile.saneUtilitySpace.saneIssueMap
										.get(issue.name);
								correspondingIssue.weight += getRoundValue();
							}
						}
					}
				}
				opponentProfile.addToHistory(uglyBid);
				this.opponentProfiles.getOpponentProfiles().put(sender, opponentProfile);
			}
		}
	}

	@Override
	public String getDescription() {
		return "Caduceus";
	}

	private Bid getBestBid() {
		try {
			return this.utilitySpace.getMaxUtilityBid();
		} catch (Exception e) {
			e.printStackTrace();
		}
		return null;
	}

	public SaneUtilitySpace getMySaneUtilitySpace() {
		if (mySaneUtilitySpace == null) {
			mySaneUtilitySpace = new SaneUtilitySpace();
			mySaneUtilitySpace.initWithCopy((AdditiveUtilitySpace) this.utilitySpace);
		}

		return mySaneUtilitySpace;
	}

	private double getRoundValue() {

		double roundValue = (2 * Math.pow(this.getTimeLine().getTime(), 2)) - (101 * this.getTimeLine().getTime())
				+ 100;

		DecimalFormat decimalFormat = new DecimalFormat("#.###");
		decimalFormat.setRoundingMode(RoundingMode.CEILING);

		return Double.parseDouble(decimalFormat.format(roundValue));

	}

	private Bid getRandomBid() {
		HashMap<Integer, Value> values = new HashMap<Integer, Value>(); // pairs
		// <issuenumber,chosen
		// value
		// string>
		List<Issue> issues = utilitySpace.getDomain().getIssues();
		Random randomnr = new Random();

		// create a random bid with utility>MINIMUM_BID_UTIL.
		// note that this may never succeed if you set MINIMUM too high!!!
		// in that case we will search for a bid till the time is up (3 minutes)
		// but this is just a simple agent.
		Bid bid = null;
		do {
			for (Issue lIssue : issues) {
				switch (lIssue.getType()) {
				case DISCRETE:
					IssueDiscrete lIssueDiscrete = (IssueDiscrete) lIssue;
					int optionIndex = randomnr.nextInt(lIssueDiscrete.getNumberOfValues());
					values.put(lIssue.getNumber(), lIssueDiscrete.getValue(optionIndex));
					break;
				case REAL:
					IssueReal lIssueReal = (IssueReal) lIssue;
					int optionInd = randomnr.nextInt(lIssueReal.getNumberOfDiscretizationSteps() - 1);
					values.put(lIssueReal.getNumber(),
							new ValueReal(lIssueReal.getLowerBound()
									+ (lIssueReal.getUpperBound() - lIssueReal.getLowerBound()) * (double) (optionInd)
											/ (double) (lIssueReal.getNumberOfDiscretizationSteps())));
					break;
				case INTEGER:
					IssueInteger lIssueInteger = (IssueInteger) lIssue;
					int optionIndex2 = lIssueInteger.getLowerBound()
							+ randomnr.nextInt(lIssueInteger.getUpperBound() - lIssueInteger.getLowerBound());
					values.put(lIssueInteger.getNumber(), new ValueInteger(optionIndex2));
					break;
				default:
					return null;
				}
			}
			bid = new Bid(utilitySpace.getDomain(), values);
		} while (getUtility(bid) < selfReservationValue);

		return bid;
	}
}
