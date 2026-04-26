package agents.anac.y2015.JonnyBlack;

import java.util.Collections;
import java.util.List;
import java.util.Vector;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.DefaultAction;
import genius.core.actions.Offer;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.ValueDiscrete;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;
import genius.core.utility.AdditiveUtilitySpace;

/**
 * This is your negotiation party.
 */
public class JonnyBlack extends AbstractNegotiationParty {

	/**
	 * Please keep this constructor. This is called by genius.
	 *
	 * @param utilitySpace
	 *            Your utility space.
	 * @param deadlines
	 *            The deadlines set for this negotiation.
	 * @param timeline
	 *            Value counting from 0 (start) to 1 (end).
	 * @param randomSeed
	 *            If you use any randomization, use this seed for it.
	 */
	double agreeVal = 1;
	Bid lastBid;
	int[] issueOrder;
	int[][] issueValOrder;
	public double finalStopVal = 0.6;
	Vector<BidHolder> acceptableBids;
	Vector<Party> parties = new Vector<Party>();
	int agentToFavor = 0;
	double care = 0.4;
	int lastorder = 0;
	int topNofOpp = 100;
	int round = 0;
	double unwillingness = 1.1;

	@Override
	public void init(NegotiationInfo info) {
		super.init(info);
		initializeCounts((AdditiveUtilitySpace) utilitySpace);
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

		// with 50% chance, counter offer
		// if we are the first party, also offer.
		round++;
		if (round % 10 == 0) {
			this.agreeVal = Functions.calcStopVal(parties, topNofOpp,
					(AdditiveUtilitySpace) utilitySpace);
			System.out.println(getPartyId());
			unwillingness *= .995;
			System.out.println(unwillingness);
			agreeVal *= this.unwillingness;
			if (agreeVal >= 1)
				agreeVal = 0.99;
			System.out.println("AGREE VAL = " + this.agreeVal);
			if (topNofOpp > 10) {
				topNofOpp -= 5;
			}
			System.out.println(topNofOpp);
		}
		double d = 0;
		if (lastBid != null)
			d = Functions.getBidValue((AdditiveUtilitySpace) utilitySpace,
					lastBid);
		// System.out.println(d);
		if (d >= this.stopValue())
			return new Accept(getPartyId(), lastBid);
		care *= 1.004;
		Bid b = createBid();
		if (parties.size() > 0) {
			agentToFavor++;
			agentToFavor = agentToFavor % parties.size();
		}
		return new Offer(getPartyId(), b);

	}

	/**
	 * All offers proposed by the other parties will be received as a message.
	 * You can use this information to your advantage, for example to predict
	 * their utility.
	 *
	 * @param sender
	 *            The party that did the action.
	 * @param action
	 *            The action that party did.
	 */
	@Override
	public void receiveMessage(AgentID sender, Action action) {
		super.receiveMessage(sender, action);
		if (action instanceof Accept) {
			// if(sender instanceof AbstractNegotiationParty)
			// {
			// String accepter=((AbstractNegotiationParty)
			// sender).getPartyId().toString();
			// Party p=new Party(accepter, null);
			// p = parties.get(parties.indexOf(p));
			// ArrayList<Issue> issues = lastBid.getIssues();
			// for(int i =0;i<issues.size();i++)
			// {
			// IssueDiscrete id = (IssueDiscrete)utilitySpace.getIssue(i);
			// int choice =
			// id.getValueIndex((ValueDiscrete)lastBid.getValue(i+1));
			// p.counts[i][choice]++;
			// }
			// p.calcWeights();
			// }
		} else {
			Bid b = DefaultAction.getBidFromAction(action);
			lastBid = b;
		}

		if (sender != null) {
			Party p = new Party(sender.toString(), issueValOrder);
			if (!parties.contains(p)) {
				parties.add(p);
				p.setOrderedBids(this.acceptableBids,
						(AdditiveUtilitySpace) utilitySpace);
			} else {
				p = parties.get(parties.indexOf(p));
			}
			if (lastBid != null
					&& !action.equals(new Accept(getPartyId(), lastBid))) {
				List<Issue> issues = lastBid.getIssues();
				try {
					for (int i = 0; i < issues.size(); i++) {
						IssueDiscrete id = (IssueDiscrete) ((AdditiveUtilitySpace) utilitySpace)
								.getIssue(i);
						int choice = id.getValueIndex(
								(ValueDiscrete) lastBid.getValue(i + 1));
						p.counts[i][choice]++;
					}
					p.calcWeights();
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
		}
		// Here you can listen to other parties' messages
	}

	public void initializeCounts(AdditiveUtilitySpace us) {
		this.issueOrder = Functions.calcOrderOfIssues(us);
		this.issueValOrder = Functions.calcOrderOfIssueVals(us);
		this.acceptableBids = getFeasibleBids();
		Collections.sort(acceptableBids);
		parties = new Vector<Party>();
	}

	double stopValue() {
		return agreeVal;
	}

	public Bid createBid() {
		// for(Party p :parties)
		// {
		// p.show();
		// }

		if (parties.size() > 0)
			for (int i = lastorder + 1; i < acceptableBids.size(); i++) {
				BidHolder bh = acceptableBids.get(i);
				if (bh.v > stopValue()
						&& parties.get(agentToFavor).getPredictedUtility(bh.b,
								(AdditiveUtilitySpace) utilitySpace) > care) {
					lastorder = i;
					return bh.b;
				}
				if (bh.v < stopValue())
					break;
			}

		lastorder = 0;
		return acceptableBids.get(0).b;
	}

	public Vector<BidHolder> getFeasibleBids() {
		Vector<BidHolder> bids = new Vector<BidHolder>();
		Bid b = Functions.getCopyOfBestBid((AdditiveUtilitySpace) utilitySpace);
		bids = recurseBids(b, bids, 0);
		System.out.println("Vector Size:" + bids.size());
		return bids;
	}

	public Vector<BidHolder> recurseBids(Bid b, Vector<BidHolder> v, int is) {
		Vector<BidHolder> v1 = new Vector<BidHolder>();
		if (is == issueOrder.length) {
			BidHolder bh = new BidHolder();
			bh.b = b;
			bh.v = Functions.getBidValue((AdditiveUtilitySpace) utilitySpace,
					b);
			v1.addElement(bh);

			return v1;
		}
		for (int i = 0; i < issueValOrder[issueOrder[is]].length; i++) {
			Bid b1 = new Bid(b);
			int issueID = issueOrder[is];
			int item = issueValOrder[issueID][i] - 1;
			ValueDiscrete val = Functions
					.getVal((AdditiveUtilitySpace) utilitySpace, issueID, item);
			b1 = b1.putValue(issueID + 1, val);
			if (Functions.getBidValue((AdditiveUtilitySpace) utilitySpace,
					b1) > this.finalStopVal) {
				v1.addAll(recurseBids(b1, v1, is + 1));
			}
		}
		return v1;
	}

	@Override
	public String getDescription() {
		return "ANAC2015";
	}
}
