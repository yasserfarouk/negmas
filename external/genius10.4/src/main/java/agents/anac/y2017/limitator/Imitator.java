package agents.anac.y2017.limitator;

import java.util.List;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.Offer;
import genius.core.issue.Issue;
import genius.core.issue.Value;
import genius.core.list.Tuple;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;
import genius.core.persistent.PersistentDataType;
import genius.core.persistent.StandardInfo;
import genius.core.persistent.StandardInfoList;

class Opponent {
	private String ID;
	String Model;
	ArrayList<Double> utils = new ArrayList<Double>();
	HashMap<Issue, HashMap<Value, Integer>> bids_tab = new HashMap<Issue, HashMap<Value, Integer>>();
	HashMap<String, ArrayList<Double>> my_history = new HashMap<String, ArrayList<Double>>();

	public Opponent(String name) {
		// Class constructor
		ID = name;
	}

	public String your_name() {
		return ID;
	}
}

/*
 * Sample party that accepts the Nth offer, where N is the number of sessions
 * this [agent-profile] already did.
 */
// **********************************************************************
public class Imitator extends AbstractNegotiationParty {
	private List<Opponent> Participants_List = new ArrayList<Opponent>();
	private Bid lastReceivedBid, mybid, max_bid = null;
	private Double res_val = 0.70;
	private int nrChosenActions = 0; // number of times chosenAction was called.
	private StandardInfoList history;
	private String my_name = "";
	static List<String> Agents_List = new ArrayList<String>();
	static int roundCount = 1;
	static List<Opponent> opponents = new ArrayList<Opponent>();

	// ************************************************************
	private static Boolean look_up(List<Opponent> party_list,
			String party_name) {
		Boolean res = false;
		for (int y = 0; y <= party_list.size() - 1; y++) {
			if (party_list.get(y).your_name().equals(party_name))
				res = true;

		}
		return res;
	}

	// *************************************************************************
	private static int get_opponent_index(List<Opponent> party_list,
			String party_name) {
		int res = 0;
		for (int y = 0; y <= party_list.size() - 1; y++) {
			if (party_list.get(y).your_name().equals(party_name))
				res = y;
		}
		return res;
	}

	// *************************************************************************
	public Bid bid_generator() {
		Bid sample_bid = generateRandomBid();
		Bid sample_bid2 = getMaxUtilityBid();
		Integer const_1 = 0;
		Integer const_2 = 0;
		ArrayList<HashMap<Issue, HashMap<Value, Integer>>> agents_bids_dataframe = new ArrayList<HashMap<Issue, HashMap<Value, Integer>>>();
		for (Opponent opp : Participants_List)// Collecting bids and frequency
												// of issues for all opponents
												// in a list
		{
			agents_bids_dataframe.add(opp.bids_tab);
		}
		for (Issue issue : sample_bid.getIssues()) {
			for (int cnt = 0; cnt <= agents_bids_dataframe.size() - 2; cnt++) {
				Integer tmp_freq_1 = agents_bids_dataframe.get(cnt).get(issue)
						.values().iterator().next();
				const_1 += tmp_freq_1;
				Integer tmp_freq_2 = agents_bids_dataframe.get(cnt + 1)
						.get(issue).values().iterator().next();
				const_2 += tmp_freq_2;
				Value tmp_val_1 = agents_bids_dataframe.get(cnt).get(issue)
						.keySet().iterator().next();
				Value tmp_val_2 = agents_bids_dataframe.get(cnt + 1).get(issue)
						.keySet().iterator().next();
				// Choose the issue which has the max frequency between the
				// issues of all other agents issues and consistency for
				// more than quarter of time passed till now
				if ((tmp_freq_1 > tmp_freq_2) && (tmp_freq_1 > (Math
						.floor((getTimeLine().getCurrentTime()) / 2)))) {
					sample_bid = sample_bid.putValue(issue.getNumber(),
							tmp_val_1);
				}
				// Choose the issue which has the max frequency between the
				// issues of all other agents issues and consistency for
				// more than quarter of time passed till now
				else if ((tmp_freq_1 < tmp_freq_2) && (tmp_freq_2 > (Math
						.floor((getTimeLine().getCurrentTime()) / 2)))) {
					sample_bid = sample_bid.putValue(issue.getNumber(),
							tmp_val_2);
				} else if ((tmp_freq_1 == tmp_freq_2)
						&& (tmp_freq_2 > (Math
								.floor((getTimeLine().getCurrentTime()) / 2)))
						&& (tmp_freq_1 > (Math.floor(
								(getTimeLine().getCurrentTime()) / 2)))) {
					if (const_1 > const_2)// which one is more stubborn?
						sample_bid = sample_bid.putValue(issue.getNumber(),
								tmp_val_1);
					else if (const_1 < const_2)// which one is more stubborn?
						sample_bid = sample_bid.putValue(issue.getNumber(),
								tmp_val_2);
					else if (const_1 == const_2)// both equal stubborn then put
												// my maximum utility issue :)
						sample_bid = sample_bid.putValue(issue.getNumber(),
								sample_bid2.getValue(issue.getNumber()));
					else
						sample_bid = sample_bid.putValue(issue.getNumber(),
								sample_bid2.getValue(issue.getNumber()));
				}
				// if none of agents are stubborn by above conditions then
				// choose the value that has
				// maximum value for me
				else {
					sample_bid = sample_bid.putValue(issue.getNumber(),
							sample_bid2.getValue(issue.getNumber()));
				}
			}
		}
		return sample_bid;
	}

	// *************************************************************************
	public Bid getMaxUtilityBid() {
		try {
			max_bid = utilitySpace.getMaxUtilityBid();
			return max_bid;
		} catch (Exception e) {
			max_bid = bid_generator();
			return max_bid;
		}
	}

	// ***********************************************************************
	@Override
	// public void init(AbstractUtilitySpace utilSpace, Deadline dl,
	// TimeLineInfo tl, long randomSeed, AgentID agentId,
	// PersistentDataContainer data)
	public void init(NegotiationInfo info)

	{
		// super.init(utilSpace, dl, tl, randomSeed, agentId, data);
		// if (getData().getPersistentDataType() != PersistentDataType.STANDARD)
		// {
		// throw new IllegalStateException("need standard persistent data");
		// }
		// roundCount++;
		super.init(info);

		System.out.println("Discount Factor is "
				+ getUtilitySpace().getDiscountFactor());
		System.out.println("Reservation Value is "
				+ getUtilitySpace().getReservationValueUndiscounted());

		if (getData().getPersistentDataType() != PersistentDataType.STANDARD) {
			throw new IllegalStateException("need standard persistent data");
		}
		history = (StandardInfoList) getData().get();

		if (!history.isEmpty()) {
			// example of using the history. Compute for each party the maximum
			// utility of the bids in last session.
			Map<String, Double> maxutils = new HashMap<String, Double>();
			StandardInfo lastinfo = history.get(history.size() - 1);
			for (Tuple<String, Double> offered : lastinfo.getUtilities()) {
				String party = offered.get1();
				Double util = offered.get2();
				maxutils.put(party, maxutils.containsKey(party)
						? Math.max(maxutils.get(party), util) : util);
			}
			System.out.println(maxutils); // notice tournament suppresses all
											// output.
		}
	}

	// ************************************************************************
	@Override
	public Action chooseAction(List<Class<? extends Action>> validActions) {
		double increase_rate = 0;
		increase_rate = Math.floor(Math.floor(res_val / 100) * Math.floor(Math
				.abs(getTimeLine().getTotalTime()
						- (getTimeLine().getTime()) * 100)
				/ getTimeLine().getTotalTime()));
		nrChosenActions++;
		if ((lastReceivedBid != null)) {
			if (utilitySpace
					.getUtility(lastReceivedBid) >= (res_val + increase_rate)) {
				return new Accept(getPartyId(), lastReceivedBid);
			} else {
				mybid = bid_generator();
				return new Offer(getPartyId(), mybid);
			}
		} else {
			mybid = bid_generator();
			return new Offer(getPartyId(), mybid);
		}
	}

	// =======================================================
	@Override
	public void receiveMessage(AgentID sender, Action action) {
		int in = 0;
		super.receiveMessage(sender, action);
		if (action instanceof Offer) {// if they offer some bid
			lastReceivedBid = ((Offer) action).getBid();// take the bid
			String partyName = ((Offer) action).getAgent().toString();// who is
																		// the
																		// agent
																		// that
																		// offers
																		// the
																		// bid
			my_name = partyName.substring(0, partyName.indexOf("@"));// extract
																		// the
																		// name
			if (look_up(Participants_List, my_name) == false)// if there is no
																// such agent in
																// list of
																// agents then
																// create one
			{
				Opponent party = new Opponent(my_name);
				Participants_List.add(party);
			}
			in = get_opponent_index(Participants_List, my_name);
			for (Issue issue : ((Offer) action).getBid().getIssues())// each
																		// issue
																		// e.g.
																		// Food,
																		// Drinks,
																		// Location,
																		// . . .
			{
				if (!(Participants_List.get(in).bids_tab.containsKey(issue)))// if
																				// there
																				// is
																				// no
																				// such
																				// issue
																				// then
																				// create
																				// issue
				{
					HashMap<Value, Integer> issue_hash = new HashMap<Value, Integer>();
					Value key_value = ((Offer) action).getBid()
							.getValue(issue.getNumber());// ingredient
															// of
															// issue
					issue_hash.put(key_value, 1);// put issue and it's initial
													// value in the list
					Participants_List.get(in).bids_tab.put(issue, issue_hash);
				} else {
					HashMap<Value, Integer> issue_hash = new HashMap<Value, Integer>();// if
																						// issue
																						// exists
																						// then
																						// look
																						// to
																						// values
					Value key_value = ((Offer) action).getBid()
							.getValue(issue.getNumber());// ingredient
															// of
															// issue
					if (!(Participants_List.get(in).bids_tab.get(issue).keySet()
							.contains(key_value))) {
						issue_hash.put(key_value, 1);// then create one with
														// initial value 1
						Participants_List.get(in).bids_tab.put(issue,
								issue_hash);
					} else {
						// if the value exists already then increase the
						// frequency
						issue_hash.put(key_value,
								(Participants_List.get(in).bids_tab.get(issue)
										.get(key_value).intValue() + 1));
						Participants_List.get(in).bids_tab.put(issue,
								issue_hash);
					}
				}
			}
		}
	}

	// ===================================================
	@Override
	public String getDescription() {
		return "ANAC2017";
	}
}