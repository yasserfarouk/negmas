package agents.anac.y2017.group3;

import java.util.List;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.Offer;
import genius.core.issue.Value;
import genius.core.list.Tuple;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;
import genius.core.persistent.PersistentDataType;
import genius.core.persistent.StandardInfo;
import genius.core.persistent.StandardInfoList;

/**
 * Sample party that accepts the Nth offer, where N is the number of sessions
 * this [agent-profile] already did.
 */
public class Group3 extends AbstractNegotiationParty {

	private Bid lastReceivedBid = null;
	private Bid currentBid = null;
	private boolean isCorrectionAppliedBecauseOfHistory = false;
	private StandardInfoList history;
	private boolean historyAnalyzed = false;
	private boolean counted = true;
	private ArrayList<String> agentNames = new ArrayList<String>();
	ArrayList<BidData> bids = new ArrayList<BidData>();
	Random random = new Random();
	static private double MIN_ACCEPTABLE_UTILITY = 0.7;

	// init() will be called before starting negotiation
	@Override
	public void init(NegotiationInfo info) {

		super.init(info);

		System.out.println("Discount Factor is "
				+ getUtilitySpace().getDiscountFactor());
		System.out.println("Reservation Value is "
				+ getUtilitySpace().getReservationValueUndiscounted());

		if (getData().getPersistentDataType() != PersistentDataType.STANDARD) {
			throw new IllegalStateException("need standard persistent data");
		}

		// use history to get previous negotiation utilities
		history = (StandardInfoList) getData().get();

	}

	@Override
	public Action chooseAction(List<Class<? extends Action>> validActions) {
		// if lastReceivedBid is null --> you are starter party
		// zamanin yarisina kadar max bid teklif ediliyor.
		// bu arada analyze edicek bid de birikmis oluyor.
		// kalan yarisinda analize gore bid sunuluyor.
		try {
			System.err.println(getUtility(lastReceivedBid));
			if (getUtility(lastReceivedBid) > 0.85
					&& timeline.getTime() < 0.5) {
				return new Accept(getPartyId(), lastReceivedBid);
			} else if (getUtility(lastReceivedBid) > 0.75
					&& timeline.getTime() < 0.8) {
				return new Accept(getPartyId(), lastReceivedBid);
			} else if (getUtility(lastReceivedBid) > MIN_ACCEPTABLE_UTILITY
					&& timeline.getTime() < 0.9) {
				return new Accept(getPartyId(), lastReceivedBid);
			} else if (timeline.getTime() > 0.95) {
				return new Accept(getPartyId(), lastReceivedBid);
			} else if (timeline.getTime() < 0.2) {
				Bid a = this.generateRandomBid();
				while (this.getUtility(a) < 0.8) {
					a = this.generateRandomBid();
				}
				return new Offer(getPartyId(), a);
			} else {
				currentBid = analyzeBids();
				return new Offer(getPartyId(), currentBid);
			}
		} catch (Exception e) {
			System.err.println(e);
		}
		Bid a = this.generateRandomBid();
		while (this.getUtility(a) < 0.8) {
			a = this.generateRandomBid();
		}
		return new Offer(getPartyId(), a);

	}

	private Bid analyzeBids() throws Exception {
		// burada biriktirilen biddler kontrol edilip bir strateji uygulanacak
		// Eger onceki historylerden birinde bekledigimizin ustunde bir bid
		// kabul edilmisse onu minimumla degistiriyoruz
		// cunku demekki olabilir
		System.err.println("min utility " + MIN_ACCEPTABLE_UTILITY);
		if (!isCorrectionAppliedBecauseOfHistory) {
			System.out.println("HISTORY SIZE   >>>> " + history.size());
			for (StandardInfo si : history) {
				System.out.println("AGREEMENT: " + si.getAgreement());
				System.out
						.println("UTILITIES: " + si.getUtilities().toString());
				if (si.getAgreement().get2() >= MIN_ACCEPTABLE_UTILITY)
					MIN_ACCEPTABLE_UTILITY = si.getAgreement().get2();
				else {
					// eger 0 sa sonuca ulasilamamis burada biz ustumuze duseni
					// yapip bir sonuc elde edilebilsin diye
					// kabul edebilecegimiz degeri dusuruyoruz. bu friendly
					// approachimizi destekliyor.
					// eger 0 degilde sadece mevcut minimumdan kucukse bu
					// durumda belli ki son anda kabul ettigimiz bir offer
					// o halde minimumu biraz dusurup son anda kabul edecegimiz
					// daha kotu offer ihtimallerini azaltmis oluyoruz
					// 0.5tense -> mevcut minimum utility-0.02 daha iyidir
					MIN_ACCEPTABLE_UTILITY -= 0.02;
					break;
				}
			}
			isCorrectionAppliedBecauseOfHistory = true;
		}
		HashMap<Integer, Value> values = new HashMap<Integer, Value>();
		for (int i = 0; i < lastReceivedBid.getIssues().size(); i++) {
			int max = 0;
			Value value = null;
			for (int j = 0; j < agentNames.size(); j++) {
				// burdan elde ettigim commoni kendimle karsilastiricam
				if (findCurrentBidData(agentNames.get(j)).assumedValues
						.get(i) > max) {
					max = findCurrentBidData(agentNames.get(j)).assumedValues
							.get(i);
					value = findCurrentBidData(agentNames.get(j)).bid
							.getValue(i + 1);
				}
			}
			values.put(i + 1, value);
		}
		Bid bid = new Bid(utilitySpace.getDomain(), values);
		if (getUtility(bid) > 0.9)
			return bid;
		else {
			return manupalateAssumedValues(bid);
		}
	}

	private Bid manupalateAssumedValues(Bid bid) throws Exception {
		// TODO Auto-generated method stub
		HashMap<Integer, Value> values = bid.getValues();
		while (getUtility(bid) < 0.75) {
			int issueNum = random
					.nextInt(lastReceivedBid.getIssues().size() - 1) + 1;
			values.put(issueNum,
					utilitySpace.getMaxUtilityBid().getValue(issueNum));
			bid = new Bid(utilitySpace.getDomain(), values);
		}
		System.out.println("My offer: " + getUtility(bid));
		return bid;
	}

	private void countAgentData(AgentID sender, Action action) {
		if (agentNames.contains(sender.getName())) {
			counted = false;
			return;
		} else {
			agentNames.add(sender.getName());
			if (action instanceof Offer) {
				BidData bd = new BidData(sender.getName(),
						((Offer) action).getBid());
				for (int i = 0; i < ((Offer) action).getBid().getIssues()
						.size(); i++) {
					bd.assumedValues.add(i, 0);
				}
				bids.add(bd);
				return;
			} else {
				BidData bd = new BidData(sender.getName(),
						((Accept) action).getBid());
				for (int i = 0; i < ((Accept) action).getBid().getIssues()
						.size(); i++) {
					bd.assumedValues.add(i, 0);
				}
				bids.add(bd);
				return;
			}

		}
	}

	/*
	 * Order: init() --> receiveMessage() ---> chooseAction()
	 */
	@Override
	public void receiveMessage(AgentID sender, Action action) {
		super.receiveMessage(sender, action);
		if (counted && sender != null) {
			countAgentData(sender, action);
		}
		try {
			if (action instanceof Offer && counted) {
				lastReceivedBid = ((Accept) action).getBid();
				BidData bidData = new BidData(sender.getName(),
						((Offer) action).getBid());
				bids.add(bidData);
			}
			if (action instanceof Offer) {
				lastReceivedBid = ((Offer) action).getBid();
				BidData currentBidData = findCurrentBidData(sender.getName());
				if (currentBidData != null) {
					for (int i = 0; i < lastReceivedBid.getIssues()
							.size(); i++) {
						if (currentBidData.bid.getValue(i + 1)
								.equals(lastReceivedBid.getValue(i + 1))) {
							currentBidData.assumedValues.set(i,
									(currentBidData.assumedValues.get(i) + 1));
						}
					}
				}
			}
			if (action instanceof Accept) {
				lastReceivedBid = ((Accept) action).getBid();
				BidData currentBidData = findCurrentBidData(sender.getName());
				if (currentBidData != null) {
					for (int i = 0; i < lastReceivedBid.getIssues()
							.size(); i++) {
						if (currentBidData.bid.getValue(i + 1)
								.equals(lastReceivedBid.getValue(i + 1))) {
							currentBidData.assumedValues.set(i,
									(currentBidData.assumedValues.get(i) + 1));
						}
					}
				}
			}

			if (!history.isEmpty() && historyAnalyzed == false) {
				analyzeHistory();
				isCorrectionAppliedBecauseOfHistory = false;
			}

		} catch (Exception e) {
			System.err.println(e);
		}
	}

	private BidData findCurrentBidData(String sender) {
		for (BidData bid : bids) {
			if (bid.agentName.equals(sender)) {
				return bid;
			}
		}
		return null;
	}

	public void analyzeHistory() {
		historyAnalyzed = true;
		// from recent to older history records
		for (int h = history.size() - 1; h >= 0; h--) {

			System.out.println("History index: " + h);

			StandardInfo lastinfo = history.get(h);
			System.out.println("historyNo: " + h + " myID: " + getPartyId());
			int counter = 0;
			for (Tuple<String, Double> offered : lastinfo.getUtilities()) {
				counter++;

				String party = offered.get1(); // get partyID -> example:
												// ConcederParty@15
				Double util = offered.get2(); // get the offer utility

				System.out.println(
						"PartyID: " + party + " utilityForMe: " + util);
				System.out.println();
				// just print first 3 bids, not the whole history
				if (counter == 3)
					break;
			}
			System.out.println("\n");

		}

	}

	@Override
	public String getDescription() {
		return "ANAC2017";
	}

}