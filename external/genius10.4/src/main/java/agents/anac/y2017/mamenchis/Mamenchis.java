package agents.anac.y2017.mamenchis;

import java.util.List;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.Offer;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.IssueInteger;
import genius.core.issue.Value;
import genius.core.issue.ValueInteger;
import genius.core.list.Tuple;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;
import genius.core.persistent.PersistentDataType;
import genius.core.persistent.StandardInfo;
import genius.core.persistent.StandardInfoList;
import genius.core.utility.AbstractUtilitySpace;

/**
 * Sample party that accepts the Nth offer, where N is the number of sessions
 * this [agent-profile] already did.
 */

public class Mamenchis extends AbstractNegotiationParty {

	private Bid lastReceivedBid = null;
	private int nrChosenActions;
	private StandardInfoList history;
	private int numOfIssues;
	private int numOfParties;
	private double reservationValue;
	private double initialUpper;
	private double initialExponent;
	private double upper;
	private double lower;
	private double exponent;
	private double acceptUtil;
	private double utility;
	private List<Bid>[] proposedBids;
	private int[] numOfBids;
	private Map<Integer, Map<Value, Integer>>[] valueFrequences;
	private Map<Integer, Integer>[] issueChangeFrequences;
	private double[] maxUtils;
	private double[] minUtils;
	private LinkedList<Bid>[] proposedMaxBid;
	private LinkedList<Bid>[] proposedMinBid;
	private List<Bid>[] topBidsInOppsSpace;
	private LinkedList<Bid>[] topBidsInMySpace;
	private Bid myBid;
	private List<Bid> fullUtilBids;
	private int myIndex;
	private String oppName;
	private int oppIndex;
	private boolean[] acceptAction;
	private PreferModel[] pModel;
	private Bid lastConcessionPoint;
	private List<Bid>[] acceptableBidsWithOppValues;
	private List<Bid> candidateBids;
	private List<Bid> lastPeriodBidList;
	private int myNextPartyIndex;
	private int myLastPartyIndex;
	private List<Bid>[] maxProductBids;
	private List<Bid> maxUtilAdditionBidByOpps;
	private double maxUtilAdditionByOpps;
	private double time1;
	private double time2;
	private double time3;
	private boolean hasHistory;
	private boolean hasHistWithRightPosition;
	private boolean hasInitParamByHist;
	private List<Tuple<Bid, Double>> agreements;
	private boolean hasAgreementHist;
	private List<Bid> maxAgreedBids;
	private Map<String, List<Double>>[] histUtils;
	private Map<String, Double>[] maxUtilsInHist;
	private List<Bid>[] maxUtilBidsByHist;
	private List<Bid> sameBidsInOpps;

	@Override
	public void init(NegotiationInfo info) {

		super.init(info);

		if (getData().getPersistentDataType() != PersistentDataType.STANDARD) {
			throw new IllegalStateException("need standard persistent data");
		}
		history = (StandardInfoList) getData().get();

		if (!history.isEmpty()) {
			Map<String, Double> maxutils = new HashMap<String, Double>();
			StandardInfo lastinfo = history.get(history.size() - 1);
			for (Tuple<String, Double> offered : lastinfo.getUtilities()) {
				String party = offered.get1();
				Double util = offered.get2();
				maxutils.put(party, maxutils.containsKey(party)
						? Math.max(maxutils.get(party), util) : util);
			}
		}
		hasAgreementHist = false;
		if (!history.isEmpty()) {
			hasHistory = true;
			hasInitParamByHist = false;
			agreements = new ArrayList<>();
			maxAgreedBids = new ArrayList<>();
			int numOfAgent = history.get(0).getAgentProfiles().size();
			histUtils = new HashMap[numOfAgent];
			maxUtilsInHist = new HashMap[numOfAgent];
			maxUtilBidsByHist = new ArrayList[numOfAgent];
			for (int i = 0; i < numOfAgent; i++) {
				histUtils[i] = new HashMap<>();
				maxUtilsInHist[i] = new HashMap<>();
				maxUtilBidsByHist[i] = new ArrayList<>();
			}
			for (StandardInfo sessionInfo : history) {
				if (sessionInfo.getAgreement().get1() != null) {
					agreements.add(sessionInfo.getAgreement());
					hasAgreementHist = true;
				}
				for (Tuple<String, Double> offered : sessionInfo
						.getUtilities()) {
					String party = offered.get1();
					double util = offered.get2();
					int index = Integer.parseInt(party.split("@")[1])
							% numOfAgent;
					if (!histUtils[index].containsKey(party)) {
						List<Double> utils = new ArrayList<>();
						utils.add(util);
						histUtils[index].put(party, utils);
					} else {
						histUtils[index].get(party).add(util);
					}
				}
			}
			maxAgreedBids = getMaxBidsInAgreements();
			for (int partyIndex = 0; partyIndex < numOfAgent; partyIndex++) {
				for (String party : histUtils[partyIndex].keySet()) {
					String partyName = party.split("@")[0];
					List<Double> utils = histUtils[partyIndex].get(party);
					double maxUtil = getMaxInArray(utils);
					if (!maxUtilsInHist[partyIndex].containsKey(partyName)) {
						maxUtilsInHist[partyIndex].put(partyName, maxUtil);
					} else if (maxUtilsInHist[partyIndex]
							.get(partyName) < maxUtil) {
						maxUtilsInHist[partyIndex].put(partyName, maxUtil);
					}
				}
			}
		} else {
			hasHistory = false;
			hasHistWithRightPosition = false;
		}

		nrChosenActions = 0;
		numOfIssues = getUtilitySpace().getDomain().getIssues().size();
		fullUtilBids = new ArrayList<>();
		fullUtilBids = getFullUtilBid(info.getUtilitySpace());
		reservationValue = getUtilitySpace().getReservationValue();
		sameBidsInOpps = new ArrayList<>();
		maxUtilAdditionBidByOpps = new ArrayList<>();
		lastPeriodBidList = new ArrayList<>();
		maxUtilAdditionByOpps = 0.0;
		initialUpper = 0.9;
		initialExponent = 20;
		upper = initialUpper;
		lower = reservationValue;
		exponent = initialExponent;
		time1 = 0.9;
		time2 = 0.99;
		time3 = 0.995;
		hasInitParamByHist = false;
	}

	@Override
	public Action chooseAction(List<Class<? extends Action>> validActions) {
		nrChosenActions++;
		Action action = null;
		Bid bid = null;
		acceptUtil = getAccept();
		if (lastReceivedBid != null) {
			if (utility >= acceptUtil
					&& !hasProposedMyLastRound(lastReceivedBid)) {
				bid = new Bid(lastReceivedBid);
				action = new Accept(getPartyId(), bid);
			} else {
				if (timeline.getTime() > time3) {
					if (hasHistory) {
						if (!maxAgreedBids.isEmpty()) {
							bid = maxAgreedBids.get(
									nrChosenActions % maxAgreedBids.size());
							action = new Offer(getPartyId(), bid);
						} else {
							if (!lastPeriodBidList.isEmpty()) {
								bid = lastPeriodBidList.get(nrChosenActions
										% lastPeriodBidList.size());
								action = new Offer(getPartyId(), bid);
							} else {
								bid = lastReceivedBid;
								action = new Accept(getPartyId(), bid);
								return action;
							}
						}
					} else {
						if (!lastPeriodBidList.isEmpty()) {
							bid = lastPeriodBidList.get(
									nrChosenActions % lastPeriodBidList.size());
						} else {
							bid = lastReceivedBid;
							action = new Accept(getPartyId(), bid);
							myBid = bid;
							return action;
						}
					}
					action = new Offer(getPartyId(), bid);
					myBid = bid;
				} else if (timeline.getTime() > time2) {
					if (nrChosenActions % 2 == 1) {
						if (maxProductBids[myIndex].size() <= 0) {
							maxProductBids[myIndex] = getMaxSocialWelfareBidsInList(
									proposedBids[myIndex]);
						}
						if (maxProductBids[myIndex].size() > 0) {
							Bid nBid = maxProductBids[myIndex]
									.get(nrChosenActions
											% maxProductBids[myIndex].size());
							if (hasProposedMyLastRound(nBid)) {
								bid = proposedBids[myIndex].get(
										nrChosenActions % numOfBids[myIndex]);
							} else {
								bid = nBid;
							}
						} else {
							bid = proposedBids[myIndex]
									.get(nrChosenActions % numOfBids[myIndex]);
						}
					} else {
						if (candidateBids.isEmpty()) {
							if (!acceptAction[myNextPartyIndex]
									&& !acceptAction[myLastPartyIndex]) {
								generateCandidateFromOppTopBids(acceptUtil,
										candidateBids);
								generateCandidateFromMyTopBids(acceptUtil,
										candidateBids);
							} else if (!acceptAction[myNextPartyIndex]
									&& acceptAction[myLastPartyIndex]) {
								generateCandidateFromOppTopAndMyTopBids(
										myNextPartyIndex, acceptUtil,
										candidateBids);
								generateCandidateFromMyFullAndOppTopBids(
										acceptUtil, myNextPartyIndex,
										candidateBids);
							} else {
								generateCandidateFromOppTopAndMyTopBids(
										myLastPartyIndex, acceptUtil,
										candidateBids);
								generateCandidateFromMyFullAndOppTopBids(
										acceptUtil, myLastPartyIndex,
										candidateBids);
							}
						}
						if (!candidateBids.isEmpty()) {
							bid = candidateBids.remove(0);
							if (hasProposedMyLastRound(bid)) {
								bid = proposedBids[myIndex].get(
										nrChosenActions % numOfBids[myIndex]);
							}
						} else {
							bid = proposedBids[myIndex]
									.get(nrChosenActions % numOfBids[myIndex]);
						}
						action = new Offer(getPartyId(), bid);
					}
					if (lastPeriodBidList.isEmpty()) {
						lastPeriodBidList = getMaxUtilBidsInList(
								sameBidsInOpps);
					}
					if (lastPeriodBidList.isEmpty()) {
						mergeBids(proposedBids[myNextPartyIndex].get(0),
								proposedBids[myLastPartyIndex].get(0), lower,
								lastPeriodBidList, lastPeriodBidList);
					}
				} else if (timeline.getTime() > time1) {
					if (candidateBids.isEmpty()) {
						if (nrChosenActions % 3 == 0) {
							generateCandidateFromOppTopBids(acceptUtil,
									candidateBids);
							generateCandidateFromMyTopBids(acceptUtil,
									candidateBids);
						} else {
							if (!acceptAction[myNextPartyIndex]) {
								generateCandidateFromMyFullAndOppTopBids(
										acceptUtil, myNextPartyIndex,
										candidateBids);
								generateCandidateFromMyFullAndMyTopBids(
										acceptUtil, myNextPartyIndex,
										candidateBids);
								generateCandidateFromOppTopAndMyTopBids(
										myNextPartyIndex, acceptUtil,
										candidateBids);
							}
							if (!acceptAction[myLastPartyIndex]) {
								generateCandidateFromMyFullAndOppTopBids(
										acceptUtil, myLastPartyIndex,
										candidateBids);
								generateCandidateFromMyFullAndMyTopBids(
										acceptUtil, myLastPartyIndex,
										candidateBids);
								generateCandidateFromOppTopAndMyTopBids(
										myLastPartyIndex, acceptUtil,
										candidateBids);
							}
						}
					}
					if (!candidateBids.isEmpty()) {
						bid = candidateBids.remove(0);
						if (hasProposedMyLastRound(bid)) {
							bid = proposedBids[myIndex]
									.get(nrChosenActions % numOfBids[myIndex]);
						}
					} else {
						bid = proposedBids[myIndex]
								.get(nrChosenActions % numOfBids[myIndex]);
					}
				} else {
					if (candidateBids.isEmpty()) {
						generateCandidateFromMyFullBids(proposedBids[myIndex]);
						generateCandidateFromMyTopBids(acceptUtil,
								proposedBids[myIndex]);
						generateCandidateFromMyFullAndMyTopBids(acceptUtil,
								myNextPartyIndex, proposedBids[myIndex]);
						generateCandidateFromMyFullAndMyTopBids(acceptUtil,
								myLastPartyIndex, proposedBids[myIndex]);
					}
					if (!candidateBids.isEmpty()) {
						bid = candidateBids.remove(0);
					} else {
						Bid nBid = fullUtilBids
								.get(nrChosenActions % fullUtilBids.size());
						if (proposedBids[myIndex].contains(nBid)) {
							bid = proposedBids[myIndex]
									.get(nrChosenActions % numOfBids[myIndex]);
						} else {
							bid = nBid;
						}
					}
				}
				if (bid == null) {
					lastConcessionPoint = getBidWithLessBoundedUtil(
							lastConcessionPoint, acceptUtil);
					bid = lastConcessionPoint;
					if (hasProposedMyLastRound(bid)) {
						bid = proposedBids[myIndex]
								.get(nrChosenActions % numOfBids[myIndex]);
					}
				}
				if (bid == null) {
					bid = myBid;
				}
				action = new Offer(getPartyId(), bid);
			}
		} else {
			bid = new Bid(
					fullUtilBids.get(nrChosenActions % fullUtilBids.size()));
			action = new Offer(getPartyId(), bid);
		}
		myBid = bid;
		numOfBids[myIndex]++;
		proposedBids[myIndex].add(bid);
		return action;
	}

	@SuppressWarnings("unchecked")
	@Override
	public void receiveMessage(AgentID sender, Action action) {
		super.receiveMessage(sender, action);
		if (sender == null) {
			numOfParties = getNumberOfParties();
			myIndex = Integer.parseInt(getPartyId().toString().split("@")[1])
					% numOfParties;
			myNextPartyIndex = (myIndex + 1) % numOfParties;
			myLastPartyIndex = (myIndex + numOfParties - 1) % numOfParties;
			proposedBids = new ArrayList[numOfParties];
			numOfBids = new int[numOfParties];
			valueFrequences = new HashMap[numOfParties];
			issueChangeFrequences = new HashMap[numOfParties];
			acceptAction = new boolean[numOfParties];
			maxUtils = new double[numOfParties];
			minUtils = new double[numOfParties];
			proposedMaxBid = new LinkedList[numOfParties];
			proposedMinBid = new LinkedList[numOfParties];
			pModel = new PreferModel[numOfParties];
			acceptableBidsWithOppValues = new ArrayList[numOfParties];
			candidateBids = new ArrayList<>();
			maxProductBids = new ArrayList[numOfParties];
			topBidsInOppsSpace = new ArrayList[numOfParties];
			topBidsInMySpace = new LinkedList[numOfParties];
			for (int i = 0; i < numOfParties; i++) {
				proposedBids[i] = new ArrayList<>();
				valueFrequences[i] = new HashMap<>();
				issueChangeFrequences[i] = new HashMap<>();
				proposedMaxBid[i] = new LinkedList<>();
				proposedMinBid[i] = new LinkedList<>();
				pModel[i] = new PreferModel(getUtilitySpace(), numOfIssues);
				acceptableBidsWithOppValues[i] = new ArrayList<>();
				maxProductBids[i] = new ArrayList<>();
				topBidsInOppsSpace[i] = new ArrayList<>();
				topBidsInMySpace[i] = new LinkedList<>();
			}
		} else {
			oppName = sender.toString().split("@")[0];
			oppIndex = Integer.parseInt(sender.toString().split("@")[1])
					% numOfParties;
			if (action instanceof Offer) {
				acceptAction[oppIndex] = false;
				lastReceivedBid = ((Offer) action).getBid();
			}
			if (action instanceof Accept) {
				acceptAction[oppIndex] = true;
				lastReceivedBid = ((Accept) action).getBid();
			}
			utility = getUtility(lastReceivedBid);
			proposedBids[oppIndex].add(lastReceivedBid);
			numOfBids[oppIndex]++;
			updateMaxOrMinBids(oppIndex);
			if (isDifferentBid(oppIndex)) {
				updateSameBidsInOpps(oppIndex, lastReceivedBid);
				updateTopBidsInOppsSpace(oppIndex);
				updateTopBidsInMySpace(oppIndex);
			}
			if (hasHistory) {
				if (!hasInitParamByHist) {
					hasHistWithRightPosition = false;
					for (String name : maxUtilsInHist[oppIndex].keySet()) {
						if (oppName.equals(name)) {
							hasHistWithRightPosition = true;
						}
					}
					if (hasHistWithRightPosition) {
						double[] arrs = new double[3];
						for (int i = 0; i < getNumberOfParties(); i++) {
							if (i == myIndex) {
								continue;
							}
							for (String name : maxUtilsInHist[i].keySet()) {
								if (oppIndex == i) {
									arrs[i] = maxUtilsInHist[oppIndex]
											.get(oppName);
								} else {
									if (!name.equals(oppName)) {
										arrs[i] = maxUtilsInHist[i].get(name);
									}
								}
							}
						}
						double max = 0.0;
						double min = 1.0;
						for (int i = 0; i < 3; i++) {
							if (i == myIndex) {
								continue;
							}
							if (max < arrs[i]) {
								max = arrs[i];
							}
							if (min > arrs[i]) {
								min = arrs[i];
							}
						}
						if (maxAgreedBids.isEmpty()) {
							if (min - 0.3 > reservationValue) {
								lower = min - 0.3;
							} else {
								lower = reservationValue;
							}
						} else {
							double agreedUtil = getUtility(
									maxAgreedBids.get(0));
							double[] utilArrs = { max, min, agreedUtil };
							upper = getMaxInArray(utilArrs);
							for (int i = 0; i < utilArrs.length; i++) {
								if (utilArrs[i] == upper) {
									utilArrs[i] = 0.0;
								}
							}
							lower = getMinInArray(utilArrs) - 0.1;
							if (lower <= 0) {
								lower = reservationValue;
							}
						}
						exponent = initialExponent * (upper - lower) / 2;
					} else {
						double[] arrs = new double[3];
						for (int i = 0; i < getNumberOfParties(); i++) {
							if (i == myIndex) {
								continue;
							}
							for (String name : maxUtilsInHist[i].keySet()) {
								arrs[i] = maxUtilsInHist[i].get(name);
							}
						}
						double max = 0.0;
						double min = 1.0;
						for (int i = 0; i < 3; i++) {
							if (i == myIndex) {
								continue;
							}
							if (max < arrs[i]) {
								max = arrs[i];
							}
							if (min > arrs[i]) {
								min = arrs[i];
							}
						}
						if (maxAgreedBids.isEmpty()) {
							if (min - 0.3 > reservationValue) {
								lower = min - 0.3;
							} else {
								lower = reservationValue;
							}
						} else {
							double agreedUtil = getUtility(
									maxAgreedBids.get(0));
							double[] utilArrs = { max, min, agreedUtil, 0.9 };
							upper = getMaxInArray(utilArrs);
							for (int i = 0; i < utilArrs.length; i++) {
								if (utilArrs[i] == upper) {
									utilArrs[i] = 0.0;
								}
							}
							lower = getMinInArray(utilArrs) - 0.2;
							if (lower <= 0) {
								lower = reservationValue;
							}
						}
						exponent = initialExponent * (upper - lower) / 2;
					}
					hasInitParamByHist = true;
				}
				if (hasInitParamByHist) {
					if (hasHistWithRightPosition) {

					} else {

					}
				}
			} else {
				if (isDifferentBid(oppIndex)) {
					pModel[oppIndex].updateEvaluators(lastReceivedBid);
					if (timeline.getTime() > 0.5 && timeline.getTime() < 0.9) {
						updateMaxUtilAddition(oppIndex, lastReceivedBid);
					}
				}
				updateConcessionParam();
			}
		}
	}

	@Override
	public String getDescription() {
		return "ANAC2017";
	}

	private double getAccept() {
		double accept = 1.0;
		accept = upper - (upper + 0.001 - lower)
				* Math.pow(timeline.getTime(), exponent);
		return accept;
	}

	private List<Bid> getFullUtilBid(AbstractUtilitySpace utilitySpace) {
		Bid bid = null;
		List<Bid> bidList = new ArrayList<>();
		try {
			bid = utilitySpace.getMaxUtilityBid();
		} catch (Exception e) {
			e.printStackTrace();
			System.err.println(e.toString());
		}
		if (!bidList.contains(bid)) {
			bidList.add(bid);
		}
		List<Issue> issues = getUtilitySpace().getDomain().getIssues();
		for (Issue item : issues) {
			Bid newBid = null;
			switch (item.getType()) {
			case DISCRETE:
				IssueDiscrete discrete = (IssueDiscrete) item;
				for (Value value : discrete.getValues()) {
					newBid = bid.putValue(discrete.getNumber(), value);
					if (getUtility(newBid) == 1.0
							&& !bidList.contains(newBid)) {
						bidList.add(newBid);
					}
				}
				break;
			case INTEGER:
				break;
			default:
				break;
			}
		}
		return bidList;
	}

	@SuppressWarnings("unused")
	private Bid getBidWithLessUtil(Bid bid) {
		Bid srcBid = null;
		Bid nBid = null;
		if (bid == null) {
			srcBid = fullUtilBids.get(0);
		} else {
			srcBid = new Bid(bid);
		}
		double oldU = getUtility(srcBid);
		List<Bid> bidList = new ArrayList<>();
		for (int i = 0; i < numOfIssues; i++) {
			Issue issue = srcBid.getIssues().get(i);
			switch (issue.getType()) {
			case DISCRETE:
				IssueDiscrete issueDiscrete = (IssueDiscrete) issue;
				for (Value value : issueDiscrete.getValues()) {
					nBid = srcBid.putValue(i + 1, value);
					if (oldU > getUtility(nBid)) {
						bidList.add(nBid);
					}
				}
				break;
			case INTEGER:
				IssueInteger issueInteger = (IssueInteger) issue;
				for (int j = issueInteger.getLowerBound(); j <= issueInteger
						.getUpperBound(); j++) {
					nBid = srcBid.putValue(i + 1, new ValueInteger(j));
					if (oldU > getUtility(nBid)) {
						bidList.add(nBid);
					}
				}
				break;
			default:
				break;
			}
		}
		if (bidList.isEmpty()) {
			nBid = new Bid(srcBid);
		} else {
			double distance = oldU - 0.0;
			for (Bid item : bidList) {
				double distance2 = oldU - getUtility(item);
				if (distance2 < distance) {
					nBid = new Bid(item);
					distance = distance2;
				}
			}
		}
		return nBid;
	}

	private Bid getBidWithLessBoundedUtil(Bid bid, double min) {
		Bid srcBid = null;
		Bid nBid = null;
		if (bid == null) {
			srcBid = fullUtilBids.get(0);
		} else {
			srcBid = new Bid(bid);
		}
		double oldU = getUtility(srcBid);
		List<Bid> bidList = new ArrayList<>();
		for (int i = 0; i < numOfIssues; i++) {
			Issue issue = srcBid.getIssues().get(i);
			double newU = 0.0;
			switch (issue.getType()) {
			case DISCRETE:
				IssueDiscrete issueDiscrete = (IssueDiscrete) issue;
				for (Value value : issueDiscrete.getValues()) {
					nBid = srcBid.putValue(i + 1, value);
					newU = getUtility(nBid);
					if (oldU > newU && newU >= min) {
						bidList.add(nBid);
					}
				}
				break;
			case INTEGER:
				IssueInteger issueInteger = (IssueInteger) issue;
				for (int j = issueInteger.getLowerBound(); j <= issueInteger
						.getUpperBound(); j++) {
					nBid = srcBid.putValue(i + 1, new ValueInteger(j));
					newU = getUtility(nBid);
					if (oldU > newU && newU >= min) {
						bidList.add(nBid);
					}
				}
				break;
			default:
				break;
			}
		}
		if (bidList.isEmpty()) {
			nBid = new Bid(srcBid);
		} else {
			double distance = oldU - 0.0;
			for (Bid item : bidList) {
				double distance2 = oldU - getUtility(item);
				if (distance2 < distance) {
					nBid = new Bid(item);
					distance = distance2;
				}
			}
		}
		return nBid;
	}

	private void updateProposedMaxBid(int partyId) {
		if (proposedMaxBid[partyId].isEmpty()) {
			if (utility > maxUtils[partyId]) {
				maxUtils[partyId] = utility;
				proposedMaxBid[partyId].add(lastReceivedBid);
			}
		} else {
			if (maxUtils[partyId] > utility) {
				return;
			}
			if (maxUtils[partyId] == utility
					&& !proposedMaxBid[partyId].contains(lastReceivedBid)) {
				proposedMaxBid[partyId].add(lastReceivedBid);
			}
			if (maxUtils[partyId] < utility) {
				proposedMaxBid[partyId].clear();
				proposedMaxBid[partyId].add(lastReceivedBid);
				maxUtils[partyId] = utility;
			}
		}
	}

	private void updateProposedMinBid(int partyId) {
		if (proposedMinBid[partyId].isEmpty()) {
			proposedMinBid[partyId].add(lastReceivedBid);
			if (0 >= minUtils[partyId]) {
				minUtils[partyId] = utility;
			}
		} else {
			if (minUtils[partyId] < utility) {
				return;
			}
			if (minUtils[partyId] == utility
					&& !proposedMinBid[partyId].contains(lastReceivedBid)) {
				proposedMinBid[partyId].add(lastReceivedBid);
			}
			if (minUtils[partyId] > utility) {
				proposedMinBid[partyId].clear();
				proposedMinBid[partyId].add(lastReceivedBid);
				minUtils[partyId] = utility;
			}
		}
	}

	private void updateMaxOrMinBids(int partyId) {
		updateProposedMaxBid(partyId);
		updateProposedMinBid(partyId);
	}

	private void updateTopBidsInOppsSpace(int partyId) {
		if (topBidsInOppsSpace[partyId].size() < 5
				&& !topBidsInOppsSpace[partyId].contains(lastReceivedBid)) {
			topBidsInOppsSpace[partyId].add(lastReceivedBid);
		}
	}

	private void updateTopBidsInMySpace(int partyId) {
		if (topBidsInMySpace[partyId].contains(lastReceivedBid)) {
			return;
		} else {
			if (topBidsInMySpace[partyId].size() < 5) {
				topBidsInMySpace[partyId].add(lastReceivedBid);
			} else {
				int removeIndex = -1;
				double maxDistance = 0.0;
				for (int i = 0; i < topBidsInMySpace[partyId].size(); i++) {
					double distance = utility
							- getUtility(topBidsInMySpace[partyId].get(i));
					if (distance > maxDistance) {
						maxDistance = distance;
						removeIndex = i;
					}
				}
				if (removeIndex != -1) {
					topBidsInMySpace[partyId].set(removeIndex, lastReceivedBid);
				}
			}
		}
	}

	private void updateConcessionParam() {
		if (!hasHistory) {
			// ����Ϊ�����������С��utility
			double maxInMinUtil = 0.0;
			double minInMinUtil = 1.0;
			double minInMaxUtil = 1.0;
			// double[] avgUtil = new double[numOfParties];
			for (int i = 0; i < numOfParties; i++) {
				if (i == myIndex) {
					// avgUtil[i] = 1.0;
					continue;
				}
				if (maxUtils[i] == 0.0 || minUtils[i] == 0.0) {
					continue;
				}
				double utilInMin2 = minUtils[i];
				if (utilInMin2 > maxInMinUtil) {
					maxInMinUtil = utilInMin2;
				}
				if (utilInMin2 < minInMinUtil) {
					minInMinUtil = utilInMin2;
				}
				double utilInMax = maxUtils[i];
				if (minInMaxUtil > utilInMax) {
					minInMaxUtil = utilInMax;
				}
				// avgUtil[i] = (maxUtils[i]+minUtils[i])/2;
			}
			double lowerbound = 0.0;
			if (reservationValue > minInMinUtil - 0.01) {
				lowerbound = reservationValue;
			} else {
				lowerbound = minInMinUtil - 0.01;
			}
			lower = lowerbound;
			if (minInMaxUtil <= 0.0) {
				exponent = 1.0;
			} else {
				exponent = initialExponent * minInMaxUtil;
			}
		} else {

		}
	}

	private double getMinInArray(double arr[]) {
		double min = arr[0];
		for (int i = 0; i < arr.length; i++) {
			if (min > arr[i]) {
				min = arr[i];
			}
		}
		return min;
	}

	private double getMaxInArray(double[] arr) {
		double max = 0.0;
		for (int i = 0; i < arr.length; i++) {
			if (max < arr[i]) {
				max = arr[i];
			}
		}
		return max;
	}

	private double getMaxInArray(List<Double> list) {
		double max = 0.0;
		for (double item : list) {
			if (item > max) {
				max = item;
			}
		}
		return max;
	}

	private boolean isDifferentBid(int partyId) {
		if (proposedBids[partyId].size() < 2) {
			return true;
		} else {
			if (proposedBids[partyId].get(numOfBids[partyId] - 1).equals(
					proposedBids[partyId].get(numOfBids[partyId] - 2))) {
				return false;
			} else {
				return true;
			}
		}
	}

	private void alterBidValue(Bid srcBid, Map<Integer, Value> valueMap,
			int issueNr, double acceptableUtil, List<Bid> bidList,
			List<Bid> validationList) {
		Bid nBid = new Bid(srcBid);
		if (getUtility(nBid) >= acceptableUtil
				&& !validationList.contains(nBid)) {
			bidList.add(nBid);
		}
		if (!valueMap.containsKey(issueNr)) {
			return;
		}
		for (int j = issueNr; j <= valueMap.size(); j++) {
			Value value = valueMap.get(j);
			if (value.equals(srcBid.getValue(j))) {
				continue;
			}
			nBid = srcBid.putValue(j, value);
			int item = j + 1;
			alterBidValue(nBid, valueMap, item, acceptableUtil, bidList,
					validationList);
		}
	}

	private void mergeBids(Bid fir, Bid sec, double acceptableUtil,
			List<Bid> bidList, List<Bid> validationList) {
		if (fir == null || sec == null) {
			return;
		}
		if (fir.equals(sec)) {
			if (!validationList.contains(fir)) {
				bidList.add(fir);
			}
			return;
		}
		alterBidValue(fir, sec.getValues(), 1, acceptableUtil, bidList,
				validationList);
	}

	private void generateCandidateFromMyFullBidsAndOppsTopBids(double minUtil,
			int oppIndex, List<Bid> validationList) {
		if (proposedBids[oppIndex].isEmpty()) {
			return;
		}
		for (Bid item : fullUtilBids) {
			for (Bid item2 : topBidsInOppsSpace[oppIndex]) {
				if (item.equals(item2)) {
					continue;
				}
				mergeBids(item, item2, minUtil, candidateBids, validationList);
			}
		}
	}

	private void generateCandidate(double acceptableUtil,
			List<Bid> validationList) {
		generateCandidateFromMyTopBids(acceptableUtil, validationList);
		generateCandidateFromOppTopBids(acceptableUtil, validationList);
	}

	private List<Bid> getMaxSocialWelfareBidsInList(List<Bid> bidList) {
		List<Bid> maxAdditionBids = new ArrayList<>();
		List<Bid> maxProductBids = new ArrayList<>();
		List<Bid> maxSocialWelfareBids = new ArrayList<>();
		double maxAddition = 0.0;
		double maxProduct = 0.0;
		for (Bid bid : bidList) {
			double bidAddition = 0.0;
			double bidProduct = 1.0;
			bidAddition = getUtilAddition(bid);
			bidProduct = getUtilProduct(bid);
			if (bidAddition > maxAddition) {
				maxAdditionBids.clear();
				maxAdditionBids.add(bid);
				maxAddition = bidAddition;
			} else if (bidAddition == maxAddition
					&& !maxAdditionBids.contains(bid)) {
				maxAdditionBids.add(bid);
			}
			if (bidProduct > maxProduct) {
				maxProductBids.clear();
				maxProductBids.add(bid);
				maxProduct = bidProduct;
			} else if (bidProduct == maxProduct
					&& !maxProductBids.contains(bid)) {
				maxProductBids.add(bid);
			}
		}
		maxSocialWelfareBids.addAll(maxAdditionBids);
		maxSocialWelfareBids.addAll(maxProductBids);
		return maxSocialWelfareBids;
	}

	private double getUtilProduct(Bid bid) {
		double product = 1.0;
		for (int i = 0; i < numOfParties; i++) {
			if (i == myIndex) {
				product *= getUtility(bid);
			}
			product *= pModel[i].getUtil(bid);
		}
		return product;
	}

	private double getUtilAddition(Bid bid) {
		double addition = 0.0;
		for (int i = 0; i < numOfParties; i++) {
			if (i == myIndex) {
				addition += getUtility(bid);
			}
			addition += pModel[i].getUtil(bid);
		}
		return addition;
	}

	private List<Bid> getMaxUtilBidsInList(List<Bid> bidList) {
		List<Bid> bids = new ArrayList<>();
		double maxUtil = 0.0;
		for (Bid item : bidList) {
			double util = getUtility(item);
			if (util < maxUtil) {
				continue;
			}
			if (util > maxUtil) {
				bids.clear();
				bids.add(item);
				maxUtil = util;
			} else {
				bids.add(item);
			}
		}
		return bids;
	}

	private void updateSameBidsInOpps(int proposedPartyId, Bid bid) {
		boolean bidHasProposedByAllOpps = true;
		for (int i = 0; i < numOfParties; i++) {
			if (i == myIndex || i == proposedPartyId) {
				continue;
			}
			if (!proposedBids[i].contains(bid)) {
				bidHasProposedByAllOpps = false;
				return;
			} else {
				bidHasProposedByAllOpps = true;
			}
		}
		if (bidHasProposedByAllOpps) {
			sameBidsInOpps.add(bid);
		}
	}

	private void updateMaxUtilAddition(int partyId, Bid bid) {
		double utilAddition = 0.0;
		for (int i = 0; i < numOfParties; i++) {
			if (partyId == myIndex) {
				utilAddition += getUtility(bid);
			} else {
				utilAddition += pModel[i].getUtil(bid);
			}
		}
		if (utilAddition < maxUtilAdditionByOpps) {
			return;
		}
		if (utilAddition > maxUtilAdditionByOpps) {
			maxUtilAdditionBidByOpps.clear();
			maxUtilAdditionBidByOpps.add(bid);
			maxUtilAdditionByOpps = utilAddition;
		} else {
			maxUtilAdditionBidByOpps.add(bid);
		}
	}

	private void generateCandidateFromOppTopAndMyTopBids(int partyId,
			double minUtil, List<Bid> validationList) {
		if (proposedBids[partyId].isEmpty()) {
			return;
		}
		for (Bid item : topBidsInOppsSpace[partyId]) {
			for (Bid maxBid : topBidsInMySpace[partyId]) {
				mergeBids(maxBid, item, minUtil, candidateBids, validationList);
			}
		}
	}

	private List<Bid> getMaxBidsInAgreements() {
		double maxUtil = 0.0;
		List<Bid> maxBids = new ArrayList<>();
		for (Tuple<Bid, Double> item : agreements) {
			if (item.get2() < maxUtil) {
				continue;
			}
			if (item.get2() > maxUtil) {
				maxBids.clear();
				maxBids.add(item.get1());
				maxUtil = item.get2();
			} else {
				if (!maxBids.contains(item.get1())) {
					maxBids.add(item.get1());
				}
			}
		}
		return maxBids;
	}

	private boolean hasProposedMyLastRound(Bid bid) {
		if (numOfBids[myIndex] <= 0) {
			return false;
		}
		Bid lastRoundBid = proposedBids[myIndex].get(numOfBids[myIndex] - 1);
		return lastRoundBid.equals(bid);
	}

	private void generateCandidateFromMyFullBids(List<Bid> validationList) {
		for (Bid item1 : fullUtilBids) {
			for (Bid item2 : fullUtilBids) {
				if (item1.equals(item2)) {
					if (validationList.contains(item1)) {
						continue;
					} else {
						candidateBids.add(item1);
						continue;
					}
				} else {
					mergeBids(item1, item2, 0.9, candidateBids, validationList);
				}
			}
		}
	}

	private void generateCandidateFromMyFullAndMyTopBids(double accept,
			int partyId, List<Bid> validationList) {
		for (Bid item1 : fullUtilBids) {
			for (Bid item2 : topBidsInMySpace[partyId]) {
				mergeBids(item1, item2, accept, candidateBids, validationList);
			}
		}
	}

	private void generateCandidateFromMyFullAndOppTopBids(double accept,
			int partyId, List<Bid> validationList) {
		for (Bid item1 : fullUtilBids) {
			for (Bid item2 : topBidsInOppsSpace[partyId]) {
				mergeBids(item1, item2, accept, candidateBids, validationList);
			}
		}
	}

	private void generateCandidateFromMyTopBids(double acceptableUtil,
			List<Bid> validationList) {
		for (Bid fBid : topBidsInMySpace[myNextPartyIndex]) {
			for (Bid sBid : topBidsInMySpace[myLastPartyIndex]) {
				alterBidValue(fBid, sBid.getValues(), 1, acceptableUtil,
						candidateBids, validationList);
			}
		}
	}

	private void generateCandidateFromOppTopBids(double acceptableUtil,
			List<Bid> validationList) {
		if (numOfBids[myLastPartyIndex] == 0
				|| numOfBids[myNextPartyIndex] == 0) {
			return;
		}
		for (Bid item1 : topBidsInOppsSpace[myNextPartyIndex]) {
			for (Bid item2 : topBidsInOppsSpace[myLastPartyIndex]) {
				if (item1.equals(item2)) {
					continue;
				}
				mergeBids(item1, item2, acceptableUtil, candidateBids,
						validationList);
			}
		}
	}
}