package agents.anac.y2017.parsagent3;

import java.util.List;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import negotiator.parties.AbstractTimeDependentNegotiationParty;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.EndNegotiation;
import genius.core.actions.Offer;
import genius.core.boaframework.SortedOutcomeSpace;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.list.Tuple;
import genius.core.parties.NegotiationInfo;
import genius.core.persistent.PersistentDataType;
import genius.core.persistent.StandardInfo;
import genius.core.persistent.StandardInfoList;
import genius.core.timeline.DiscreteTimeline;

public class ShahAgent extends AbstractTimeDependentNegotiationParty {
	Bid lastBid;
	Bid myLastBid;
	Bid myBestBid;
	Bid lastReceivedBid;
	String oppAName;
	String oppBName;
	private StandardInfoList history;
	boolean fornullAgent = false;
	int round;
	Integer TimePeriod;
	ArrayList<Bid> oppABids = new ArrayList<Bid>();
	ArrayList<Bid> oppBBids = new ArrayList<Bid>();
	ArrayList<Cluster> clusterA;
	ArrayList<Cluster> clusterB;
	ClusterHistory clusterHistoryA;
	ClusterHistory clusterHistoryB;
	int numberOfclusters = 20;
	int slotNum = 100;
	double myConcessionRate = 0.15d;
	double ConstantUtility = 1.0d;
	float myReservation = 0.8f;
	int currentSlotA;
	int currentSlotB;
	boolean clusterIsRefreshA;
	boolean clusterIsRefreshB;
	float concessionA;
	float sumConcessionA;
	float sumConcessionB;
	int concessionNumber;
	// ArrayList<Float> concessionAHis;
	// ArrayList<Float> concessionBHis;
	float concessionB;
	ArrayList<Bid> centsA = new ArrayList<Bid>();
	ArrayList<Bid> centsB = new ArrayList<Bid>();
	private double rValue;
	private double dFactor;
	SortedOutcomeSpace outcomeSpace;
	float minLimit = 0.85f;
	boolean conceed = true;

	public ShahAgent() {
		fornullAgent = false;
	}

	public void clusterOpponentBidWithoutFirstLevel(ArrayList<Bid> newCenters,
			ArrayList<Bid> bidHistory, String opParty, boolean repeat) {

		int k = numberOfclusters;
		ArrayList<Cluster> clusters;
		if (opParty.equals("A"))
			clusters = clusterA;
		else
			clusters = clusterB;
		if (!repeat) {

			if (opParty.equals("A"))
				clusterA = clusters;
			else
				clusterB = clusters;
		}
		for (int i = 0; i < newCenters.size(); ++i)
			defineCluster(clusters, newCenters.get(i));
		if (refreshClusterCenters(clusters)) {

			for (int i = 0; i < clusters.size(); ++i) {
				clusters.get(i).getMembers().clear();

			}
			clusterOpponentBid(bidHistory, opParty, true);
		}
	}

	public void clusterOpponentBid(ArrayList<Bid> bidHistory, String opParty,
			boolean repeat) {

		int k = numberOfclusters;
		ArrayList<Cluster> clusters;
		if (opParty.equals("A"))
			clusters = clusterA;
		else
			clusters = clusterB;
		if (bidHistory.size() > k) {
			if (!repeat) {
				if (clusters == null) {
					clusters = new ArrayList<Cluster>(k);
					for (int i = 0; i < k; ++i) {
						clusters.add(new Cluster());
					}
				}
				if (opParty.equals("A"))
					clusterA = clusters;
				else
					clusterB = clusters;

				int next = bidHistory.size() / k;
				int temp = 0;
				for (int i = 0; i < k; ++i) {
					clusters.get(i).setCenter(bidHistory.get(temp));
					temp += next;
				}
				for (int i = 0; i < clusters.size(); ++i) {
					if (clusters.get(i).getMembers() != null)
						clusters.get(i).getMembers().clear();
				}
			}
			for (int i = 0; i < bidHistory.size(); ++i) {
				defineCluster(clusters, bidHistory.get(i));
			}
			if (refreshClusterCenters(clusters)) {

				for (int i = 0; i < clusters.size(); ++i) {
					clusters.get(i).getMembers().clear();
				}
				clusterOpponentBid(bidHistory, opParty, true);
			}
		}
	}

	public boolean refreshClusterCenters(ArrayList<Cluster> clusters) {
		// ArrayList<Issue> dissues = utilitySpace.getDomain().getIssues();
		boolean needRefresh = false;
		for (int i = 0; i < clusters.size(); ++i) {
			Cluster clus = clusters.get(i);
			clus.getMembers().add(0, clus.getCenter());
			float[] dist = new float[clus.getMembers().size()];
			float[][] dismatrix = new float[clus.getMembers().size()][clus
					.getMembers().size()];
			for (int j = 0; j < clus.getMembers().size(); ++j) {
				Bid checkBid = clus.getMembers().get(j);
				float distance = 0;
				for (int k = 0; k < j; ++k) {
					distance += dismatrix[k][j];
				}
				for (int k = j + 1; k < clus.getMembers().size(); ++k) {
					// if (k != j) {
					float temp = calDist(checkBid, clus.getMembers().get(k));
					distance += temp;
					dismatrix[j][k] = temp;

				}
				dist[j] = distance / clus.getMembers().size();
				distance = 0f;
			}
			float min = dist[0];
			String out = dist[0] + "";

			int index = 0;
			for (int n = 1; n < dist.length; ++n) {
				// out += " " + dist[n];

				if (dist[n] < min) {
					min = dist[n];
					index = n;
				}
			}

			if (index != 0) {
				needRefresh = true;
			}

			clus.setCenter(clus.getMembers().get(index));
			clus.getMembers().remove(index);

		}
		return needRefresh;
	}

	public void defineCluster(ArrayList<Cluster> clusters, Bid bid) {

		float[] dises = new float[clusters.size()];
		for (int i = 0; i < clusters.size(); ++i) {
			dises[i] = calDist(bid, clusters.get(i).getCenter());
			if (clusters.get(i).members == null)
				clusters.get(i).members = new ArrayList<Bid>();
		}
		float min = dises[0];
		int index = 0;
		for (int j = 0; j < dises.length; ++j) {
			if (dises[j] < min) {
				min = dises[j];
				index = j;
			}
		}

		clusters.get(index).members.add(bid);

	}

	@Override
	public String getDescription() {
		return "ANAC2017";
	}

	class ClusterHistory {
		ArrayList<ArrayList<Bid>> prevCenters;
		ArrayList<Float> maxDistance;
		ArrayList<Float> distributionFactor;
		ArrayList<Integer> submitTime;
	}

	class Cluster {
		Bid center;
		ArrayList<Bid> members;

		public void setCenter(Bid center) {
			this.center = center;
		}

		public Bid getCenter() {
			return center;
		}

		public void setMembers(ArrayList<Bid> members) {
			this.members = members;
		}

		public ArrayList<Bid> getMembers() {
			return members;
		}
	}

	@Override
	public void init(NegotiationInfo info) {
		super.init(info);
		rValue = utilitySpace.getReservationValue().doubleValue();
		dFactor = utilitySpace.getDiscountFactor();
		outcomeSpace = new SortedOutcomeSpace(getUtilitySpace());

		if (getData().getPersistentDataType() != PersistentDataType.STANDARD) {
			throw new IllegalStateException("need standard persistent data");
		}
		history = (StandardInfoList) getData().get();

		if (!history.isEmpty()) {
			// System.out.println("sizeeee " + history.size());
			Map<String, Double> maxutils = new HashMap<String, Double>();
			Map<String, Double> minutils = new HashMap<String, Double>();
			StandardInfo lastinfo = history.get(history.size() - 1);
			int i = 0;
			for (Tuple<String, Double> offered : lastinfo.getUtilities()) {
				++i;
				String party = offered.get1().indexOf("@") != -1 ? offered
						.get1().substring(0, offered.get1().indexOf("@"))
						: offered.get1();
				Double util = offered.get2();
				maxutils.put(party, maxutils.containsKey(party)
						? Math.max(maxutils.get(party), util) : util);
				minutils.put(party, minutils.containsKey(party)
						? Math.min(minutils.get(party), util) : util);
			}
			Set keys = maxutils.keySet();
			double conceedA = 0;
			double conceedB = 0;
			for (Object partyK : keys) {
				if (((String) partyK).indexOf("ShahAgent") == -1) {
					if (conceedA == 0) {
						conceedA = maxutils.get(partyK) - minutils.get(partyK);
					} else
						conceedB = maxutils.get(partyK) - minutils.get(partyK);
				}

			}
			if (conceedA < 0.3 || conceedB < 0.3)
				conceed = false; // for choosing policy of conceesion

			if (conceedA > 0.8 && conceedB > 0.8) {
				minLimit = 0.7f;
			}
			if (conceedA > 0.6 && conceedB > 0.6) {
				minLimit = 0.8f;
			}
		}

	}

	public boolean isLastSecond() {
		double time = timeline.getTime();
		double second = time * 3.0 * 60.0;
		if (second >= (3.0 * 60 - 1.0))
			return true;
		return false;
	}

	public boolean isLastTime() {
		double time = timeline.getTime();
		double second = time * 3.0 * 60.0;
		if (second >= (3.0 * 60 - 1.0) + 0.8)
			return true;
		return false;
	}

	@Override
	public Action chooseAction(List validActions) {

		if (round == 100)
			round = 0;
		++round;
		double time = timeline.getTime();
		if (selectEndNegotiation(time) && utilitySpace
				.getUtility(lastBid) < rValue * Math.pow(dFactor, time)) {
			return new EndNegotiation(getPartyId());

		}
		if (lastBid != null
				&& utilitySpace.getUtility(lastBid) >= getMyutility()) {
			return new Accept(getPartyId(), lastReceivedBid);
			// } else if (isLastSecond() &&
			// (lastBid != null && utilitySpace.getUtility(lastBid) >=
			// (getMyutility() - 0.1))) {
			// return new Accept(getPartyId(),lastReceivedBid);
		} else if (isLastTime() && (lastBid != null && utilitySpace
				.getUtility(lastBid) >= (getMyutility() - 0.05))) {
			return new Accept(getPartyId(), lastReceivedBid);
		} else { // propose a new bid
			Bid newBid;

			try {
				double random;
				double temp = getLowerBound();
				if (temp < minLimit)
					temp = minLimit;

				do {
					random = new Random().nextDouble();
				} while (random <= temp);
				newBid = outcomeSpace.getBidNearUtility(random).getBid();

			} catch (Exception e) {
				newBid = outcomeSpace.getBidNearUtility(0.9).getBid();
			}
			myLastBid = newBid;
			lastBid = myLastBid;
			return new Offer(getPartyId(), newBid);
		}

	}

	public double getLowerBound() {
		double offset = (timeline instanceof DiscreteTimeline)
				? 1.0D / ((DiscreteTimeline) timeline).getTotalRounds() : 0.0D;
		if (conceed)
			minLimit = myReservation;
		return (minLimit
				+ ((0.9f - minLimit) * (1 - myF(timeline.getTime() - offset))));
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
		String agentName = sender != null ? sender.toString() : "null";
		fornullAgent = !fornullAgent;
		if (action != null && action instanceof Offer) {
			Bid newBid = ((Offer) action).getBid();
			lastReceivedBid = ((Offer) action).getBid();
			try {
				if (oppAName != null && oppAName.equals(agentName)) {
					addBidToList(oppABids, newBid);
					currentSlotA++;

				} else if (oppBName != null && oppBName.equals(agentName)) {
					addBidToList(oppBBids, newBid);
					currentSlotB++;

				} else if (oppAName == null) {
					oppAName = agentName;
				} else if (oppBName == null) {
					oppBName = agentName;
				}

				lastBid = newBid;
			} catch (Exception e) {
				// System.out.println("Exception 33 " + e.getMessage());
			}
		} else if (action != null && action instanceof Accept) {
			Bid newBid = lastBid;
			if (oppAName != null && oppAName.equals(agentName)) {
				addBidToList(oppABids, newBid);
				currentSlotA++;

			} else if (oppBName != null && oppBName.equals(agentName)) {
				addBidToList(oppBBids, newBid);
				currentSlotB++;

			} else if (oppAName == null) {
				oppAName = agentName;
			} else if (oppBName == null) {
				oppBName = agentName;
			}
		}
		if (round == slotNum && oppAName != null
				&& oppAName.equals(agentName)) {

			clusterOpponentBid(oppABids, "A", false);
			clusterIsRefreshA = true;
			TimePeriod = new Integer(
					(new Double(timeline.getCurrentTime() / 60)).intValue()
							+ 1);

		}
		if (round == slotNum && oppBName != null
				&& oppBName.equals(agentName)) {

			clusterOpponentBid(oppBBids, "B", false);

			clusterIsRefreshB = true;
			TimePeriod = new Integer(
					(new Double(timeline.getCurrentTime() / 60)).intValue()
							+ 1);
		}

		// Here you can listen to other parties' messages
	}

	public void addBidToList(ArrayList<Bid> mybids, Bid newbid) {
		mybids.add(newbid);
	}

	public float calDist(Bid b1, Bid b2) {
		List<Issue> dissues = utilitySpace.getDomain().getIssues();
		float distance = 0;
		Set<Integer> keys = b1.getValues().keySet();
		Object[] keyCenterVals = b2.getValues().keySet().toArray();
		Object[] keyVals = keys.toArray();
		for (int j = 0; j < dissues.size(); ++j) {

			if (dissues.get(j) instanceof IssueDiscrete) {

				if (!b1.getValues().get(keyVals[j]).toString().equals(
						b2.getValues().get(keyCenterVals[j]).toString())) {
					distance += 1;
				}
			} else {
				float dis = ((Float
						.parseFloat(b1.getValues().get(keyVals[j]).toString()))
						- (Float.parseFloat(b2.getValues().get(keyCenterVals[j])
								.toString())) / dissues.get(j).getChildCount());
				distance += dis < 0 ? dis * -1f : dis;
			}
		}

		return distance;
	}

	public double getMyTargetUtility(float pmin) {

		double offset = (timeline instanceof DiscreteTimeline)
				? 1.0D / ((DiscreteTimeline) timeline).getTotalRounds() : 0.0D;
		return (pmin + ((1.0f - pmin) * (1 - myF(timeline.getTime() - offset))))
				* Math.pow(dFactor, timeline.getTime());
	}

	public double myF(double t) {
		if (getE() == 0.0D)
			return 0.0D;
		else
			return Math.pow(t, 1.0D / getE());
	}

	public double getMyutility() {
		int clusterHistoryASize = 0;
		int clusterHistoryBSize = 0;
		float w1 = 0.4f, w2 = 0.2f, w3 = 0.4f;
		if (clusterIsRefreshA && clusterA != null) {
			ArrayList<Bid> centersA = new ArrayList<Bid>();
			float[][] distA = new float[numberOfclusters][numberOfclusters];
			float unormalA = 0;
			float normal = 1.0f / numberOfclusters;
			for (int i = 0; i < clusterA.size(); ++i) {
				if (clusterA.get(i).getMembers().size() != 0)
					centersA.add(clusterA.get(i).getCenter());

				float temp = normal
						- ((float) clusterA.get(i).getMembers().size())
								/ (float) oppABids.size();
				unormalA += temp < 0 ? -1 * temp : temp;
			}

			float maxA = 0;

			for (int i = 0; i < numberOfclusters && i < centersA.size(); ++i) {
				for (int j = i + 1; j < numberOfclusters
						&& j < centersA.size(); ++j) {
					distA[i][j] = calDist(centersA.get(i), centersA.get(j));
					if (distA[i][j] > maxA)
						maxA = distA[i][j];

				}
			}
			if (clusterHistoryA == null) {
				clusterHistoryA = new ClusterHistory();
				clusterHistoryA.distributionFactor = new ArrayList<Float>();
				clusterHistoryA.submitTime = new ArrayList<Integer>();
				clusterHistoryA.prevCenters = new ArrayList<ArrayList<Bid>>();
				clusterHistoryA.maxDistance = new ArrayList<Float>();
			}

			clusterHistoryA.distributionFactor.add(
					clusterHistoryA.distributionFactor.size(),
					new Float(unormalA));
			clusterHistoryA.maxDistance.add(clusterHistoryA.maxDistance.size(),
					new Float(maxA));
			clusterHistoryA.prevCenters.add(clusterHistoryA.prevCenters.size(),
					new ArrayList<Bid>(centersA));
			clusterHistoryA.submitTime.add(clusterHistoryA.submitTime.size(),
					TimePeriod);
			clusterHistoryASize = clusterHistoryA.distributionFactor.size() - 1;
			for (int i = 0; i < clusterA.size(); ++i) {

				clusterA.get(i).getMembers().clear();
				oppABids.clear();

			}

		}
		if (clusterIsRefreshB && clusterB != null) {
			ArrayList<Bid> centersB = new ArrayList<Bid>();

			float[][] distB = new float[numberOfclusters][numberOfclusters];

			float unormalB = 0;
			float normal = 1.0f / numberOfclusters;

			for (int i = 0; i < clusterB.size(); ++i) {
				if (clusterB.get(i).getMembers().size() != 0)
					centersB.add(clusterB.get(i).getCenter());
				float temp = normal
						- ((float) clusterB.get(i).getMembers().size())
								/ (float) oppBBids.size();
				unormalB += temp < 0 ? -1 * temp : temp;
			}

			float maxB = 0;
			for (int i = 0; i < numberOfclusters && i < centersB.size(); ++i) {
				for (int j = i + 1; j < numberOfclusters
						&& j < centersB.size(); ++j) {
					distB[i][j] = calDist(centersB.get(i), centersB.get(j));
					if (distB[i][j] > maxB)
						maxB = distB[i][j];
				}
			}

			if (clusterHistoryB == null) {
				clusterHistoryB = new ClusterHistory();
				clusterHistoryB.distributionFactor = new ArrayList<Float>();
				clusterHistoryB.prevCenters = new ArrayList<ArrayList<Bid>>();
				clusterHistoryB.maxDistance = new ArrayList<Float>();
				clusterHistoryB.submitTime = new ArrayList<Integer>();
			}
			clusterHistoryB.distributionFactor.add(
					clusterHistoryB.distributionFactor.size(),
					new Float(unormalB));
			clusterHistoryB.maxDistance.add(clusterHistoryB.maxDistance.size(),
					new Float(maxB));
			clusterHistoryB.prevCenters.add(clusterHistoryB.prevCenters.size(),
					new ArrayList<Bid>(centersB));
			clusterHistoryB.submitTime.add(clusterHistoryB.submitTime.size(),
					TimePeriod);
			clusterHistoryBSize = clusterHistoryB.distributionFactor.size() - 1;
			for (int i = 0; i < clusterB.size(); ++i) {
				clusterB.get(i).getMembers().clear();
				oppBBids.clear();
			}
		}
		if (clusterIsRefreshA && clusterA != null) {
			clusterIsRefreshA = false;

			ArrayList<Bid> tempArr = new ArrayList<Bid>();
			for (int j = 0; j < clusterHistoryA.prevCenters
					.get(clusterHistoryASize).size(); ++j) {
				centsA.add(clusterHistoryA.prevCenters.get(clusterHistoryASize)
						.get(j));
				tempArr.add(clusterHistoryA.prevCenters.get(clusterHistoryASize)
						.get(j));
			}

			clusterOpponentBidWithoutFirstLevel(tempArr, centsA, "A", false);
			float unormalA = 0;

			float normal = 1.0f / numberOfclusters;
			for (int i = 0; i < clusterA.size(); ++i) {

				float temp = normal
						- ((float) clusterA.get(i).getMembers().size())
								/ (float) centsA.size();
				unormalA += temp < 0 ? -1 * temp : temp;
			}
			float maxA = 0;
			float[][] distA = new float[numberOfclusters][numberOfclusters];
			for (int i = 0; i < numberOfclusters; ++i) {
				for (int j = i + 1; j < numberOfclusters; ++j) {
					distA[i][j] = calDist(clusterA.get(i).getCenter(),
							clusterA.get(j).getCenter());
					if (distA[i][j] > maxA)
						maxA = distA[i][j];

				}
			}
			float meanDistributionA = 0f;
			float maxCenterDistanceA = 0f;
			int issueSize = utilitySpace.getDomain().getIssues().size();
			maxCenterDistanceA = maxA / issueSize;
			meanDistributionA = unormalA
					/ (1.0f + (numberOfclusters - 2) * normal);

			w1 = 0.6f;
			w3 = 0.4f;
			float ca = (w1 * maxCenterDistanceA - w3 * meanDistributionA);

			ca = ca < 0 ? 0 : ca;
			concessionA = concessionA > ca ? concessionA : ca;
			float temp;
			if (conceed)
				temp = (concessionA + concessionB) / 2.0f;
			else
				temp = concessionA > concessionB ? concessionB : concessionA;
			myReservation = 1.0f - temp < myReservation ? 1.0f - temp
					: myReservation; // for
										// bilateral
			myConcessionRate = temp > myConcessionRate ? temp
					: myConcessionRate;
			double tempConstant = getMyTargetUtility(myReservation);
			if (tempConstant < ConstantUtility)
				ConstantUtility = tempConstant;
			for (int i = 0; i < clusterA.size(); ++i) {
				clusterA.get(i).getMembers().clear();
			}

		}
		if (clusterIsRefreshB && clusterB != null) {
			clusterIsRefreshB = false;
			ArrayList<Bid> tempArr = new ArrayList<Bid>();
			for (int j = 0; j < clusterHistoryB.prevCenters
					.get(clusterHistoryBSize).size(); ++j) {
				centsB.add(clusterHistoryB.prevCenters.get(clusterHistoryBSize)
						.get(j));
				tempArr.add(clusterHistoryB.prevCenters.get(clusterHistoryBSize)
						.get(j));
			}
			clusterOpponentBidWithoutFirstLevel(tempArr, centsB, "B", false);
			float unormalA = 0;

			float normal = 1.0f / numberOfclusters;
			for (int i = 0; i < clusterB.size(); ++i) {

				float temp = normal
						- ((float) clusterB.get(i).getMembers().size())
								/ (float) centsB.size();
				unormalA += temp < 0 ? -1 * temp : temp;
			}
			float maxA = 0;
			float[][] distA = new float[numberOfclusters][numberOfclusters];
			for (int i = 0; i < numberOfclusters; ++i) {
				for (int j = i + 1; j < numberOfclusters; ++j) {
					distA[i][j] = calDist(clusterB.get(i).getCenter(),
							clusterB.get(j).getCenter());
					if (distA[i][j] > maxA)
						maxA = distA[i][j];

				}
			}
			float meanDistributionB = 0f;
			float maxCenterDistanceB = 0f;

			int issueSize = utilitySpace.getDomain().getIssues().size();
			maxCenterDistanceB = maxA / issueSize;
			meanDistributionB = unormalA
					/ (1.0f + (numberOfclusters - 2) * normal);
			w1 = 0.6f;
			w3 = 0.4f;
			float cb = (w1 * maxCenterDistanceB - w3 * meanDistributionB);
			cb = cb < 0 ? 0 : cb;
			concessionB = concessionB > cb ? concessionB : cb;
			float temp;
			if (conceed)
				temp = (concessionA + concessionB) / 2.0f;
			else
				temp = concessionA > concessionB ? concessionB : concessionA;
			myReservation = 1.0f - temp < myReservation ? 1.0f - temp
					: myReservation;
			myConcessionRate = temp > myConcessionRate ? temp
					: myConcessionRate;
			double tempConstant = getMyTargetUtility(myReservation);
			if (tempConstant < ConstantUtility)
				ConstantUtility = tempConstant;
			for (int i = 0; i < clusterB.size(); ++i) {
				clusterB.get(i).getMembers().clear();
			}

		}
		return ConstantUtility;
	}

	@Override
	public double getE() {
		return myConcessionRate;
	}

	public boolean selectEndNegotiation(double time) {
		return rValue * Math.pow(dFactor, time) >= getMyutility();
	}

}
