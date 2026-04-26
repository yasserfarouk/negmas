package agents.anac.y2016.pars2;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.DeadlineType;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.Offer;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.timeline.Timeline;
import genius.core.utility.AdditiveUtilitySpace;
import genius.core.utility.UtilitySpace;
import negotiator.parties.AbstractTimeDependentNegotiationParty;

/**
 * This is ParsAgent party.
 */
public class ParsAgent2 extends AbstractTimeDependentNegotiationParty {
	Bid lastBid;
	Bid myLastBid;
	Bid myBestBid;
	String lastAction;
	String oppAName;
	String oppBName;
	boolean isLastRound;
	double myutility;
	double ConstantUtility = 0.8d;
	boolean Imfirst = false;
	long prevTime = 0;
	Boolean withDiscount = null;
	boolean fornullAgent = false;
	long beginTime;
	int round;

	ArrayList<BidUtility> opponentAB = new ArrayList<BidUtility>();
	OpponentPreferences oppAPreferences = new OpponentPreferences();
	OpponentPreferences oppBPreferences = new OpponentPreferences();
	ArrayList<Bid> oppABids = new ArrayList<Bid>();
	ArrayList<Bid> oppBBids = new ArrayList<Bid>();
	ArrayList<Bid> rejectedBid = new ArrayList<Bid>();
	ArrayList<Cluster> clusterA;
	ArrayList<Cluster> clusterB;
	ClusterHistory clusterHistoryA;
	ClusterHistory clusterHistoryB;
	int numberOfclusters = 3;
	// long timePeriod=30;
	int slotNum = 100;
	double myConcessionRate = 0.15d;
	int currentSlotA;
	int currentSlotB;
	boolean clusterIsRefreshA;
	boolean clusterIsRefreshB;
	float concessionA;
	float concessionB;

	public ParsAgent2() {
		myutility = 0.80000000000000004D;
		Imfirst = false;
		withDiscount = null;
		fornullAgent = false;
		opponentAB = new ArrayList();
		oppAPreferences = new OpponentPreferences();
		oppBPreferences = new OpponentPreferences();
		rejectedBid = new ArrayList<Bid>();
		beginTime = System.currentTimeMillis();
		prevTime = beginTime;
		lastAction = "offer";

	}

	public static void main(String[] arg) {
		ParsAgent2 p = new ParsAgent2();
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
				// }
				for (int i = 0; i < clusters.size(); ++i) {
					// ArrayList<Bid> removable = clusters.get(i).getMembers();
					if (clusters.get(i).getMembers() != null)
						// clusters.get(i).getMembers().removeAll(removable);
						clusters.get(i).getMembers().clear();
				}
			}
			for (int i = 0; i < bidHistory.size(); ++i) {
				defineCluster(clusters, bidHistory.get(i));
			}

			if (refreshClusterCenters(clusters)) {

				for (int i = 0; i < clusters.size(); ++i) {
					// ArrayList<Bid> removable = clusters.get(i).getMembers();
					// clusters.get(i).getMembers().removeAll(removable);
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

	class ClusterHistory {
		ArrayList<ArrayList<Bid>> prevCenters;
		ArrayList<Float> maxDistance;
		ArrayList<Float> distributionFactor;
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
	public ParsAgent2(UtilitySpace utilitySpace,
			Map<DeadlineType, Object> deadlines, Timeline timeline,
			long randomSeed) {
		// Make sure that this constructor calls it's parent.
		super();
		/*
		 * Comment for Bug
		 * 
		 * Object[] keys = deadlines.keySet().toArray(); for (int i = 0; i <
		 * keys.length; ++i) { if (keys[i] instanceof DeadlineType) {
		 * DeadlineType type=(DeadlineType) keys[i]; round =
		 * Integer.parseInt(deadlines.get(keys[i]).toString()); } }
		 */
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
	public Action chooseAction(List validActions) {
		++round;
		lastAction = "offer";
		try {
			if (withDiscount == null)
				withDiscount = utilitySpace.isDiscounted();
			if (rejectedBid.size() > 10)
				rejectedBid.remove(0);
			if (myBestBid == null)
				myBestBid = utilitySpace.getMaxUtilityBid();
			long roundLong = System.currentTimeMillis() - prevTime;
			prevTime = System.currentTimeMillis();
			roundLong = roundLong == 0 ? 1 : roundLong;
			long remainTime = (3 * 60 * 1000)
					- (System.currentTimeMillis() - beginTime);

			isLastRound = Math.round(remainTime / roundLong) <= 10;
			if (isLastRound)
				ConstantUtility = ((0.7d + (remainTime / roundLong) / 10));
			if (currentSlotA > slotNum) {

				currentSlotA = 0;
				clusterOpponentBid(oppABids, "A", false);
				clusterIsRefreshA = true;
			}
			if (currentSlotB > slotNum) {
				currentSlotB = 0;
				clusterOpponentBid(oppBBids, "B", false);
				clusterIsRefreshB = true;
			}

			if (lastBid == null) {
				Imfirst = true;
				Bid b = getMybestBid(myBestBid, 0);
				myLastBid = b;
				lastBid = myLastBid;
				return new Offer(getPartyId(), b);
			} else if (utilitySpace.getUtility(lastBid) >= getMyutility()) {
				lastAction = "Accept";
				return new Accept(getPartyId(), lastBid);
			} else {
				Bid b = offerMyNewBid();
				if (b == null || utilitySpace.getUtility(b) < getMyutility()) {
					b = getMybestBid(myBestBid, 0);
				}

				if (isLastRound && prob(b) < 0.5) {
					b = bestMutualCenters();

					if (utilitySpace.getUtility(b) >= getMyutility()) {
						myLastBid = b;
						lastBid = myLastBid;
						return new Offer(getPartyId(), b);
					}
				} else {
					myLastBid = b;
					lastBid = myLastBid;
					return new Offer(getPartyId(), b);
				}
			}

		} catch (Exception e) {
			e.printStackTrace();
		}
		Bid mb = null;
		try {
			mb = getMybestBid(myBestBid, 0);
			myLastBid = mb;
			lastBid = myLastBid;
			return new Offer(getPartyId(), mb);
		} catch (Exception e) {
			lastAction = "Accept";
			return new Accept(getPartyId(), lastBid);
		}

	}

	public Bid bestMutualCenters() {
		float min = 0f;
		int centerindex = 0;
		int centerindex2 = 0;
		for (int i = 0; i < clusterA.size(); ++i) {
			for (int j = 0; j < clusterB.size(); ++j) {
				float temp = calDist(clusterA.get(i).getCenter(),
						clusterB.get(j).getCenter());
				if (temp < min) {
					min = temp;
					centerindex = i;
					centerindex2 = j;
				}
			}
		}
		if (utilitySpace.getUtility(
				clusterA.get(centerindex).getCenter()) > utilitySpace
						.getUtility(clusterB.get(centerindex2).getCenter()))
			return clusterA.get(centerindex).getCenter();
		return clusterB.get(centerindex2).getCenter();

	}

	public float prob(Bid mybid) {
		float prob = 0f;
		float sum = 0f;
		int issueNum = utilitySpace.getDomain().getIssues().size();
		if (oppABids.size() != 0) {
			for (int i = 0; i < oppABids.size(); ++i) {
				sum += calDist(mybid, oppABids.get(i));
			}
			float probA = (sum / oppABids.size()) / issueNum;
			sum = 0f;
			for (int i = 0; i < oppBBids.size(); ++i) {
				sum += calDist(mybid, oppBBids.get(i));
			}
			float probB = (sum / oppBBids.size()) / issueNum;
			prob = (probA + probB) / 2;
		}
		return prob;
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
			if (!lastAction.equals("Accept"))
				rejectedBid.add(rejectedBid.size(), myLastBid);
			Bid newBid = ((Offer) action).getBid();
			withDiscount = utilitySpace.isDiscounted();

			BidUtility opBid;
			try {
				opBid = new BidUtility(newBid, utilitySpace.getUtility(newBid),
						System.currentTimeMillis());
				if (oppAName != null && oppAName.equals(agentName)) {
					// addBidToList(oppAPreferences.getOpponentBids(), opBid);
					addBidToList(oppABids, newBid);
					currentSlotA++;

				} else if (oppBName != null && oppBName.equals(agentName)) {
					// addBidToList(oppBPreferences.getOpponentBids(), opBid);
					addBidToList(oppBBids, newBid);
					currentSlotB++;

				} else if (oppAName == null) {
					oppAName = agentName;
					// oppAPreferences.getOpponentBids().add(opBid);
				} else if (oppBName == null) {
					oppBName = agentName;
					// oppBPreferences.getOpponentBids().add(opBid);
				}
				calculateParamForOpponent((oppAName.equals(agentName)
						? oppAPreferences : oppBPreferences), newBid);

				lastBid = newBid;
			} catch (Exception e) {
				// System.out.println("Exception 33 " + e.getMessage());
			}
		} else if (action != null && action instanceof Accept) {
			BidUtility opBid = null;
			if (!lastBid.equals(myLastBid)) {
				try {

					opBid = new BidUtility(lastBid,
							utilitySpace.getUtility(lastBid),
							System.currentTimeMillis());
					addBidToList(opponentAB, opBid);

				} catch (Exception e) {
					// System.out.println("Exception 44" + e.getMessage());
				}
			}
		}

		// Here you can listen to other parties' messages
	}

	/*
	 * public int MyBestValue(int issueindex) { ArrayList<Issue> dissues =
	 * utilitySpace.getDomain().getIssues(); Issue isu =
	 * dissues.get(issueindex); HashMap map = new HashMap(); double maxutil =
	 * 0d; int maxvalIndex = 0; try { map = myBestBid.getValues(); } catch
	 * (Exception e) { System.out.println("Exception 3323  " + e.getMessage());
	 * } if (isu instanceof IssueDiscrete) { IssueDiscrete is =
	 * (IssueDiscrete)isu; for (int num = 0; num < is.getNumberOfValues();
	 * ++num) { map.put(new Integer(issueindex + 1), is.getValue(num)); Bid
	 * temp; double u = 0d; try { temp = new Bid(utilitySpace.getDomain(), map);
	 * u = utilitySpace.getUtility(temp); } catch (Exception e) {
	 * System.out.println("Exception 98989  " + e.getMessage()); } if (u >
	 * maxutil) { maxutil = u; maxvalIndex = num; } break;
	 * 
	 * 
	 * } } return maxvalIndex; }
	 */

	public Bid offerMyNewBid() {
		Bid bidNN = null;
		boolean loop = false;
		try {
			if (opponentAB != null && opponentAB.size() != 0) {
				bidNN = getNNBid(opponentAB);

			}

			if (bidNN == null
					|| utilitySpace.getUtility(bidNN) < getMyutility()) {

				ArrayList isues = getMutualIssues();
				HashMap map = new HashMap();
				Bid bid;
				List<Issue> dissues = utilitySpace.getDomain().getIssues();
				boolean hasnotNull = false;
				for (int i = 0; i < isues.size(); ++i) {
					if (isues.get(i) != null)
						hasnotNull = true;
				}
				if (hasnotNull) {
					for (int i = 0; i < isues.size(); ++i) {
						ArrayList keyVal = (ArrayList) isues.get(i);
						if (keyVal != null
								&& dissues.get(i) instanceof IssueDiscrete) {
							IssueDiscrete is = (IssueDiscrete) dissues.get(i);
							for (int num = 0; num < is
									.getNumberOfValues(); ++num) {
								if (is.getValue(num).toString()
										.equals(keyVal.get(0).toString())) {
									map.put(new Integer(i + 1),
											is.getValue(num));
									break;
								}

							}

						} else if (keyVal == null
								&& dissues.get(i) instanceof IssueDiscrete) {
							IssueDiscrete is = (IssueDiscrete) dissues.get(i);
							if (!loop)
								map.put(new Integer(i + 1), myBestBid
										.getValues().get(new Integer(i + 1)));
							else
								map.put(new Integer(i + 1),
										getRandomValue(dissues.get(i)));

						} else if (keyVal != null) {
							map.put(new Integer(i + 1), keyVal.get(0));
						}

					}

					try {
						bid = new Bid(utilitySpace.getDomain(),
								(HashMap) map.clone());

						if (utilitySpace.getUtility(bid) > getMyutility()) {
							if (rejectedBid.contains(bid) && !loop) {
								loop = true;
							} else
								return bid;
						} else {
							return getMybestBid(myBestBid, 0);
						}
					} catch (Exception e) {
						// System.out.println("Exception 55 " + e.getMessage());
					}
				}
			} else
				return bidNN;
		} catch (Exception e) {
			// System.out.println("Exception 121212 == " + e.getMessage());
			e.printStackTrace();
		}
		return getMybestBid(myBestBid, 0);
	}

	public ArrayList getMutualIssues() {
		ArrayList mutualList = new ArrayList();
		List<Issue> dissues = utilitySpace.getDomain().getIssues();
		int twocycle = 2;
		while (twocycle > 0) {
			mutualList = new ArrayList();
			for (int i = 0; i < dissues.size(); ++i) {
				if (oppAPreferences.getRepeatedissue()
						.get(dissues.get(i).getName()) != null) {
					HashMap vals = (HashMap) oppAPreferences.getRepeatedissue()
							.get(dissues.get(i).getName());
					HashMap valsB = (HashMap) oppBPreferences.getRepeatedissue()
							.get(dissues.get(i).getName());
					Object[] keys = vals.keySet().toArray();
					int[] max = new int[] { 0, 0 };
					Object[] maxkey = new Object[] { null, null };
					for (int j = 0; j < keys.length; ++j) {
						Integer temp = (Integer) vals.get(keys[j]);
						if (temp.intValue() > max[0]) {
							max[0] = temp.intValue();
							maxkey[0] = keys[j];
						} else if (temp.intValue() > max[1]) {
							max[1] = temp.intValue();
							maxkey[1] = keys[j];
						}
					}
					if (valsB != null) {
						Object[] keysB = valsB.keySet().toArray();
						int[] maxB = new int[] { 0, 0 };
						;
						Object[] maxkeyB = new Object[] { null, null };
						for (int j = 0; j < keysB.length; ++j) {
							Integer temp = (Integer) valsB.get(keysB[j]);
							if (temp.intValue() > maxB[0]) {
								maxB[0] = temp.intValue();
								maxkeyB[0] = keysB[j];
							} else if (temp.intValue() > maxB[1]) {
								maxB[1] = temp.intValue();
								maxkeyB[1] = keysB[j];
							}
						}

						if (twocycle == 2) {
							if (maxkey[0] != null && maxkeyB[0] != null
									&& maxkey[0].equals(maxkeyB[0])) {
								ArrayList l = new ArrayList();
								l.add(maxkey[0]);
								l.add(maxB[0]);
								l.add(max[0]);
								mutualList.add(i, l);
							} else
								mutualList.add(i, null);
						} else {
							boolean insideloop = true;
							for (int m = 0; insideloop && m < 2; ++m) {
								for (int z = 0; insideloop && z < 2; ++z) {
									if (maxkey[m] != null && maxkeyB[z] != null
											&& maxkey[m].equals(maxkeyB[z])) {
										ArrayList l = new ArrayList();
										l.add(maxkey[m]);
										l.add(maxB[z]);
										l.add(max[m]);
										mutualList.add(i, l);
										insideloop = false;
									}
								}
							}
							if (insideloop)
								mutualList.add(i, null);

						}
					} else {
						mutualList.add(i, null);
						oppBPreferences.getRepeatedissue()
								.put(dissues.get(i).getName(), new HashMap());
					}
				} else {
					oppAPreferences.getRepeatedissue()
							.put(dissues.get(i).getName(), new HashMap());
					mutualList.add(i, null);
				}
				if (((HashMap) oppAPreferences.getRepeatedissue()
						.get(dissues.get(i).getName())).size() == 0
						|| ((HashMap) oppAPreferences.getRepeatedissue()
								.get(dissues.get(i).getName())).size() == 0) {
					twocycle--;
				}
			}
			if (twocycle != 0) {
				twocycle--;
				float nullval = 0.0f;
				for (int i = 0; i < mutualList.size(); ++i) {
					if (mutualList.get(i) != null) {
						++nullval;
					}
				}
				nullval = nullval / mutualList.size();
				if (nullval >= 0.5)
					twocycle--;

			}
		}
		return mutualList;
	}

	public Bid getNNBid(ArrayList<BidUtility> oppAB) {
		List<Issue> dissues = utilitySpace.getDomain().getIssues();
		ArrayList<Bid> maxBids = new ArrayList<Bid>();
		ArrayList<Bid> maximumBids = new ArrayList<Bid>();

		int size = 0;

		Bid newBid;

		int loop = Math.round((float) Math.sqrt(dissues.size()));
		for (int i = 0; i < 5; ++i) {
			if (oppAB != null) {
				if (oppAB.size() > i) {
					maximumBids.add(i, oppAB.get(i).getBid());
				}
			} else
				return null;
		}
		// while (exloop < dissues.size()) {

		Bid maximumBid = null;
		maximumBid = myBestBid;

		size = 0;
		// while (oppAB != null && oppAB.size() > size) {
		for (int i = 0; i < 5; ++i) {
			int loop2 = loop;
			while (loop2 >= 0 && maximumBids.size() > i) {
				int bi = chooseBestIssue();
				// }
				Bid b = maximumBids.get(i);

				try {
					HashMap vals = b.getValues();

					if (utilitySpace.getUtility(b) > getMyutility()) {
						int index = 0;
						for (int k = 0; k < maxBids.size(); ++k) {
							if (utilitySpace
									.getUtility(maxBids.get(k)) < utilitySpace
											.getUtility(b)) {
								index = k;
								break;
							}
						}
						maxBids.add(index, new Bid(b));
						// maxutility = utilitySpace.getUtility(maxBid);
						// loop = -1;
					}
					if (maximumBid != null)
						vals.put(bi, maximumBid.getValue(bi)); // vals.put(bi,
																// getRandomValue(dissues.get(bi
																// - 1)));
					else
						vals.put(bi, getRandomValue(dissues.get(bi - 1)));

					newBid = new Bid(utilitySpace.getDomain(), vals);

					if (utilitySpace.getUtility(newBid) >= getMyutility()) {
						int index = 0;
						for (int k = 0; k < maxBids.size(); ++k) {
							if (utilitySpace
									.getUtility(maxBids.get(k)) < utilitySpace
											.getUtility(newBid)) {
								index = k;
								break;
							}
						}
						maxBids.add(index, new Bid(newBid));
						// maxutility = utilitySpace.getUtility(maxBid);
						// loop = -1;
					}

				} catch (Exception e) {
					// System.out.println("Exception 66 " + e.getMessage());
					e.printStackTrace();
				}

				loop2--;
			}

		}
		Bid maxBid = null;
		float maxProb = 0;
		int index = -1;
		double rand = Math.random();

		if (maxBids.size() != 0) {
			loop = Math.round((float) (rand * maxBids.size()));
			loop = loop == 0 ? 1 : loop;
			for (int i = 0; i < loop; i++) {
				if (!rejectedBid.contains(maxBids.get(i))
						&& prob(maxBids.get(i)) > maxProb) {
					maxProb = prob(maxBids.get(i));
					index = i;
				}
			}
			if (index != -1)
				return new Bid(maxBids.get(index));
		}
		return maxBid;
	}

	public int chooseBestIssue() {
		double random = Math.random();
		double sumWeight = 0d;
		AdditiveUtilitySpace utilitySpace1 = (AdditiveUtilitySpace) utilitySpace;

		for (int i = utilitySpace.getDomain().getIssues().size(); i > 0; --i) {
			sumWeight += utilitySpace1.getWeight(i);
			if (sumWeight > random)
				return i;
		}
		return 0;
	}

	public int chooseWorstIssue() {
		double random = Math.random() * 100;
		AdditiveUtilitySpace utilitySpace1 = (AdditiveUtilitySpace) utilitySpace;
		double sumWeight = 0d;
		int minin = 1;
		double min = 1.0d;
		for (int i = utilitySpace.getDomain().getIssues().size(); i > 0; --i) {
			sumWeight += 1.0d / utilitySpace1.getWeight(i);
			if (utilitySpace1.getWeight(i) < min) {
				min = utilitySpace1.getWeight(i);
				minin = i;
			}
			if (sumWeight > random)
				return i;
		}
		return minin;
	}

	public Bid getMybestBid(Bid sugestBid, int time) {
		List<Issue> dissues = utilitySpace.getDomain().getIssues();
		Bid newBid = new Bid(sugestBid);
		int index = chooseWorstIssue();
		boolean loop = true;
		long bidTime = System.currentTimeMillis();
		while (loop) {
			if ((System.currentTimeMillis() - bidTime) / 1000 > 3)
				break;
			newBid = new Bid(sugestBid);
			try {
				HashMap map = newBid.getValues();
				map.put(index, getRandomValue(dissues.get(index - 1)));
				newBid = new Bid(utilitySpace.getDomain(), map);
				if (utilitySpace.getUtility(newBid) > getMyutility()) {
					return newBid;
				}
			} catch (Exception e) {
				// System.out.println("Exception in my best bid : " +
				// e.getMessage());
				e.printStackTrace();
				loop = false;
			}
		}
		return sugestBid;
	}

	public void addBidToList(ArrayList<BidUtility> mybids, BidUtility newbid) {
		int index = mybids.size();
		for (int i = 0; i < mybids.size(); ++i) {
			if (mybids.get(i).getUtility() <= newbid.getUtility()) {
				// if (!mybids.get(i).getBid().equals(newbid.getBid()))
				index = i;
				break;
				// else
				// return;
			}
		}
		mybids.add(index, newbid);

	}

	public void addBidToList(ArrayList<Bid> mybids, Bid newbid) {

		mybids.add(newbid);

	}

	public void calculateParamForOpponent(OpponentPreferences op, Bid bid) {
		List<Issue> dissues = utilitySpace.getDomain().getIssues();
		HashMap bidVal = bid.getValues();
		Object[] keys = bidVal.keySet().toArray();

		for (int i = 0; i < dissues.size(); ++i) {
			if (op.getRepeatedissue().get(dissues.get(i).getName()) != null) {

				HashMap vals = (HashMap) op.getRepeatedissue()
						.get(dissues.get(i).getName());

				try {
					if (vals.get(bidVal.get(keys[i])) != null) {
						Integer repet = (Integer) vals.get(bidVal.get(keys[i]));
						repet = repet + 1;
						vals.put(bidVal.get(keys[i]), repet);
					} else {
						vals.put(bidVal.get(keys[i]), new Integer(1));
					}
				} catch (Exception e) {
					// System.out.println("Exception 88 " + e.getMessage());
				}
			} else {
				HashMap h = new HashMap();
				// op.getRepeatedissue().get(dissues.get(i).getName());
				try {

					h.put(bidVal.get(keys[i]), new Integer(1));
				} catch (Exception e) {
					// System.out.println("Exception 99 " + e.getMessage());
				}
				op.getRepeatedissue().put(dissues.get(i).getName(), h);
			}
		}

	}

	public void setMyutility(double myutility) {
		this.myutility = myutility;
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

	public double getMyutility() {
		myutility = getTargetUtility();
		if (clusterIsRefreshA && clusterA != null) {
			clusterIsRefreshA = false;
			ArrayList<Bid> centersA = new ArrayList<Bid>();

			float[][] distA = new float[numberOfclusters][numberOfclusters];

			float unormalA = 0;

			float normal = 1.0f / numberOfclusters;
			for (int i = 0; i < clusterA.size(); ++i) {
				centersA.add(clusterA.get(i).getCenter());
				float temp = normal
						- ((float) clusterA.get(i).getMembers().size())
								/ (float) oppABids.size();
				unormalA += temp < 0 ? -1 * temp : temp;
			}

			float maxA = 0;

			for (int i = 0; i < numberOfclusters; ++i) {
				for (int j = i + 1; j < numberOfclusters; ++j) {
					// System.out.println("CenterA i "+i
					// +" "+centersA.get(i));
					// System.out.println("CenterA j "+j
					// +" "+centersA.get(j));
					distA[i][j] = calDist(centersA.get(i), centersA.get(j));
					if (distA[i][j] > maxA)
						maxA = distA[i][j];

				}
			}
			// System.out.println("max distance A ============================ "
			// +
			// maxA + " unormal A " + unormalA);
			// System.out.println("max distance B ============================ "
			// +
			// maxB + " unormal B " + unormalB);
			if (clusterHistoryA == null) {
				clusterHistoryA = new ClusterHistory();
				clusterHistoryA.distributionFactor = new ArrayList<Float>();
				clusterHistoryA.prevCenters = new ArrayList<ArrayList<Bid>>();
				clusterHistoryA.maxDistance = new ArrayList<Float>();
			}

			clusterHistoryA.distributionFactor.add(
					clusterHistoryA.distributionFactor.size(),
					new Float(unormalA));
			clusterHistoryA.maxDistance.add(clusterHistoryA.maxDistance.size(),
					new Float(maxA));
			clusterHistoryA.prevCenters.add(clusterHistoryA.prevCenters.size(),
					new ArrayList(centersA));

			for (int i = 0; i < clusterA.size(); ++i) {

				clusterA.get(i).getMembers().clear();
				oppABids.clear();

			}
			float myConcession = 0f;
			float meanDistributionA = 0f;
			float meanDistanceA = 0f;
			float maxCenterDistanceA = 0f;
			float[][] centerdis = new float[clusterHistoryA.distributionFactor
					.size()][clusterHistoryA.distributionFactor.size()];
			for (int i = 0; i < clusterHistoryA.distributionFactor
					.size(); ++i) {
				meanDistributionA += clusterHistoryA.distributionFactor.get(i);
				meanDistanceA += clusterHistoryA.maxDistance.get(i);
				for (int j = i + 1; j < clusterHistoryA.distributionFactor
						.size(); ++j) {

					float temp1 = 0f;
					float max = 0f;
					float max1 = 0f;
					for (int k = 0; k < numberOfclusters; ++k) {
						for (int l = 0; l < numberOfclusters; ++l) {
							temp1 = calDist(
									clusterHistoryA.prevCenters.get(i).get(k),
									clusterHistoryA.prevCenters.get(j).get(l));
							if (temp1 > max1)
								max1 = temp1;
						}
					}
					centerdis[i][j] = max;

					if (maxCenterDistanceA < max1)
						maxCenterDistanceA = max1;
				}

				// }
			}
			int issueSize = utilitySpace.getDomain().getIssues().size();
			maxCenterDistanceA /= issueSize;
			meanDistributionA = (meanDistributionA
					/ clusterHistoryA.distributionFactor.size()) / 1.33f;
			meanDistanceA = (meanDistanceA
					/ clusterHistoryA.distributionFactor.size()) / issueSize;

			float w1 = 0.4f, w2 = 0.2f, w3 = 0.4f;
			concessionA = (w1 * maxCenterDistanceA + w2 * meanDistanceA
					- w3 * meanDistributionA);
			myConcession = concessionA < concessionB ? concessionA
					: concessionB;
			myConcessionRate = myConcession > 0 && myConcession > 0.15d
					? myConcession : 0.15d;
		} else if (clusterIsRefreshB && clusterB != null) {

			clusterIsRefreshB = false;

			ArrayList<Bid> centersB = new ArrayList<Bid>();

			float[][] distB = new float[numberOfclusters][numberOfclusters];

			float unormalB = 0;
			float normal = 1.0f / numberOfclusters;

			for (int i = 0; i < clusterB.size(); ++i) {
				centersB.add(clusterB.get(i).getCenter());
				// System.out.println("Cluster B members === " + i + " "
				// +
				// (clusterB.get(i).getMembers().size()));
				float temp = normal
						- ((float) clusterB.get(i).getMembers().size())
								/ (float) oppBBids.size();
				unormalB += temp < 0 ? -1 * temp : temp;
			}

			float maxB = 0;
			for (int i = 0; i < numberOfclusters; ++i) {
				for (int j = i + 1; j < numberOfclusters; ++j) {
					// System.out.println("CenterA i "+i
					// +" "+centersA.get(i));
					// System.out.println("CenterA j "+j
					// +" "+centersA.get(j));

					distB[i][j] = calDist(centersB.get(i), centersB.get(j));
					// System.out.println("CenterB i "+i
					// +" "+centersB.get(i));
					// System.out.println("CenterB j "+j
					// +" "+centersB.get(j));
					if (distB[i][j] > maxB)
						maxB = distB[i][j];
				}
			}
			// System.out.println("max distance A ============================ "
			// +
			// maxA + " unormal A " + unormalA);
			// System.out.println("max distance B ============================ "
			// +
			// maxB + " unormal B " + unormalB);

			if (clusterHistoryB == null) {
				clusterHistoryB = new ClusterHistory();
				clusterHistoryB.distributionFactor = new ArrayList<Float>();
				clusterHistoryB.prevCenters = new ArrayList<ArrayList<Bid>>();
				clusterHistoryB.maxDistance = new ArrayList<Float>();
			}

			clusterHistoryB.distributionFactor.add(
					clusterHistoryB.distributionFactor.size(),
					new Float(unormalB));
			clusterHistoryB.maxDistance.add(clusterHistoryB.maxDistance.size(),
					new Float(maxB));
			clusterHistoryB.prevCenters.add(clusterHistoryB.prevCenters.size(),
					new ArrayList(centersB));
			for (int i = 0; i < clusterB.size(); ++i) {

				// ArrayList<Bid> removable = clusterA.get(i).getMembers();
				// clusterA.get(i).getMembers().removeAll(removable);
				clusterB.get(i).getMembers().clear();
				// removable = clusterB.get(i).getMembers();
				// clusterB.get(i).getMembers().removeAll(removable);

				oppBBids.clear();

			}
			float myConcession = 0f;
			float meanDistributionB = 0f;
			float meanDistanceB = 0f;
			float maxCenterDistanceB = 0f;

			float[][] centerdis = new float[clusterHistoryB.distributionFactor
					.size()][clusterHistoryB.distributionFactor.size()];
			for (int i = 0; i < clusterHistoryB.distributionFactor
					.size(); ++i) {
				meanDistributionB += clusterHistoryB.distributionFactor.get(i);

				meanDistanceB += clusterHistoryB.maxDistance.get(i);

				for (int j = i + 1; j < clusterHistoryB.distributionFactor
						.size(); ++j) {

					float temp = 0f;
					float temp1 = 0f;
					float max = 0f;
					float max1 = 0f;
					for (int k = 0; k < numberOfclusters; ++k) {
						for (int l = 0; l < numberOfclusters; ++l) {
							temp = calDist(
									clusterHistoryB.prevCenters.get(i).get(k),
									clusterHistoryB.prevCenters.get(j).get(l));
							if (temp > max)
								max = temp;

						}
					}
					centerdis[i][j] = max;
					if (maxCenterDistanceB < max)
						maxCenterDistanceB = max;

				}

				// }
			}
			int issueSize = utilitySpace.getDomain().getIssues().size();

			maxCenterDistanceB /= issueSize;
			meanDistributionB = (meanDistributionB
					/ clusterHistoryB.distributionFactor.size()) / 1.33f;
			meanDistanceB = (meanDistanceB
					/ clusterHistoryB.distributionFactor.size()) / issueSize;

			// System.out.println("for Opponent A All
			// ===============================meanDistributionA == "
			// +
			// meanDistributionA + " meanDistanceA ==" +
			// meanDistanceA + " maxCenterDistanceA==" +
			// maxCenterDistanceA);
			// System.out.println("for Opponent B All
			// ===============================meanDistributionB == "
			// +
			// meanDistributionB + " meanDistanceB ==" +
			// meanDistanceB + " maxCenterDistanceB==" +
			// maxCenterDistanceB);
			float w1 = 0.4f, w2 = 0.2f, w3 = 0.4f;
			// System.out.println("concession A =========== " +
			// (w1 * maxCenterDistanceA + w2 * meanDistanceA -
			// w3 * meanDistributionA));
			// System.out.println("concession B =========== " +
			// (w1 * maxCenterDistanceB + w2 * meanDistanceB -
			// w3 * meanDistributionB));

			concessionB = (w1 * maxCenterDistanceB + w2 * meanDistanceB
					- w3 * meanDistributionB);
			myConcession = concessionA < concessionB ? concessionA
					: concessionB;
			myConcessionRate = myConcession > 0 && myConcession > 0.15d
					? myConcession : 0.15d;
		}

		// {
		if (myutility < ConstantUtility)
			return ConstantUtility;
		else
			return myutility;
	}

	@Override
	public double getE() {
		if (withDiscount)
			return myConcessionRate + 0.05d;
		return myConcessionRate;
	}

	private class OpponentPreferences {
		private HashMap repeatedissue = new HashMap();
		private ArrayList selectedValues;
		ArrayList<BidUtility> opponentBids = new ArrayList<BidUtility>();

		public void setRepeatedissue(HashMap repeatedissue) {
			this.repeatedissue = repeatedissue;
		}

		public HashMap getRepeatedissue() {
			return repeatedissue;
		}

		public void setSelectedValues(ArrayList selectedValues) {
			this.selectedValues = selectedValues;
		}

		public ArrayList getSelectedValues() {
			return selectedValues;
		}

		public void setOpponentBids(
				ArrayList<ParsAgent2.BidUtility> opponentBids) {
			this.opponentBids = opponentBids;
		}

		public ArrayList<ParsAgent2.BidUtility> getOpponentBids() {
			return opponentBids;
		}
	}

	private class BidUtility {
		private Bid bid;
		private double utility;
		private long time;

		BidUtility(Bid b, double u, long t) {
			this.bid = b;
			this.utility = u;
			this.time = t;
		}

		BidUtility(BidUtility newbid) {
			this.bid = newbid.getBid();
			this.utility = newbid.getUtility();
			this.time = newbid.getTime();
		}

		public void setBid(Bid bid) {
			this.bid = bid;
		}

		public Bid getBid() {
			return bid;
		}

		public void setUtility(double utility) {
			this.utility = utility;
		}

		public double getUtility() {
			return utility;
		}

		public void setTime(long time) {
			this.time = time;
		}

		public long getTime() {
			return time;
		}
	}

	@Override
	public String getDescription() {
		return "ANAC2016";
	}

}
