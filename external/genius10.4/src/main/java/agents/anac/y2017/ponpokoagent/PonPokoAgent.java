package agents.anac.y2017.ponpokoagent;

import java.util.List;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Random;
import java.util.Set;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.Domain;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.Offer;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.IssueInteger;
import genius.core.issue.IssueReal;
import genius.core.issue.ValueDiscrete;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;
import genius.core.utility.UtilitySpace;

public class PonPokoAgent extends AbstractNegotiationParty {

	private Bid lastReceivedBid = null;
	private Domain domain = null;

	private List<BidInfo> lBids;

	private double threshold_low = 0.99;
	private double threshold_high = 1.0;

	private final int PATTERN_SIZE = 5;
	private int pattern = 0;

	@Override
	public void init(NegotiationInfo info) {

		super.init(info);
		this.domain = getUtilitySpace().getDomain();

		lBids = new ArrayList<>(AgentTool.generateRandomBids(this.domain, 30000,
				this.rand, this.utilitySpace));
		Collections.sort(lBids, new BidInfoComp().reversed());

		pattern = rand.nextInt(PATTERN_SIZE);
	}

	@Override
	public Action chooseAction(List<Class<? extends Action>> validActions) {

		// 譲歩度合いの設定
		if (pattern == 0) {
			threshold_high = 1 - 0.1 * timeline.getTime();
			threshold_low = 1 - 0.1 * timeline.getTime()
					- 0.1 * Math.abs(Math.sin(this.timeline.getTime() * 40));
		} else if (pattern == 1) {
			threshold_high = 1;
			threshold_low = 1 - 0.22 * timeline.getTime();
		} else if (pattern == 2) {
			threshold_high = 1 - 0.1 * timeline.getTime();
			threshold_low = 1 - 0.1 * timeline.getTime()
					- 0.15 * Math.abs(Math.sin(this.timeline.getTime() * 20));
		} else if (pattern == 3) {
			threshold_high = 1 - 0.05 * timeline.getTime();
			threshold_low = 1 - 0.1 * timeline.getTime();
			if (timeline.getTime() > 0.99) {
				threshold_low = 1 - 0.3 * timeline.getTime();
			}
		} else if (pattern == 4) {
			threshold_high = 1 - 0.15 * this.timeline.getTime()
					* Math.abs(Math.sin(this.timeline.getTime() * 20));
			threshold_low = 1 - 0.21 * this.timeline.getTime()
					* Math.abs(Math.sin(this.timeline.getTime() * 20));
		} else {
			threshold_high = 1 - 0.1 * timeline.getTime();
			threshold_low = 1
					- 0.2 * Math.abs(Math.sin(this.timeline.getTime() * 40));
		}

		// Accept判定
		if (lastReceivedBid != null) {
			if (getUtility(lastReceivedBid) > threshold_low) {
				return new Accept(getPartyId(), lastReceivedBid);
			}
		}

		// Offerするbidの選択
		Bid bid = null;
		while (bid == null) {
			bid = AgentTool.selectBidfromList(this.lBids, this.threshold_high,
					this.threshold_low);
			if (bid == null) {
				threshold_low -= 0.0001;
			}
		}
		return new Offer(getPartyId(), bid);
	}

	@Override
	public void receiveMessage(AgentID sender, Action action) {
		super.receiveMessage(sender, action);
		if (action instanceof Offer) {
			lastReceivedBid = ((Offer) action).getBid();
		}
	}

	@Override
	public String getDescription() {
		return "ANAC2017";
	}

}

class AgentTool {

	private static Random random = new Random();

	public static Bid selectBidfromList(List<BidInfo> bidInfoList,
			double higerutil, double lowwerutil) {
		List<BidInfo> bidInfos = new ArrayList<>();
		// Wouter #1536 java8 not allowed, changed the code.
		for (BidInfo bidInfo : bidInfoList) {
			if (bidInfo.getutil() <= higerutil
					&& bidInfo.getutil() >= lowwerutil) {
				bidInfos.add(bidInfo);
			}
		}
		if (bidInfos.size() == 0) {
			return null;
		} else {
			return bidInfos.get(random.nextInt(bidInfos.size())).getBid();
		}
	}

	public static Set<BidInfo> generateRandomBids(Domain d, int numberOfBids,
			Random random, UtilitySpace utilitySpace) {
		Set<BidInfo> randombids = new HashSet<>();
		for (int i = 0; i < numberOfBids; i++) {
			Bid b = d.getRandomBid(random);
			randombids.add(new BidInfo(b, utilitySpace.getUtility(b)));
		}
		return randombids;
	}

	public static long getNumberOfPosibleBids(Domain d) {
		List lIssues = d.getIssues();
		if (lIssues.isEmpty()) {
			return 0;
		}
		long lNumberOfPossibleBids = 1;
		for (Iterator it = lIssues.iterator(); it.hasNext();) {
			Issue lIssue = (Issue) it.next();
			if (lIssue instanceof IssueDiscrete) {
				lNumberOfPossibleBids *= ((IssueDiscrete) lIssue)
						.getNumberOfValues();
			} else if (lIssue instanceof IssueInteger) {
				lNumberOfPossibleBids *= ((IssueInteger) lIssue)
						.getNumberOfDiscretizationSteps();
			} else if (lIssue instanceof IssueReal) {
				lNumberOfPossibleBids *= ((IssueReal) lIssue)
						.getNumberOfDiscretizationSteps();
			} else {
				// wtf happened
				return 0;
			}
		}
		return lNumberOfPossibleBids;
	}

	public static List<Bid> getAllPossibleBid2(Domain d) {
		List<Bid> possiblebids = new ArrayList<>();
		List lIssues = d.getIssues();
		int[] count = new int[lIssues.size()];
		for (int i = 0; i < lIssues.size(); i++) {
			Issue issue = (Issue) (lIssues.get(i));
			if (issue instanceof IssueDiscrete) {
				List<ValueDiscrete> lValues = ((IssueDiscrete) issue)
						.getValues();
				lValues.get(0).getValue();
			} else if (issue instanceof IssueInteger) {

			} else if (issue instanceof IssueReal) {

			}
		}

		return null;
	}

	// 可能なbidをすべて列挙する.
	public static List<Bid> getAllPossibleBids(Domain dom) {
		List allpossiblebids = new ArrayList<Bid>();
		List lIssues = dom.getIssues();
		int[] count = new int[lIssues.size()];
		Arrays.fill(count, 0);
		int cur = 0;
		while (cur != lIssues.size()) {
			HashMap e = new HashMap();
			for (int i = 0; i < lIssues.size(); i++) {
				e.put(Integer.valueOf(
						((IssueDiscrete) (lIssues.get(i))).getNumber()),
						((IssueDiscrete) (lIssues.get(i))).getValue(count[i]));
			}

			Bid newbid = new Bid(dom, e);
			allpossiblebids.add(newbid);
			boolean changed = false;
			for (int i = 0; i <= cur && cur != lIssues.size(); i++) {
				if (!changed) {
					if (count[i] < ((IssueDiscrete) (lIssues.get(i)))
							.getValues().size() - 1) {
						count[i]++;
						changed = true;
					} else {
						count[i] = 0;
						if (i == cur) {
							cur++;
						}
					}
				}
			}
		}
		return allpossiblebids;
	}
}

// bidとその効用を保存するクラス
class BidInfo {
	Bid bid;
	double util;

	public BidInfo(Bid b) {
		this.bid = b;
		util = 0.0;
	}

	public BidInfo(Bid b, double u) {
		this.bid = b;
		util = u;
	}

	public void setutil(double u) {
		util = u;
	}

	public Bid getBid() {
		return bid;
	}

	public double getutil() {
		return util;
	}

	// 適当実装
	@Override
	public int hashCode() {
		return bid.hashCode();
	}

	public boolean equals(BidInfo bidInfo) {
		return bid.equals(bidInfo.getBid());
	}

	@Override
	public boolean equals(Object obj) {
		if (obj == null) {
			return false;
		}
		if (obj instanceof BidInfo) {
			return ((BidInfo) obj).getBid().equals(bid);
		} else {
			return false;
		}
	}

}

final class BidInfoComp implements Comparator<BidInfo> {
	BidInfoComp() {
		super();
	}

	@Override
	public int compare(BidInfo o1, BidInfo o2) {
		return Double.compare(o1.getutil(), o2.getutil());
	}
}
