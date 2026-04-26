/*
 * Author: Max W. Y. Lam (Aug 1 2015)
 * Version: Milestone 1
 * 
 * */

package agents.anac.y2016.maxoops;

import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Offer;
import genius.core.timeline.TimeLineInfo;
import genius.core.utility.AbstractUtilitySpace;
import genius.core.utility.AdditiveUtilitySpace;

public class DMComponent {

	MaxOops agent;
	TimeLineInfo timeline;
	AbstractUtilitySpace utilitySpace;

	public OPTComponent bidsOpt;

	public DMComponent(MaxOops agent, AbstractUtilitySpace utilitySpace,
			TimeLineInfo timeline) {
		this.agent = agent;
		this.utilitySpace = utilitySpace;
		this.timeline = timeline;
		this.bidsOpt = new OPTComponent(agent,
				(AdditiveUtilitySpace) utilitySpace);
	}

	public boolean termination() {
		double time = timeline.getTime();
		double lastBidUtil = utilitySpace.getUtility(agent.lastBid);
		if (agent.delta == 1)
			return false;
		if (time >= 0.9985 + agent.delta * 0.0008 && lastBidUtil <= agent.theta) {
			return true;
		}
		if (agent.TFC.lf_slope < agent.theta
				/ Math.pow(agent.delta, timeline.getTime() * agent.theta)) {
			return true;
		}
		return false;
	}

	public boolean acceptance() {
		double time = timeline.getTime();
		double f_thre = agent.TFC.thresholdFunc();
		if (time >= 0.9985 + agent.delta * 0.0008)
			return true;
		int conflict = (agent.opponentsMaxNumDistinctBids - agent.opponentsMinNumDistinctBids)
				+ (agent.opponentsMaxNumDistinctAccepts - agent.opponentsMinNumDistinctAccepts);
		if (conflict < agent.numParties
				&& agent.prevLastAction instanceof Offer) {
			return false;
		}
		if (time > 0.5 && agent.myLastAction instanceof Accept) {
			return false;
		} else {
			double lastBidUtil = utilitySpace.getUtility(agent.lastBid);
			if (agent.prevLastAction instanceof Accept) {
				lastBidUtil /= Math.pow(agent.delta * 0.95, agent.delta * time);
			}
			if (time < agent.delta * 0.9) {
				if (Math.min(f_thre * Math.pow(agent.delta, time / 5.),
						agent.uqUtil + agent.stdUtil) <= lastBidUtil) {
					return true;
				}
				return false;
			}
			if (f_thre * Math.pow(agent.delta, time / 3.) <= lastBidUtil) {
				return true;
			}
			return false;
		}
	}

	public Bid bidProposal() {
		Bid bid = null;
		double f_thre = agent.TFC.thresholdFunc();
		try {
			bid = bidsOpt.getOptimalBidByThredhold(f_thre + (1.5 - agent.delta)
					* agent.stdUtil);
		} catch (Exception e) {
			System.out.println((new StringBuilder(
					"getOptimalBidByThredhold failed!!\n")));
			e.printStackTrace(System.out);
			e.printStackTrace();
			return null;
		}
		return bid;
	}

}
