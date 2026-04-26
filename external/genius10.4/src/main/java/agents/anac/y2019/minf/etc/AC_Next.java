package agents.anac.y2019.minf.etc;

import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import genius.core.boaframework.AcceptanceStrategy;
import genius.core.boaframework.Actions;
import genius.core.boaframework.BOAparameter;
import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.OfferingStrategy;
import genius.core.boaframework.OpponentModel;
import genius.core.timeline.Timeline;

public class AC_Next extends AcceptanceStrategy {

	private double a;
	private double b;

	/** Negotiation Information */
	private NegotiationInfo negotiationInfo;

	public AC_Next(){
	}

	public AC_Next(NegotiationInfo negInfo){
		negotiationInfo = negInfo;
	}

	public AC_Next(NegotiationSession negoSession, OfferingStrategy strat,
			double alpha, double beta) {
		this.negotiationSession = negoSession;
		this.offeringStrategy = strat;
		this.a = alpha;
		this.b = beta;
	}

	@Override
	public void init(NegotiationSession negoSession, OfferingStrategy strat,
			OpponentModel opponentModel, Map<String, Double> parameters)
			throws Exception {
		this.negotiationSession = negoSession;
		this.offeringStrategy = strat;

		if (parameters.get("a") != null || parameters.get("b") != null) {
			a = parameters.get("a");
			b = parameters.get("b");
		} else {
			a = 1;
			b = 0;
		}
	}

	@Override
	public String printParameters() {
		String str = "[a: " + a + " b: " + b + "]";
		return str;
	}

	@Override
	public Actions determineAcceptability() {
		double time = negotiationSession.getTime();
		double lastOpponentBidUtil = negotiationSession.getOpponentBidHistory()
				.getLastBidDetails().getMyUndiscountedUtil();
		double nextMyBidUtil = offeringStrategy.getNextBid().getMyUndiscountedUtil();
		double threshold = negotiationInfo.getThreshold(time);

		if (a * lastOpponentBidUtil + b >= (nextMyBidUtil + threshold)/2.0D) {
			return Actions.Accept;
		}

		if (time >= 0.975 && lastOpponentBidUtil >= 0.9D) {
			return Actions.Accept;
		}

		if (threshold <= negotiationSession.getUtilitySpace().getReservationValueUndiscounted()){
			return Actions.Break;
		}

		if (negotiationSession.getTimeline().getType() == Timeline.Type.Time && time >= 0.999) {
			if (negotiationSession.getUtilitySpace().getReservationValueUndiscounted() <= lastOpponentBidUtil) {
				return Actions.Accept;
			}
		}

		if (negotiationSession.getTimeline().getType() == Timeline.Type.Rounds
				&& negotiationSession.getTimeline().getCurrentTime() == negotiationSession.getTimeline().getTotalTime()-1){
			if (negotiationSession.getUtilitySpace().getReservationValueUndiscounted() <= lastOpponentBidUtil) {
				if(!negotiationInfo.isFirst() ||
						negotiationSession.getOpponentBidHistory().getBestBidDetails().getMyUndiscountedUtil()
						!= negotiationSession.getOpponentBidHistory().getWorstBidDetails().getMyUndiscountedUtil()){
					return Actions.Accept;
				}
			}
		}

		return Actions.Reject;
	}

	@Override
	public Set<BOAparameter> getParameterSpec() {

		Set<BOAparameter> set = new HashSet<BOAparameter>();
		set.add(new BOAparameter("a", 1.0,
				"Accept when the opponent's utility * a + b is greater than the utility of our current bid"));
		set.add(new BOAparameter("b", 0.0,
				"Accept when the opponent's utility * a + b is greater than the utility of our current bid"));

		return set;
	}

	@Override
	public String getName() {
		return "AC_Next";
	}
}
